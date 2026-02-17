from src.models.AE_NODE.training.data_functions import *
import time 
import torch.nn.functional as F
#from torchdiffeq import odeint

class Training_Losses():
    def __init__(self, ae_node_instance):
        self.parent = ae_node_instance
        
    def loss_sup_mixed(self, fields:list, boundary_conditions:tc.tensor, dt:tc.tensor, length_of_padding: tc.tensor, train: bool, loss_dict):
        if not self.parent.all_on_gpu:
            for count, i in enumerate(fields):
                fields[count] = i.to(self.parent.device)
            boundary_conditions = boundary_conditions.to(self.parent.device)
            length_of_padding = length_of_padding.to(self.parent.device)
            
            dt = dt.to(self.parent.device)
        
        original_size = fields[0].size()
        
        #First loss: invertibility of autoencoder enc-dec-enc
        
        if not self.parent.is_AE_frozen:
            definitive_latent, latent_boundaries, l1, regularization_latent = self.auto_encoding_loss(fields, boundary_conditions, length_of_padding, train, loss_dict)
            
        else:
            # AE is frozen - only encode, skip reconstruction
            with tc.no_grad():
                definitive_latent, _, latent_boundaries, regularization_latent = self.parent.encoder(fields, self.parent.is_AE_frozen, boundary_conditions)
            
            if train:
                l1 = [tc.tensor(0.0, device=self.parent.device),  # l1_mean
                    tc.tensor(0.0, device=self.parent.device),  # l1_per_shape (will broadcast)
                    tc.tensor(0.0, device=self.parent.device)]  # l1_latent
            else:
                l1 = [tc.tensor(0.0, device=self.parent.device),  # l1_mean
                    tc.tensor(0.0, device=self.parent.device),  # l1_per_shape
                    tc.tensor(0.0, device=self.parent.device),  # l1_mean_denormalized
                    tc.tensor(0.0, device=self.parent.device),  # l1_mean_denormalized_per_variable
                    tc.tensor(0.0, device=self.parent.device)]  # l1_latent
            
            regularization_latent = tc.tensor(0.0, device=self.parent.device)

        if self.parent.is_coupled[0] == True or (self.parent.is_coupled[0] == False and self.parent.is_coupled[1] == 'NODE'):
            l2_TF, l2_AR_latent, l3, l_full_reconstruction = self.latent_dynamics_loss(fields, definitive_latent, latent_boundaries, length_of_padding, original_size, train, dt, loss_dict)
            
        elif (self.parent.is_coupled[0] == False and self.parent.is_coupled[1] == 'AE'):
            with tc.no_grad():
                l2_TF, l2_AR_latent, l3, l_full_reconstruction = self.latent_dynamics_loss(fields, definitive_latent, latent_boundaries, length_of_padding, original_size, train, dt, loss_dict)

        return l1, l2_TF, l2_AR_latent, l3, l_full_reconstruction, regularization_latent

    def auto_encoding_loss(self, fields:list , boundaries:tc.tensor,length_of_padding:tc.tensor ,train:bool, loss_dict):
        
        definitive_latent, latent_per_shape , latent_boundaries, regularization_latent = self.parent.encoder(fields, self.parent.is_AE_frozen, boundaries)
        reconstructed_variables, reconstructed_latent_per_shape = self.parent.decoder(definitive_latent, self.parent.is_AE_frozen)
        # separate the reconstruction of boundaries and fields
        for count, i in enumerate(reconstructed_variables):
            size = i.size()
            reconstructed_variables[count] = tc.reshape(reconstructed_variables[count], ((fields[0].size()[0],fields[0].size()[1]) + size[1:]))
        
        for count, _ in enumerate(reconstructed_latent_per_shape):
            reconstructed_latent_per_shape[count] = tc.reshape(reconstructed_latent_per_shape[count], (fields[0].size()[0],fields[0].size()[1],-1))
            latent_per_shape[count] = tc.reshape(latent_per_shape[count], (fields[0].size()[0],fields[0].size()[1],-1))
        
        if train:
            l1_mean, l1_per_shape = auto_encoding_MSE(reconstructed_variables, fields, time_series_losses_weights_AR_full_reconstruction = None, length_of_padding = length_of_padding) #field reconstruction
            l1_latent, _ = auto_encoding_MSE(reconstructed_latent_per_shape, latent_per_shape,time_series_losses_weights_AR_full_reconstruction =  None, length_of_padding = length_of_padding)  #latent reconstruction per shape
            l1 = [l1_mean * loss_dict['AE'][0] , l1_per_shape*  loss_dict['AE'][0] , l1_latent * loss_dict['AE'][1]]
        else:
            l1_mean, l1_per_shape = auto_encoding_MSE(reconstructed_variables, fields, time_series_losses_weights_AR_full_reconstruction = None, length_of_padding = length_of_padding) 
            l1_latent, _ = auto_encoding_MSE(reconstructed_latent_per_shape, latent_per_shape,time_series_losses_weights_AR_full_reconstruction =  None, length_of_padding = length_of_padding)  #latent reconstruction per shape
            #reconstructed_variables = standard_and_inverse_normalization_field(reconstructed_variables, self.parent.maxima_or_mean, self.parent.minima_or_std, self.parent.which_normalization, True)
            #fields = standard_and_inverse_normalization_field(fields, self.parent.maxima_or_mean, self.parent.minima_or_std, self.parent.which_normalization, True)
            l1_mean_denormalized, l1_mean_denormalized_per_variable = auto_encoding_MSE(reconstructed_variables, fields,time_series_losses_weights_AR_full_reconstruction = None, length_of_padding = length_of_padding, is_denormalized_validation = False)

            l1 = [l1_mean * loss_dict['AE'][0], l1_per_shape * loss_dict['AE'][0], l1_mean_denormalized * loss_dict['AE'][0], l1_mean_denormalized_per_variable * loss_dict['AE'][0], l1_latent * loss_dict['AE'][1]]
            
        return definitive_latent, latent_boundaries, l1, regularization_latent

    def latent_dynamics_loss(self, fields:list, definitive_latent:tc.tensor, latent_boundaries:tc.tensor, length_of_padding:tc.tensor, original_size:tuple, train:bool, dt:tc.tensor, loss_dict):
        number_batches = original_size[0]
        number_of_time_steps = original_size[1]
        latent_dim = definitive_latent.size()[-1]
        
        definitive_latent = definitive_latent.reshape(number_batches, number_of_time_steps, latent_dim)
        input_processor = definitive_latent[:, 0:-1, :].reshape(number_batches*(number_of_time_steps-1),latent_dim)
        
        dt = tc.reshape(dt[:,0:-1],(number_batches*(number_of_time_steps-1), 1))
        
        latent_boundaries = tc.reshape(latent_boundaries,(number_batches, number_of_time_steps , latent_boundaries.size(-1)))
        latent_boundaries = latent_boundaries[:,0:-1,:].reshape(number_batches * (number_of_time_steps-1) , latent_boundaries.size(-1))
        
        if loss_dict['TF'] <= 0:
            l2_TF = tc.tensor(0.0)
        else:
            e2_latent_TF = self.processor(input_processor, dt, latent_boundaries, 'mine')
            e2_latent_TF = e2_latent_TF.reshape(number_batches, (number_of_time_steps-1), latent_dim)
            l2_TF = dynamics_MSE(e2_latent_TF, definitive_latent[:, 1:, :], None , False, F.relu((length_of_padding-1))) * loss_dict['TF']
        
        if loss_dict['Random_DT'] <= 0:
            l3 = tc.tensor(0.0)
        else:
            random_dt = tc.rand(number_batches*(number_of_time_steps-1),1, device=self.parent.device) * dt
            e2_middle_latent = self.processor( input_processor,random_dt, latent_boundaries, 'mine')
            e2_final = self.processor(e2_middle_latent, dt-random_dt, latent_boundaries, 'mine')
            e2_final = e2_final.reshape(number_batches, (number_of_time_steps-1), latent_dim)
            l3 = dynamics_MSE(e2_final, definitive_latent[:, 1:, :], None, False, F.relu((length_of_padding-1))) * loss_dict['Random_DT']

        if loss_dict['AR_latent'] <= 0.0 or (not (self.parent.is_coupled[0]) and (self.parent.is_coupled[1] == 'AE')):
            l2_AR_latent = tc.tensor(0.0)
            l_full_reconstruction = (tc.tensor(0.0,device=self.parent.device), tc.zeros(self.parent.number_of_different_domains, device = self.parent.device))
        else:
            l2_AR_latent, l_full_reconstruction = self.advance_from_ic(fields, definitive_latent, tc.reshape(dt,(number_batches,number_of_time_steps-1)).unsqueeze(-1), latent_boundaries.reshape(number_batches , (number_of_time_steps-1) , latent_boundaries.size(-1)), length_of_padding, train, loss_dict)
            
        return l2_TF, l2_AR_latent, l3, l_full_reconstruction
             

    def advance_from_ic(self, fields:list, true_latent:tc.tensor, dt:tc.tensor, latent_boundaries:tc.tensor, length_of_padding: tc.tensor, train:bool, loss_dict:dict):
        which_technique = self.parent.autoregressive_step['which_technique']
        number_of_time_steps = fields[0].size(1)
        initial_condition = [tensor[:, 0:1, ...] for tensor in fields]
        B = true_latent.size(0)
        T = number_of_time_steps - 1
        latent_dim = true_latent.size(-1)
        if which_technique == 'fully_autoregressive' or (not train):  #Encode initial condition and evolve in latent. Always done at validation to compute the actual final loss autoregressively
            if loss_dict['full_reconstruction'][0]:
                #fields = standard_and_inverse_normalization_field(fields, self.parent.maxima_or_mean, self.parent.minima_or_std, self.parent.which_normalization, True)
                fields = [tensor[:, 1:, ...] for tensor in fields]
                
            exp_coefficient_time_series_losses_weights_AR_latent, time_series_losses_weights_AR_latent = self.get_time_series_losses_weights( T, loss_dict['AR_latent'], train, self.parent.last_time_series_weigth_AR_latent)
            exp_coefficient_time_series_losses_weights_AR_full_reconstruction, time_series_losses_weights_AR_full_reconstruction = self.get_time_series_losses_weights( T, loss_dict['full_reconstruction'][1], train, self.parent.last_time_series_weigth_AR_full_reconstruction)
            
            if exp_coefficient_time_series_losses_weights_AR_latent is not None:
                self.parent.exp_coefficient_time_series_losses_weights_AR_latent = exp_coefficient_time_series_losses_weights_AR_latent
                
            if exp_coefficient_time_series_losses_weights_AR_full_reconstruction is not None:
                self.parent.exp_coefficient_time_series_losses_weights_AR_full_reconstruction = exp_coefficient_time_series_losses_weights_AR_full_reconstruction
                
            with tc.no_grad():
                next_latent, _, _ , _ = self.parent.encoder(initial_condition, self.parent.is_AE_frozen)
            
            reconstructed_latent = tc.zeros_like(true_latent)[:,1:,:]
            for count in range(number_of_time_steps-1):
                next_latent = self.processor(next_latent, dt[:,count,:], latent_boundaries[:,count,:], self.parent.which_solver[0])
                reconstructed_latent[:,count,:] = next_latent
            
            l2_AR_latent = dynamics_MSE(reconstructed_latent, true_latent[:,1:,:], time_series_losses_weights_AR_latent, True, F.relu((length_of_padding-1)))
            
            if loss_dict['full_reconstruction'][0] and loss_dict['full_reconstruction'][1]!= 0:
                l_full_reconstruction, l_full_reconstruction_per_variable = self.compute_full_reconstruction(reconstructed_latent, B,T, latent_dim, fields, length_of_padding, time_series_losses_weights_AR_full_reconstruction)
                return l2_AR_latent * loss_dict['AR_latent'], (l_full_reconstruction, l_full_reconstruction_per_variable)
            
            elif loss_dict['full_reconstruction'][0] and loss_dict['full_reconstruction'][1]== 0:
                with tc.no_grad():
                    l_full_reconstruction, l_full_reconstruction_per_variable = self.compute_full_reconstruction(reconstructed_latent, B,T, latent_dim, fields, length_of_padding, time_series_losses_weights_AR_full_reconstruction)
                return l2_AR_latent * loss_dict['AR_latent'], (l_full_reconstruction, l_full_reconstruction_per_variable)
            
            elif not loss_dict['full_reconstruction'][0]:
                return l2_AR_latent * loss_dict['AR_latent'], (tc.tensor(0.0), tc.zeros(self.parent.number_of_different_domains, device = self.parent.device))
            else:
                raise TypeError("Something wrong")
        
    def compute_full_reconstruction(self, reconstructed_latent:tc.tensor, B:int,T:int, latent_dim:int, fields:tc.tensor, length_of_padding:tc.tensor, time_series_losses_weights_AR_full_reconstruction:tc.tensor):
        reconstructed_latent = reconstructed_latent.reshape(B * T, latent_dim)
        output_decoder, _ = self.parent.decoder(reconstructed_latent, self.parent.is_AE_frozen)

        output_decoder = [tensor.reshape((B, T) + tensor.size()[1:]) for tensor in output_decoder]
        #output_decoder = standard_and_inverse_normalization_field(output_decoder, self.parent.maxima_or_mean, self.parent.minima_or_std, self.parent.which_normalization, True)xw
        l_full_reconstruction, l_full_reconstruction_per_variable = auto_encoding_MSE(output_decoder, fields, time_series_losses_weights_AR_full_reconstruction, F.relu((length_of_padding-1)), is_denormalized_validation = False) 
        
        return l_full_reconstruction, l_full_reconstruction_per_variable
            
    def processor(self, definitive_latent:tc.tensor, dt:tc.tensor, latent_boundaries:tc.tensor, which_solver:str):
        
        def f_with_boundaries(t, x):
            out = self.parent.f(t, x, latent_boundaries)
            return out
        
        if which_solver == 'mine':
            return self.my_solver(definitive_latent, latent_boundaries, dt)
        
        # common to all odeint-based solvers
        if dt.squeeze().item() == 0.0: #padded step
            return definitive_latent
        T = self.from_dt_to_T(dt)
        
        if which_solver == 'dopri5':
            T = self.from_dt_to_T(dt)
            # cast to float32 for numerical stability of adaptive solver
            predicted_latent = odeint(
                lambda t, x: f_with_boundaries(t, x.float()).half(),
                definitive_latent.float(), 
                T.float(), 
                method='dopri5'
            )
            return predicted_latent[1:].squeeze(0).half()  # cast back to float16
        
        elif which_solver == 'rk4':
            predicted_latent = odeint(f_with_boundaries, definitive_latent, T, method='rk4')
        else:
            raise TypeError('Wrong solver name')
    
        return predicted_latent[1:].squeeze(0)
    
    def get_time_series_losses_weights(self, length_time_series:int, maximum_weight:float, compute:bool, last_time_series_weight:list):
        if not last_time_series_weight[0]:
            return None,None
        
        #increase weight of last time series time step progressively during iteration within a certain window
        if self.parent.index_in_window[-1] >= self.parent.waiting_epochs_before_new_dataset_creation[self.parent.how_many_datasets_creations-1]:
            weight_last_time_step = last_time_series_weight[1] + 1e-8
            exp_coefficient_time_series_losses_weights = 0.0
        else:
            weight_last_time_step = self.parent.index_in_window[-1] / self.parent.waiting_epochs_before_new_dataset_creation[self.parent.how_many_datasets_creations-1] * last_time_series_weight[1] + 1e-8
            exp_coefficient_time_series_losses_weights = - np.log(weight_last_time_step)/(length_time_series-1)
        
        #get exp_coefficient based on index_in_window and last_time_series_weigth_AR_latent decided a priori
        
        if compute:
            indeces = tc.arange(0,length_time_series,1)
            time_series_losses_weights = tc.exp(- indeces * exp_coefficient_time_series_losses_weights
                                                ) * maximum_weight
            time_series_losses_weights = time_series_losses_weights.to(self.parent.device)
            return exp_coefficient_time_series_losses_weights, time_series_losses_weights
        else:
            return None,None
        
    def my_solver(self, state_vector:tc.tensor, conditioning_parameters:tc.tensor, dt:tc.tensor):
        # self.parent.k = 1 is Euler
        sub_dt = dt/self.parent.substep_RK4
        predicted_latent = state_vector
        for i in range(self.parent.substep_RK4):
            predicted_latent = self.my_RK4(predicted_latent, conditioning_parameters, sub_dt)
        
        return predicted_latent
    
    def my_RK4(self, state_vector:tc.tensor, conditioning_parameters:tc.tensor, dt : tc.tensor):
        b = tc.zeros((self.parent.k, state_vector.size(0), state_vector.size(1)) , device= self.parent.device)
        b[0, :,:] = self.parent.f(None, state_vector, conditioning_parameters)
        final_sum = b[0, :,:]*self.parent.RK[str(self.parent.k)][-1][1]

        for i in range(self.parent.k-1):
            s = tc.zeros_like(state_vector, device = self.parent.device)

            for j in range(i+1):
                s +=  b[j] * self.parent.RK[str(self.parent.k)][i+1][j+1]

            b_new = self.parent.f(None, state_vector + dt * s, conditioning_parameters).unsqueeze(0).to(self.parent.device)
            b[i+1,:,:] = b_new

            final_sum += b_new.squeeze(0) * self.parent.RK[str(self.parent.k)][-1][i+2]
        predicted_latent = state_vector + final_sum * dt
        
        return predicted_latent
        
    def from_dt_to_T(self, dt):
        zero = tc.zeros(1, device=self.parent.device)
        return tc.cat([zero, tc.cumsum(dt[:,0].detach(), dim=0)])

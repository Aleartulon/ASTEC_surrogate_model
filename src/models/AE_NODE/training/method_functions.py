from src.models.AE_NODE.training.data_functions import *
import time 
import torch.nn.functional as F

class Training_Losses():
    def __init__(self, training_instance):
        self.__dict__.update(training_instance.__dict__)
        
    def loss_sup_mixed(self, fields:list, boundary_conditions:tc.tensor, dt:tc.tensor, length_of_padding: tc.tensor, loss_coeff:list, train: bool):
        if not self.all_on_gpu:
            for count, i in enumerate(fields):
                fields[count] = i.to(self.device)
            boundary_conditions = boundary_conditions.to(self.device)
            length_of_padding = length_of_padding.to(self.device)
            
            dt = dt.to(self.device)
        
        original_size = fields[0].size()
        
        #First loss: invertibility of autoencoder enc-dec-enc
        
        if self.is_coupled[0] == True or (self.is_coupled[0] == False and self.is_coupled[1] == 'AE'):
            definitive_latent, latent_boundaries, l1, regularization_latent = self.auto_encoding_loss(fields, boundary_conditions, length_of_padding, loss_coeff, train)
            
        elif (self.is_coupled[0] == False and self.is_coupled[1] == 'NODE'):
            with tc.no_grad():
                definitive_latent, latent_boundaries, l1, regularization_latent = self.auto_encoding_loss(fields, boundary_conditions, length_of_padding, loss_coeff, train)

        if self.is_coupled[0] == True or (self.is_coupled[0] == False and self.is_coupled[1] == 'NODE'):
            l2_TF, l2_AR, l3, l_final = self.latent_dynamics_loss(fields, definitive_latent, latent_boundaries, length_of_padding, original_size, train, dt, loss_coeff)
            
        elif (self.is_coupled[0] == False and self.is_coupled[1] == 'AE'):
            with tc.no_grad():
                l2_TF, l2_AR, l3, l_final = self.latent_dynamics_loss(fields, definitive_latent, latent_boundaries, length_of_padding, original_size, train, dt, loss_coeff)

        return l1, l2_TF, l2_AR, l3, l_final, regularization_latent

    def auto_encoding_loss(self, fields:list , boundaries:tc.tensor,length_of_padding:tc.tensor ,loss_coeff:list, train:bool):
        
        definitive_latent, latent_per_shape , latent_boundaries, regularization_latent = self.encoder(fields, boundaries)
        reconstructed_variables, reconstructed_latent_per_shape = self.decoder(definitive_latent)
        
        # separate the reconstruction of boundaries and fields
        for count, i in enumerate(reconstructed_variables):
            size = i.size()
            reconstructed_variables[count] = tc.reshape(reconstructed_variables[count], ((fields[0].size()[0],fields[0].size()[1]) + size[1:]))
        
        for count, _ in enumerate(reconstructed_latent_per_shape):
            reconstructed_latent_per_shape[count] = tc.reshape(reconstructed_latent_per_shape[count], (fields[0].size()[0],fields[0].size()[1],-1))
            latent_per_shape[count] = tc.reshape(latent_per_shape[count], (fields[0].size()[0],fields[0].size()[1],-1))
        
        if train:
            l1_mean, l1_per_shape = auto_encoding_MSE(reconstructed_variables, fields, length_of_padding) #field reconstruction
            l1_latent, _ = auto_encoding_MSE(reconstructed_latent_per_shape, latent_per_shape, length_of_padding)  #latent reconstruction per shape
            l1 = [l1_mean * loss_coeff[0][0] , l1_per_shape*  loss_coeff[0][0] , l1_latent * loss_coeff[0][1]]
        else:
            l1_mean, l1_per_shape = auto_encoding_MSE(reconstructed_variables, fields, length_of_padding) 
            l1_latent, _ = auto_encoding_MSE(reconstructed_latent_per_shape, latent_per_shape, length_of_padding)  #latent reconstruction per shape
            reconstructed_variables = standard_and_inverse_normalization_field(reconstructed_variables, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)
            fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)
            l1_mean_denormalized, l1_mean_denormalized_per_variable = auto_encoding_MSE(reconstructed_variables, fields, length_of_padding, is_denormalized_validation = True)

            l1 = [l1_mean * loss_coeff[0][0], l1_per_shape * loss_coeff[0][0], l1_mean_denormalized * loss_coeff[0][0], l1_mean_denormalized_per_variable * loss_coeff[0][0], l1_latent * loss_coeff[0][1]]
            
        return definitive_latent, latent_boundaries, l1, regularization_latent

    def latent_dynamics_loss(self, fields:list, definitive_latent:tc.tensor, latent_boundaries:tc.tensor, length_of_padding:tc.tensor, original_size:tuple, train:bool, dt:tc.tensor, loss_coeff:list):
        number_batches = original_size[0]
        number_of_time_steps = original_size[1]
        latent_dim = definitive_latent.size()[-1]
        
        definitive_latent = definitive_latent.reshape(number_batches, number_of_time_steps, latent_dim)
        input_processor = definitive_latent[:, 0:-1, :].reshape(number_batches*(number_of_time_steps-1),latent_dim)
        
        dt = tc.reshape(dt[:,0:-1],(number_batches*(number_of_time_steps-1), 1))
        
        latent_boundaries = tc.reshape(latent_boundaries,(number_batches, number_of_time_steps , latent_boundaries.size(-1)))
        latent_boundaries = latent_boundaries[:,0:-1,:].reshape(number_batches * (number_of_time_steps-1) , latent_boundaries.size(-1))
        
        if loss_coeff[1] <= 0:
            l2_TF = tc.tensor(0.0)
        else:
            e2_latent_TF = self.processor_First_Order(input_processor, dt, latent_boundaries)
            e2_latent_TF = e2_latent_TF.reshape(number_batches, (number_of_time_steps-1), latent_dim)
            l2_TF = dynamics_MSE(e2_latent_TF, definitive_latent[:, 1:, :], F.relu((length_of_padding-1))) * loss_coeff[1]
        
        if loss_coeff[3] <= 0:
            l3 = tc.tensor(0.0)
        else:
            random_dt = tc.rand(number_batches*(number_of_time_steps-1),1, device=self.device) * dt
            e2_middle_latent = self.processor_First_Order( input_processor,random_dt, latent_boundaries)
            e2_final = self.processor_First_Order(e2_middle_latent, dt-random_dt, latent_boundaries)
            e2_final = e2_final.reshape(number_batches, (number_of_time_steps-1), latent_dim)
            l3 = dynamics_MSE(e2_final, definitive_latent[:, 1:, :], F.relu((length_of_padding-1))) * loss_coeff[3]

        if loss_coeff[2] <= 0:
            with tc.no_grad():
                l2_AR = tc.tensor(0.0)
                l_final = tc.tensor(0.0)
        else:
            l2_AR, l_final = self.advance_from_ic(fields, definitive_latent, tc.reshape(dt,(number_batches,number_of_time_steps-1)).unsqueeze(-1), latent_boundaries.reshape(number_batches , (number_of_time_steps-1) , latent_boundaries.size(-1)), length_of_padding, loss_coeff[2], train)
            
        return l2_TF, l2_AR, l3, l_final
             

    def advance_from_ic(self, fields:list, true_latent:tc.tensor, dt:tc.tensor, latent_boundaries:tc.tensor, length_of_padding: tc.tensor, loss_coeff:float, train:bool):
        which_technique = self.autoregressive_step['which_technique']
        number_of_time_steps = fields[0].size(1)
        initial_condition = [tensor[:, 0:1, ...] for tensor in fields]
        B = true_latent.size(0)
        T = number_of_time_steps - 1
        latent_dim = true_latent.size(-1)
        
        if not (self.is_coupled[0]) and (self.is_coupled[1] == 'AE') and train:
            return tc.tensor(0.0), tc.tensor(0.0)
        
        elif not (self.is_coupled[0]) and (self.is_coupled[1] == 'AE') and not train:
            return tc.tensor(0.0), (tc.tensor(0.0),tc.tensor(0.0))
            
        if which_technique == 'fully_autoregressive' or (not train):  #Encode initial condition and evolve in latent. Always done at validation to compute the actual final loss autoregressively
            if (not train):
                fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)
                fields = [tensor[:, 1:, ...] for tensor in fields]
                
            reconstructed_latent = tc.zeros_like(true_latent)[:,1:,:]
            
            with tc.no_grad():
                next_latent, _, _ , _ = self.encoder(initial_condition)
            
            for count in range(number_of_time_steps-1):
                
                next_latent = self.processor_First_Order(next_latent, dt[:,count,:], latent_boundaries[:,count,:])
                reconstructed_latent[:,count,:] = next_latent
                
            l2_AR = dynamics_MSE(reconstructed_latent, true_latent[:,1:,:], F.relu((length_of_padding-1)))
            
            if (not train):
                reconstructed_latent = reconstructed_latent.reshape(B * T, latent_dim)
                output_decoder, _ = self.decoder(reconstructed_latent)
 
                output_decoder = [tensor.reshape((B, T) + tensor.size()[1:]) for tensor in output_decoder]
                output_decoder = standard_and_inverse_normalization_field(output_decoder, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)

                l_final, l_final_per_variable = auto_encoding_MSE(output_decoder, fields, F.relu((length_of_padding-1)), is_denormalized_validation = True) 
                
                return l2_AR * loss_coeff, (l_final, l_final_per_variable)
            
            return l2_AR * loss_coeff, tc.tensor(0.0)
            
    def processor_First_Order(self, definitive_latent:tc.tensor, dt:tc.tensor, latent_boudaries:tc.tensor):

        # self.k = 1 is Euler
        b = tc.zeros((self.k, definitive_latent.size(0), definitive_latent.size(1)) , device= self.device)
        b[0, :,:] = self.f(definitive_latent, latent_boudaries)
        final_sum = self.f(definitive_latent, latent_boudaries)*self.RK[str(self.k)][-1][1]

        for i in range(self.k-1):
            mu_in_time = latent_boudaries.clone() #avoid in place operation which messes with backprop.
            s = tc.zeros_like(definitive_latent, device = self.device)

            for j in range(i+1):
                s +=  b[j] * self.RK[str(self.k)][i+1][j+1]

            b_new = self.f(definitive_latent + dt * s, mu_in_time).unsqueeze(0).to(self.device)
            b[i+1,:,:] = b_new

            final_sum += b_new.squeeze(0) * self.RK[str(self.k)][-1][i+2]
        e2 = definitive_latent + final_sum * dt
        return e2
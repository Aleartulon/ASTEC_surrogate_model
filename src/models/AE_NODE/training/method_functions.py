from src.models.AE_NODE.training.data_functions import *
import time 

class Training_Losses():
    def __init__(self, training_instance):
        self.__dict__.update(training_instance.__dict__)
        
    def loss_sup_mixed(self, fields:list, boundary_conditions:tc.tensor, dt:tc.tensor, length_of_padding: tc.tensor, loss_coeff:list, train: bool):
        
        for count, i in enumerate(fields):
            fields[count] = i.to(self.device)
        boundary_conditions = boundary_conditions.to(self.device)
        
        dt = dt.to(self.device)
        
        original_size = fields[0].size()
        
        #First loss: invertibility of autoencoder enc-dec-enc
        
        if self.is_coupled[0] == True or (self.is_coupled[0] == False and self.is_coupled[1] == 'AE'):
            latent_vector_fields, latent_boundaries, l1, regularization_latent = self.auto_encoding_loss(fields, boundary_conditions, length_of_padding, loss_coeff, train)
            
        elif (self.is_coupled[0] == False and self.is_coupled[1] == 'NODE'):
            with tc.no_grad():
                latent_vector_fields, latent_boundaries, l1, regularization_latent = self.auto_encoding_loss(fields, boundary_conditions, length_of_padding, loss_coeff, train)

        if self.is_coupled[0] == True or (self.is_coupled[0] == False and self.is_coupled[1] == 'NODE'):
            l2_TF, l2_AR, l3, l_final = self.latent_dynamics_loss(fields, latent_vector_fields, latent_boundaries, original_size, train, dt, loss_coeff)
        elif (self.is_coupled[0] == False and self.is_coupled[1] == 'AE'):
            with tc.no_grad():
                l2_TF, l2_AR, l3, l_final = self.latent_dynamics_loss(fields, latent_vector_fields, latent_boundaries, original_size, train, dt, loss_coeff)

        return l1, l2_TF, l2_AR, l3, l_final, regularization_latent

    def auto_encoding_loss(self, fields:list , boundaries:tc.tensor,length_of_padding:tc.tensor ,loss_coeff:list, train:bool):
        
        latent_vector_fields, _ , latent_boundaries, regularization_latent = self.encoder(fields, boundaries)
        reconstructed_variables, reconstructed_boundaries = self.decoder(latent_vector_fields, latent_boundaries)
        
        # separate the reconstruction of boundaries and fields
        for count, i in enumerate(reconstructed_variables):
            size = i.size()
            reconstructed_variables[count] = tc.reshape(reconstructed_variables[count], ((fields[0].size()[0],fields[0].size()[1]) + size[1:]))
        reconstructed_boundaries = tc.reshape(reconstructed_boundaries, ((boundaries.size()[0],boundaries.size()[1]) + reconstructed_boundaries.size()[1:]))

        if train:
            l1 = MSE(reconstructed_variables, fields, length_of_padding, reconstructed_boundaries, boundaries) * loss_coeff[0]
        else:
            first = MSE(reconstructed_variables, fields, length_of_padding) * loss_coeff[0]
            
            reconstructed_variables = standard_and_inverse_normalization_field(reconstructed_variables, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)
            reconstructed_boundaries = standard_and_inverse_normalization_field([reconstructed_boundaries], self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)[0]
            
            fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)
            boundaries = standard_and_inverse_normalization_field([boundaries], self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)[0]

            second = MSE(reconstructed_variables, fields, length_of_padding, reconstructed_boundaries, boundaries, is_denormalized_validation = True) * loss_coeff[0]
            l1 = [first, second]
            
        return latent_vector_fields, latent_boundaries, l1, regularization_latent

    def latent_dynamics_loss(self, fields:list, latent_vector_fields:tc.tensor, latent_boundaries:tc.tensor, original_size:tuple, train:bool, dt:tc.tensor, loss_coeff:list):
        number_batches = original_size[0]
        number_of_time_steps = original_size[1]
        latent_dim = latent_vector_fields.size()[-1]
        
        latent_vector_fields = latent_vector_fields.reshape(number_batches, number_of_time_steps, latent_dim)
        input_processor = latent_vector_fields[:, 0:-1, :].reshape(number_batches*(number_of_time_steps-1),latent_dim)
        
        dt = tc.reshape(dt[:,0:-1],(number_batches*(number_of_time_steps-1), 1))
        
        latent_boundaries = tc.reshape(latent_boundaries,(number_batches, number_of_time_steps , latent_boundaries.size(-1)))
        latent_boundaries = latent_boundaries[:,0:-1,:].reshape(number_batches * (number_of_time_steps-1) , latent_boundaries.size(-1))
        
        if loss_coeff[1] <= 0:
            l2_TF = tc.tensor(0.0)
        else:
            e2_latent_TF = self.processor_First_Order(input_processor, dt, latent_boundaries)
            e2_latent_TF = e2_latent_TF.reshape(number_batches, (number_of_time_steps-1), latent_dim)
            l2_TF = self.L2_relative_loss_general(e2_latent_TF, latent_vector_fields[:, 1:, :], True) * loss_coeff[1]
        
        if loss_coeff[3] <= 0:
            l3 = tc.tensor(0.0)
        else:
            random_dt = tc.rand(number_batches*(number_of_time_steps-1),1, device=self.device) * dt
            e2_middle_latent = self.processor_First_Order( input_processor,random_dt, latent_boundaries)
            e2_final = self.processor_First_Order(e2_middle_latent, dt-random_dt, latent_boundaries)
            e2_final = e2_final.reshape(number_batches, (number_of_time_steps-1), latent_dim)
            l3 = self.L2_relative_loss_general(e2_final, latent_vector_fields[:, 1:, :], True) * loss_coeff[3]

        if loss_coeff[2] <= 0:
            with tc.no_grad():
                l2_AR = tc.tensor(0.0)
                l_final = tc.tensor(0.0)
        else:
            l2_AR, l_final = self.advance_from_ic(fields, latent_vector_fields, tc.reshape(dt,(number_batches,number_of_time_steps-1)).unsqueeze(-1), latent_boundaries.reshape(number_batches , (number_of_time_steps-1) , latent_boundaries.size(-1)), loss_coeff[2], train)
            
        return l2_TF, l2_AR, l3, l_final
             

    def advance_from_ic(self, fields:list, true_latent:tc.tensor, dt:tc.tensor, latent_boundaries:tc.tensor, coeff:list, train:bool):
        
        """ this function advances autoregressively the reduced vector at time t = 0 across the whole time series, effectively mimicking what happens at testing time. It is thus used to compute
            L_2^A. We use this function to compute L_2^A and to get the full predicted time series of solution fields (needed during validation to compute a validation metric).
            This function can implement 3 different algorithms depending on the value of self.start_backprop[0]:
            - self.start_backprop[0] = 0: the initial confition is encoded and the latent vectors are predicted autoregressively keeping the gradients since the encoding
            - self.start_backprop[0] = 1: the initial confition is encoded and the latent vectors are predicted autoregressively keeping the gradients since up to the previous self.start_backprop[1] time steps
            - self.start_backprop[0] = 2: the initial confition is encoded and the latent vectors are predicted autoregressively keeping the gradients since up to the previous self.start_backprop[1] time steps,
            where self.start_backprop[1]=1 in the first epoch and is then increased by 1 dynamically every TBPP_dynamic[1] epochs, where TBPP_dynamic is specified in initial_information.yaml.

        Args:
            self.encoder (class 'src.architecture.self.encoder): self.encoder
            self.f (src.architecture.F_Latent): function self.f of the ODE of the latent dynamics
            self.decoder (src.architecture.self.decoder): self.decoder
            input_encoder (torch.Tensor): tensor of dimensions [B,T,C,x_dim_1, dim_x_1, dim_x_2, ...], where B is batch_size, T is the length of the full time series, C is the number of channels of the 
                predicted solution field, dim_x_1, dim_x_2, ... are the dimensions of the first spatial dimension, second spatial dimension, etc. It is the field solution over time
            true_latent (torch.Tensor): tensor of dimensions [B,T,latent_dim], where B is batch_size, T is the length of the full time series and latent_dim is the dimension of the latent space. It is the expected latent vectors
            dt (torch.Tensor): a tensor containing the dts used to advance each snapshot in time. It has dimensions [B, T-1], where B is the batch size and T is the length of the time series. it assumes each batch evolves accordingly to the same dts 
            latent_boundaries (torch.Tensor): tensor of dimension [B, T, num_params], where B is the batch size and T is the length of the time series and num_params the number of parameters of the system
            self.k (int): stage of Runge-Kutta algorithm
            self.RK (dict): dictionary with Butcher tablue for Runge-Kutta algorithms
            self.device (torch.self.device): self.device where the training and validation are done
            self.start_backprop (list): list with values that determine up to which time-step in the past backpropagate the gradients
            size (torch.Size): tensor representing the length of each dimension of the solution field tensor
            coeff (float): coefficient that multiplies the loss function L_2^A
            dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
            train (bool): if true, this function was called inside the training loop, otherwise in the validation loop
            time_dependence_in_f (bool):  if true, the function self.f depends on time as well.

        Returns:
            torch.tensor(), torch.tensor():  the mean L_2^A,  the loss function which takes into account the field solution predicted autoregressively (only at validation)
        """    
        number_batches = fields[0].size(0)
        number_of_time_steps = fields[0].size(1)
        initial_condition = [tensor[:, 0:1, ...] for tensor in fields]
        
        if not (self.is_coupled[0]) and (self.is_coupled[1] == 'AE'):
            return tc.tensor(0.0), tc.tensor(0.0)
            
        if self.start_backprop[0] == 0 or (not train):  #Encode initial condition and evolve in latent
            l_final = tc.tensor(0., device = self.device)
            l2_AR = tc.tensor(0., device = self.device)

            if (number_of_time_steps-1-self.start_backprop[1]) == 0:
                next_latent, _, _ , _ = self.encoder(initial_condition)
            else:
                with tc.no_grad():
                    next_latent, _ , _ , _ = self.encoder(initial_condition)

            if (not train):
                fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)
            
            step = 0
            for count in range(number_of_time_steps-1):
                if count <= (number_of_time_steps-1-self.start_backprop[1]):
                    with tc.no_grad():
                        next_latent = self.processor_First_Order( next_latent, dt[:,count,:], latent_boundaries[:,count,:])
                        l2_AR += self.L2_relative_loss_general(next_latent, true_latent[:,count+1,:], True) 
                        step+=1
                else:
                    next_latent = self.processor_First_Order(next_latent, dt[:,count,:], latent_boundaries[:,count,:])
                    l2_AR += self.L2_relative_loss_general(next_latent, true_latent[:,count+1,:], True) 
                    step+=1
                if (not train):
                    output_decoder, _ = self.decoder(next_latent)
                    denorm_latent = standard_and_inverse_normalization_field(output_decoder, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)
                    fields_at_correct_time_step = [tensor[:, count+1, ...] for tensor in fields]
                    l_final += MSE(denorm_latent, fields_at_correct_time_step) #the boundary should not be taken into account here
            return l2_AR/step * coeff, l_final/(number_of_time_steps-1)
        
        elif self.start_backprop[0] == 1: #Encode ic and evolve in latent but TBPP

            place_holder = tc.zeros((number_batches,number_of_time_steps-self.start_backprop[1], true_latent.size()[-1]), device = self.device)
            next_latent, _, _, _ = self.encoder(input_encoder[:,0,...])
            place_holder[:,0,:] = next_latent.clone()
            with tc.no_grad():
                for i in range(number_of_time_steps-self.start_backprop[1]-1):
                    next_latent = self.processor_First_Order( next_latent, dt[:,i+1,:], latent_boundaries[:,i+1,:])
                    place_holder[:,i+1,:] = next_latent.detach().clone()

            place_holder = tc.reshape(place_holder,(number_batches*(number_of_time_steps-self.start_backprop[1]),true_latent.size()[-1]))
            l2_AR_1 = tc.tensor(0.0,device=self.device)
            next_latent, _, _, _ = self.encoder(input_encoder[:,0,...])
            
            for i in range(self.start_backprop[1]-1):
                next_latent = self.processor_First_Order( next_latent, dt[:,i,:], latent_boundaries[:,i,:])
                l2_AR_1 += self.L2_relative_loss_general(next_latent, true_latent[:,i+1,:], True)

            l2_AR_2 = tc.tensor(0.0,device=self.device)
            for i in range(self.start_backprop[1]):
                place_holder = self.processor_First_Order(place_holder, dt[:,i:number_of_time_steps-self.start_backprop[1]+i,:].flatten().unsqueeze(-1), latent_boundaries[:,i:number_of_time_steps-self.start_backprop[1]+i,:].reshape(-1, latent_boundaries.size(-1)))
            
            place_holder = tc.reshape(place_holder,(number_batches,number_of_time_steps-self.start_backprop[1],true_latent.size()[-1]))
            l2_AR_2 += self.L2_relative_loss_general(place_holder, true_latent[:,self.start_backprop[1]:,:], True)

            return (l2_AR_1 * (self.start_backprop[1]-1) +l2_AR_2 *(number_of_time_steps-self.start_backprop[1]))/(number_of_time_steps-1) * coeff, tc.tensor(0.0)
            
        elif self.start_backprop[0] == 2: # Encode full field self.start_backprop[1] steps in advance and from there TBPP

            l2_AR_1 = tc.tensor(0.0,device=self.device)
            next_latent, _, _, _ = self.encoder(input_encoder[:,0,...])
            for i in range(self.start_backprop[1]-1):
                next_latent = self.processor_First_Order(next_latent, dt[:,i,:], latent_boundaries[:,i,:])
                l2_AR_1 += self.L2_relative_loss_general(next_latent, true_latent[:,i+1,:], True)

            place_holder, _, _, _ = self.encoder(input_encoder)

            for i in range(self.start_backprop[1]): 
                place_holder = self.processor_First_Order( place_holder, dt[:,i:number_of_time_steps-self.start_backprop[1]+i,:].flatten().unsqueeze(-1) , latent_boundaries[:,i:number_of_time_steps-self.start_backprop[1]+i,:].reshape(-1, latent_boundaries.size(-1)))

            place_holder = tc.reshape(place_holder,(number_batches,number_of_time_steps-self.start_backprop[1],true_latent.size()[-1]))
            l2_AR_2 = self.L2_relative_loss_general(place_holder, true_latent[:,self.start_backprop[1]:,:], True)
            
            return (l2_AR_1 * (self.start_backprop[1]-1) +l2_AR_2 *(number_of_time_steps-self.start_backprop[1]))/(number_of_time_steps-1) * coeff, tc.tensor(0.0)
            
    def processor_First_Order(self, latent_vector_fields:tc.tensor, dt:tc.tensor, latent_boudaries:tc.tensor):
        """this function implements the Runge-Kutta algorithms. First_Order refers to the fact that the ODE is a first order ODE, although higher orders would still be solved by this algorithms
        simply introducing new functions.

        Args:
            self.f (src.architecture.F_Latent): function self.f of the ODE of the latent dynamics
            latent_vector_fields (torch.tensor()): tensor of dimension [B, dim_latent], where B is the batch size and dim_latent the dimension of the latent space
            dt (torch.Tensor): a tensor containing the dts used to advance each snapshot in time. It has dimensions [B, T-1], where B is the batch size and T is the length of the time series. it assumes each batch evolves accordingly to the same dts 
            latent_boudaries (tc.tensor()): tensor of dimension [B, num_params] where B is the batch size and num_params the number of parameters of the system
            self.k (int): stage of Runge-Kutta algorithm
            self.RK (dict): dictionary with Butcher tablue for Runge-Kutta algorithms
            self.device (torch.self.device): self.device where the training and validation are done
            time_dependence_in_f (bool):  if true, the function self.f depends on time as well.

        Returns:
            torch.tensor(): tensor of dimension [B, dim_latent] which contains the latent vectors advanced in time from latent_vector_fields of dt
        """    
        # self k=1 is Euler
        b = tc.zeros((self.k, latent_vector_fields.size(0), latent_vector_fields.size(1)) , device= self.device)
        b[0, :,:] = self.f(latent_vector_fields, latent_boudaries )
        final_sum = self.f(latent_vector_fields, latent_boudaries)*self.RK[str(self.k)][-1][1]

        for i in range(self.k-1):
            mu_in_time = latent_boudaries.clone() #avoid in place operation which messes with backprop.
            s = tc.zeros_like(latent_vector_fields, device = self.device)

            for j in range(i+1):
                s +=  b[j] * self.RK[str(self.k)][i+1][j+1]

            b_new = self.f(latent_vector_fields + dt * s, mu_in_time).unsqueeze(0).to(self.device)
            b[i+1,:,:] = b_new

            final_sum += b_new.squeeze(0) * self.RK[str(self.k)][-1][i+2]
        e2 = latent_vector_fields + final_sum * dt
        return e2
    
    
    def L2_relative_loss_general(self, inp:tc.tensor, target:tc.tensor, latent:bool):

        eps = tc.tensor(1e-8)
        if latent:
            #with tc.no_grad():
            norm = tc.sum(target**2, dim=-1, keepdim=True)**0.5
            L2_relative = tc.mean(tc.sum((inp - target)**2,dim=-1, keepdim=True)**0.5 / tc.max(norm, eps))
            return L2_relative
        else:

            if dim_inp > 1:
                inp = inp.flatten(start_dim=-dim_inp)
                target = target.flatten(start_dim=-dim_inp)
            norm = tc.linalg.vector_norm(target, dim=-1)
            L2_relative = tc.mean(tc.linalg.vector_norm(inp - target, dim=-1) / tc.max(norm, eps))
            return L2_relative
        
    
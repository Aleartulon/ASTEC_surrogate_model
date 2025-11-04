from src.models.ONLY_DECODER.training.data_functions import *
import time 
import torch.nn.functional as F

class Training_Losses():
    def __init__(self, training_instance):
        self.__dict__.update(training_instance.__dict__)
        
    def loss_sup_mixed(self, fields:list, boundary_conditions_and_time:tc.tensor, loss_coeff:list, train: bool):
        
        for count, i in enumerate(fields):
            fields[count] = i.to(self.device)
        boundary_conditions_and_time = boundary_conditions_and_time.to(self.device)
        
        #Mapping from time and boundaries into field
        l1 = self.auto_encoding_loss(fields, boundary_conditions_and_time, loss_coeff, train)
        return l1

    def auto_encoding_loss(self, fields:list , boundary_conditions_and_time:tc.tensor, loss_coeff:list, train:bool):
        
        reconstructed_variables, _ = self.decoder(boundary_conditions_and_time)
        
        # separate the reconstruction of boundaries and fields
        for count, i in enumerate(reconstructed_variables):
            size = i.size()
            reconstructed_variables[count] = tc.reshape(reconstructed_variables[count], ((fields[0].size()[0],fields[0].size()[1]) + size[1:]))
            
        if train:
            l1_mean, l1_per_shape = auto_encoding_MSE(reconstructed_variables, fields) #field reconstruction
            l1 = [l1_mean * loss_coeff[0] , l1_per_shape*  loss_coeff[0]]
        else:
            l1_mean, l1_per_shape = auto_encoding_MSE(reconstructed_variables, fields) 
            reconstructed_variables = standard_and_inverse_normalization_field(reconstructed_variables, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)
            fields = standard_and_inverse_normalization_field(fields, self.maxima_or_mean, self.minima_or_std, self.which_normalization, True)
            l1_mean_denormalized, l1_mean_denormalized_per_variable = auto_encoding_MSE(reconstructed_variables, fields, is_denormalized_validation = True)

            l1 = [l1_mean * loss_coeff[0], l1_per_shape * loss_coeff[0], l1_mean_denormalized * loss_coeff[0], l1_mean_denormalized_per_variable * loss_coeff[0]]
            
        return l1

    
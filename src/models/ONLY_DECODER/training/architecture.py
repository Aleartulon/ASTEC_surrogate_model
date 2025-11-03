import numpy as np
import torch as tc
import torch.nn.functional as F
from torch import nn
    
class Decoder(nn.Module):
    def __init__(self, config_training:dict, model_information:dict):
        super().__init__()
        model_information['decoding']['decoding_initial_increase']['output_dimension'] = 0
        for i in model_information['decoding']:
            if i != 'decoding_initial_increase':
                model_information['decoding']['decoding_initial_increase']['output_dimension'] += model_information['decoding'][i]['input_dimension']
                
        self.device = config_training['device']
        self.initial_increase = Fully_Connected_Decoder(config_training, model_information['decoding']['decoding_initial_increase']).to(self.device)
        self.decoder_scalar_variables = Fully_Connected_Decoder(config_training, model_information['decoding']['decoding_scalar']).to(self.device)
        self.decoder_plenum_variables = Fully_Connected_Decoder(config_training, model_information['decoding']['decoding_plenum']).to(self.device)
        self.decoder_core_variables = Convolutional_Decoder(config_training, model_information['decoding']['decoding_core']).to(self.device)
        self.decoder_vessel_variables = Convolutional_Decoder(config_training, model_information['decoding']['decoding_vessel']).to(self.device)
        self.decoder_faces_variables = Convolutional_Decoder(config_training, model_information['decoding']['decoding_faces']).to(self.device)
        self.indeces = self.get_indeces_reconstruction_latent_vectors(model_information)
                
    def get_indeces_reconstruction_latent_vectors(self, model_information:dict):
        indeces = {}
        index = 0
        for i in model_information['decoding']:
            if i != 'decoding_initial_increase':
                indeces[i] = (int( model_information['decoding'][i]['input_dimension'])+index)
                index += int( model_information['decoding'][i]['input_dimension'])
        return indeces
           
    def forward(self, definitive_latent:tc.tensor):
        concatenated_latents = self.initial_increase(definitive_latent)
        latent_scalar_variables = concatenated_latents[:,0:self.indeces['decoding_scalar']]
        latent_plenum_variables = concatenated_latents[:,self.indeces['decoding_scalar']:self.indeces['decoding_plenum']]
        latent_core_variables = concatenated_latents[:,self.indeces['decoding_plenum']:self.indeces['decoding_core']]
        latent_vessel_variables = concatenated_latents[:,self.indeces['decoding_core']:self.indeces['decoding_vessel']]
        latent_faces_variables = concatenated_latents[:,self.indeces['decoding_vessel']:self.indeces['decoding_faces']]

        latent_in_variables_separated = [latent_scalar_variables, latent_plenum_variables, latent_core_variables, latent_vessel_variables, latent_faces_variables] #useful at testing
        reconstructed_scalar_variables = self.decoder_scalar_variables(latent_scalar_variables)
        reconstructed_plenum_variables = self.decoder_plenum_variables(latent_plenum_variables)
        reconstructed_core_variables = self.decoder_core_variables(latent_core_variables)
        reconstructed_vessel_variables = self.decoder_vessel_variables(latent_vessel_variables)
        reconstructed_faces_variables = self.decoder_faces_variables(latent_faces_variables)
        
        return [reconstructed_scalar_variables, reconstructed_core_variables, reconstructed_vessel_variables, reconstructed_plenum_variables , reconstructed_faces_variables], latent_in_variables_separated
    
class Fully_Connected_Decoder(nn.Module):
    def __init__(self, config_training:dict, fully_connected_information:dict):
        super().__init__()
        self.list_of_neurons = fully_connected_information['list_of_neurons_decoder']
        self.number_of_layers = len(self.list_of_neurons)
        self.last_activation = fully_connected_information['last_activation_decoder']
        self.input_dimension = fully_connected_information['input_dimension']
        self.output_dimension = fully_connected_information['output_dimension']
        
        self.first_layer = tc.nn.Linear(self.input_dimension, self.list_of_neurons[0], bias=True)
        self.inner_layers = tc.nn.ModuleList([nn.Linear(self.list_of_neurons[i], self.list_of_neurons[i+1]) for i in range(self.number_of_layers-1)])
        self.last_layer = tc.nn.Linear(self.list_of_neurons[-1], self.output_dimension, bias=True)
        
        self.gelu = nn.GELU()
        self.activation = self.gelu
        
    def forward(self, x:tc.tensor):
        x = self.first_layer(x)
        x = self.activation(x)
        for i in self.inner_layers:
            x = i(x)
            x = self.activation(x)
            
        x = self.last_layer(x)
        
        if self.last_activation:
            x = self.activation(x)
        return x

class Convolutional_Decoder(nn.Module):
    def __init__(self, config_training:dict, convolutional_information:dict):
      
        super().__init__()
        self.first_channel = convolutional_information['first_channel']
        self.filters = np.concatenate([np.array([self.first_channel]), np.array(convolutional_information['filters_decoder']),np.array([int(convolutional_information['output_dimension'][-3])])])
        self.kernel = convolutional_information['kernel_decoder']
        
        if len(self.kernel) != len(self.filters):
            raise TypeError("Length kernel of " + str(convolutional_information['output_dimension'])+ " is not lenght of filters - 2")
        self.strides = tuple(convolutional_information['strides_decoder'])
        self.size_kernel = [(np.array(k) - 1) // 2 for k in self.kernel]  # Adjust padding based on kernel size
        self.initial_shape = convolutional_information['output_dimension']
        
        for i in range(len(self.kernel)):
            self.initial_shape[1] = np.ceil(self.initial_shape[1]/2) if self.strides[i][0] == 2 else self.initial_shape[1]
            self.initial_shape[2] = np.ceil(self.initial_shape[2]/2) if self.strides[i][1] == 2 else self.initial_shape[2]
            
        self.output_dfnn = int(self.initial_shape[1] * self.initial_shape[2] * self.first_channel)
        # Activation function (use only one for now)
        self.gelu = nn.GELU()
        self.activation = self.gelu

        # Convolutional layers and BatchNorm layers
        self.transposed_convolutionals = nn.ModuleList()
        self.len_filters = len(self.filters)
        self.last_activation = convolutional_information['last_activation_decoder']
        
        # Dense fully-connected layer
        self.dfnn = nn.Linear(convolutional_information['input_dimension'], self.output_dfnn, bias=True)
        nn.init.kaiming_uniform_(self.dfnn.weight)
        for i in range(self.len_filters-1):
            self.transposed_convolutionals.extend([nn.ConvTranspose2d(self.filters[i],self.filters[i+1],self.kernel[i],self.strides[i],padding=self.size_kernel[i], output_padding = 0)])
            nn.init.kaiming_uniform_(self.transposed_convolutionals[i].weight)

    def forward(self, x:tc.tensor):
        x = self.dfnn(x)
        if self.last_activation:
            x = self.activation(x) 
            
        x = tc.reshape(x,(x.size()[0], self.first_channel, int(self.initial_shape[-2]), int(self.initial_shape[-1])))
        length = len(self.transposed_convolutionals)
        for i in range(length-1):
            x = self.transposed_convolutionals[i](x)
            x = self.activation(x)
        x = self.transposed_convolutionals[-1](x)
        return x
import numpy as np
import torch as tc
import torch.nn.functional as F
from torch import nn

class Encoder(nn.Module):
    def __init__(self, config_training:dict, model_information:dict):
        super().__init__()
        model_information['auto_encoding']['final_reduction_and_initial_increase']['input_dimension_encoder'] = 0
        for i in model_information['auto_encoding']:
            if i != 'final_reduction_and_initial_increase' and i != 'auto_encoder_boundaries': #boundaries are not processed together with inner variables (?)
                model_information['auto_encoding']['final_reduction_and_initial_increase']['input_dimension_encoder'] += model_information['auto_encoding'][i]['output_dimension_encoder']
                
        self.device = config_training['device']
        self.encoder_scalar_variables = Fully_Connected_Encoder(config_training, model_information['auto_encoding']['auto_encoder_scalar']).to(self.device)
        self.encoder_plenum_variables = Fully_Connected_Encoder(config_training, model_information['auto_encoding']['auto_encoder_plenum']).to(self.device)
        self.encoder_boundaries_variables = Fully_Connected_Encoder(config_training, model_information['auto_encoding']['auto_encoder_boundaries']).to(self.device)
        self.encoder_core_variables = Convolutional_Encoder(config_training, model_information['auto_encoding']['auto_encoder_core']).to(self.device)
        self.encoder_vessel_variables = Convolutional_Encoder(config_training, model_information['auto_encoding']['auto_encoder_vessel']).to(self.device)
        self.encoder_faces_variables = Convolutional_Encoder(config_training, model_information['auto_encoding']['auto_encoder_faces']).to(self.device)
        self.final_reduction = Fully_Connected_Encoder(config_training, model_information['auto_encoding']['final_reduction_and_initial_increase']).to(self.device)
        self.lambda_regularization = model_information['lambda_regularization']
        
    def forward(self, fields:list, boundaries: tc.tensor = None):
        
        if boundaries is not None:
            boundaries_variables = boundaries.flatten(0, 1) 
            latent_boundaries_variables = self.encoder_boundaries_variables(boundaries_variables) #final reduced vector of boundaries
        else:
            latent_boundaries_variables = None
            
        scalar_variables = fields[0].flatten(0, 1) 
        plenum_variables = fields[3].flatten(0, 1) 
        core_variables = fields[1].flatten(0, 1) 
        vessel_variables = fields[2].flatten(0, 1) 
        faces_variables = fields[4].flatten(0, 1) 
        
        latent_scalar_variables = self.encoder_scalar_variables(scalar_variables)
        latent_plenum_variables = self.encoder_plenum_variables(plenum_variables)
        latent_core_variables = self.encoder_core_variables(core_variables)
        latent_vessel_variables = self.encoder_vessel_variables(vessel_variables)
        latent_faces_variables = self.encoder_faces_variables(faces_variables)
        
        #concatenate latent variables without latent_boundaries_variables
        latent_in_variables_separated = [latent_scalar_variables, latent_plenum_variables, latent_core_variables, latent_vessel_variables, latent_faces_variables] #useful at testing
        latent_in_variables = tc.concatenate((latent_scalar_variables, latent_plenum_variables, latent_core_variables, latent_vessel_variables, latent_faces_variables), axis = -1)
        regularization_latent = self.l1_latent_regularization(latent_in_variables, self.lambda_regularization, latent_boundaries_variables) #regularization latent space 
        final_latent = self.final_reduction(latent_in_variables) #final reduced vector of inner fields
        
        
        return final_latent, latent_in_variables_separated, latent_boundaries_variables, regularization_latent
    
    def l1_latent_regularization(self, latent_fields: list, lambda_l1: float, latent_boundaries: tc.tensor = None):
        
        if lambda_l1 != 0:
            if latent_boundaries is not None:
                l1_norm = (tc.mean(tc.abs(latent_fields)) + tc.mean(tc.abs(latent_boundaries)))/2
            else:
                l1_norm = tc.mean(tc.abs(latent_fields))
                
            return lambda_l1 * l1_norm
        else:
            return tc.tensor(0.0, device=latent_fields[0].device)
    
class Fully_Connected_Encoder(nn.Module):
    def __init__(self, config_training:dict, fully_connected_information:dict):
        super().__init__()
        
        self.input_dimension = fully_connected_information['input_dimension_encoder']
        self.output_dimension = fully_connected_information['output_dimension_encoder'] 
        self.list_of_neurons = fully_connected_information['list_of_neurons_encoder']
        self.number_of_layers = len(self.list_of_neurons)
        self.last_activation = fully_connected_information['last_activation_encoder']
        
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

class Convolutional_Encoder(nn.Module):
    def __init__(self, config_training:dict, convolutional_information:dict):
      
        super().__init__()
      
        self.filters = convolutional_information['filters_encoder']
        self.kernel = convolutional_information['kernel_encoder']
        self.dim_input = convolutional_information['input_dimension_encoder']
        self.strides = tuple(convolutional_information['strides_encoder'])
        self.channels = np.concatenate(([self.dim_input[0]], self.filters))
        self.size_kernel = [(k - 1) // 2 for k in self.kernel]  # Adjust padding based on kernel size
        self.output_dfnn = convolutional_information['output_dimension_encoder']
        
        self.input_dfnn = self.dim_input.copy()
        # Activation function (use only one for now)
        self.gelu = nn.GELU()
        self.activation = self.gelu

        # Convolutional layers and BatchNorm layers
        self.convolutionals = nn.ModuleList()
        self.len_kernel = len(self.kernel)
        self.last_activation = convolutional_information['last_activation_encoder']
    
        for i in range(self.len_kernel):
            self.convolutionals.append(nn.Conv2d(self.channels[i], self.filters[i], self.kernel[i], stride=self.strides[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
            nn.init.kaiming_uniform_(self.convolutionals[i].weight)
            self.input_dfnn[0] = self.filters[i]
            self.input_dfnn[1] = np.ceil(self.input_dfnn[1]/2) if self.strides[i][0] == 2 else self.input_dfnn[1]
            self.input_dfnn[2] = np.ceil(self.input_dfnn[2]/2) if self.strides[i][1] == 2 else self.input_dfnn[2]
            
        self.shape_before_mlp_encoder = self.input_dfnn
        convolutional_information['shape_before_mlp_encoder'] = self.shape_before_mlp_encoder
        self.input_dfnn = int(np.prod(self.input_dfnn))
        
        # Dense fully-connected layer
        self.dfnn = nn.Linear(self.input_dfnn, self.output_dfnn, bias=True)
        nn.init.kaiming_uniform_(self.dfnn.weight)

    def forward(self, x:tc.tensor):
        for i, conv_layer in enumerate(self.convolutionals):
            x = conv_layer(x)
            x = self.activation(x)
        x = tc.flatten(x, 1)  # Flatten across the batch dimension
        x = self.dfnn(x)
        if self.last_activation:
            x = self.activation(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, config_training:dict, model_information:dict):
        super().__init__()
        self.device = config_training['device']
        self.initial_increase = Fully_Connected_Decoder(config_training, model_information['auto_encoding']['final_reduction_and_initial_increase']).to(self.device)
        self.decoder_scalar_variables = Fully_Connected_Decoder(config_training, model_information['auto_encoding']['auto_encoder_scalar']).to(self.device)
        self.decoder_plenum_variables = Fully_Connected_Decoder(config_training, model_information['auto_encoding']['auto_encoder_plenum']).to(self.device)
        self.decoder_boundaries_variables = Fully_Connected_Decoder(config_training, model_information['auto_encoding']['auto_encoder_boundaries']).to(self.device)
        self.decoder_core_variables = Convolutional_Decoder(config_training, model_information['auto_encoding']['auto_encoder_core']).to(self.device)
        self.decoder_vessel_variables = Convolutional_Decoder(config_training, model_information['auto_encoding']['auto_encoder_vessel']).to(self.device)
        self.decoder_faces_variables = Convolutional_Decoder(config_training, model_information['auto_encoding']['auto_encoder_faces']).to(self.device)
        self.indeces = self.get_indeces_reconstruction_latent_vectors(model_information)
                
    def get_indeces_reconstruction_latent_vectors(self, model_information:dict):
        indeces = {}
        index = 0
        for i in model_information['auto_encoding']:
            if i != 'final_reduction_and_initial_increase' and i != 'auto_encoder_boundaries': 
                indeces[i] = (int( model_information['auto_encoding'][i]['output_dimension_encoder'])+index)
                index += int( model_information['auto_encoding'][i]['output_dimension_encoder'])
        return indeces
           
    def forward(self, latent_vector_fields:tc.tensor, latent_boundaries: tc.tensor = None):
        
        if latent_boundaries is not None:
            reconstructed_boundaries_variables = self.decoder_boundaries_variables(latent_boundaries)
        else:
            reconstructed_boundaries_variables = None
            
        concatenated_latents = self.initial_increase(latent_vector_fields)
        latent_scalar_variables = concatenated_latents[:,0:self.indeces['auto_encoder_scalar']]
        latent_plenum_variables = concatenated_latents[:,self.indeces['auto_encoder_scalar']:self.indeces['auto_encoder_plenum']]
        latent_core_variables = concatenated_latents[:,self.indeces['auto_encoder_plenum']:self.indeces['auto_encoder_core']]
        latent_vessel_variables = concatenated_latents[:,self.indeces['auto_encoder_core']:self.indeces['auto_encoder_vessel']]
        latent_faces_variables = concatenated_latents[:,self.indeces['auto_encoder_vessel']:self.indeces['auto_encoder_faces']]
        
        reconstructed_scalar_variables = self.decoder_scalar_variables(latent_scalar_variables)
        reconstructed_plenum_variables = self.decoder_plenum_variables(latent_plenum_variables)
        reconstructed_core_variables = self.decoder_core_variables(latent_core_variables)
        reconstructed_vessel_variables = self.decoder_vessel_variables(latent_vessel_variables)
        reconstructed_faces_variables = self.decoder_faces_variables(latent_faces_variables)
        
        return [reconstructed_scalar_variables, reconstructed_core_variables, reconstructed_vessel_variables, reconstructed_plenum_variables , reconstructed_faces_variables], reconstructed_boundaries_variables
    
class Fully_Connected_Decoder(nn.Module):
    def __init__(self, config_training:dict, fully_connected_information:dict):
        super().__init__()
        self.list_of_neurons = fully_connected_information['list_of_neurons_decoder']
        self.number_of_layers = len(self.list_of_neurons)
        self.last_activation = fully_connected_information['last_activation_decoder']
        self.input_dimension = fully_connected_information['output_dimension_encoder']
        self.output_dimension = fully_connected_information['input_dimension_encoder']
        
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
        self.first_channel = convolutional_information['filters_encoder'][-1]
        self.filters = np.concatenate([np.array([self.first_channel]), np.array(convolutional_information['filters_decoder']),np.array([int(convolutional_information['input_dimension_encoder'][-3])])])
        self.kernel = convolutional_information['kernel_decoder']
        if len(self.kernel) != len(self.filters):
            raise TypeError("Length kernel of " + str(convolutional_information['input_dimension_encoder'])+ " is not lenght of filters - 2")
        self.strides = tuple(convolutional_information['strides_decoder'])
        self.size_kernel = [(np.array(k) - 1) // 2 for k in self.kernel]  # Adjust padding based on kernel size
        self.input_dfnn = convolutional_information['output_dimension_encoder']
        self.shape_before_mlp_encoder = convolutional_information['shape_before_mlp_encoder']
        self.output_dfnn = int(np.prod(self.shape_before_mlp_encoder))
        
        # Activation function (use only one for now)
        self.gelu = nn.GELU()
        self.activation = self.gelu

        # Convolutional layers and BatchNorm layers
        self.transposed_convolutionals = nn.ModuleList()
        self.len_filters = len(self.filters)
        self.last_activation = convolutional_information['last_activation_decoder']
        
        # Dense fully-connected layer
        self.dfnn = nn.Linear(self.input_dfnn, self.output_dfnn, bias=True)
        nn.init.kaiming_uniform_(self.dfnn.weight)
        for i in range(self.len_filters-1):
            self.transposed_convolutionals.extend([nn.ConvTranspose2d(self.filters[i],self.filters[i+1],self.kernel[i],self.strides[i],padding=self.size_kernel[i], output_padding = 0)])
            nn.init.kaiming_uniform_(self.transposed_convolutionals[i].weight)

    def forward(self, x:tc.tensor):
        x = self.dfnn(x)
        if self.last_activation:
            x = self.activation(x) 
            
        x = tc.reshape(x,(x.size()[0], self.first_channel, int(self.shape_before_mlp_encoder[-2]), int(self.shape_before_mlp_encoder[-1])))
        length = len(self.transposed_convolutionals)
        for i in range(length-1):
            x = self.transposed_convolutionals[i](x)
            x = self.activation(x)
        x = self.transposed_convolutionals[-1](x)
        return x
    
class F_Latent(nn.Module): 
    def __init__(self, config_training:dict, model_information:dict):
        super().__init__()
        self.param_dim = model_information['auto_encoding']['auto_encoder_boundaries']['output_dimension_encoder']
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.leaky = nn.LeakyReLU()
        self.activation = self.gelu
        self.n_layers = model_information['n_layers_f']
        self.parameter_information = model_information['parameter_information']
        self.n_FiLM_conditioning = model_information['n_FiLM_conditioning']

        self.dropout = nn.Dropout(p=0.5)
        
        self.final_latent_dim = model_information['auto_encoding']['final_reduction_and_initial_increase']['output_dimension_encoder']
        n_neurons = model_information['n_neurons_f']

        if self.parameter_information == 'concatenation':
            if self.n_layers !=1:

                if self.param_dim > 0:
                    self.linears = nn.ModuleList([nn.Linear(self.final_latent_dim + self.param_dim, n_neurons, bias = True)])
                else:
                    self.linears = nn.ModuleList([nn.Linear(self.final_latent_dim, n_neurons, bias = True)])

                self.linears.extend([nn.Linear(n_neurons, n_neurons, bias = True) for i in range(self.n_layers )])
                self.linears.append(nn.Linear(n_neurons, self.final_latent_dim, bias = True))

                for i in self.linears:
                    nn.init.kaiming_uniform_(i.weight)
            else:
                if self.param_dim > 0:
                    self.dfnn = nn.Linear(self.final_latent_dim + self.param_dim, self.final_latent_dim, bias = True)
                    nn.init.kaiming_uniform_(self.dfnn.weight)
                else:
                    self.dfnn = nn.Linear(self.final_latent_dim, self.final_latent_dim, bias = True)
                    nn.init.kaiming_uniform_(self.dfnn.weight)


        elif parameter_information == 'FiLM':

            if self.param_dim > 0:
                self.param_FiLM_gamma = nn.ModuleList([nn.Linear(self.param_dim, self.final_latent_dim, bias = True)])
                self.param_FiLM_beta = nn.ModuleList([nn.Linear(self.param_dim, self.final_latent_dim, bias = True)])
                self.param_FiLM_gamma.extend([nn.Linear(self.param_dim, n_neurons, bias = True) for i in range(n_FiLM_conditioning)])
                self.param_FiLM_beta.extend([nn.Linear(self.param_dim, n_neurons, bias = True) for i in range(n_FiLM_conditioning)])

            self.linears = nn.ModuleList([nn.Linear(self.final_latent_dim, n_neurons, bias = True)])
            self.linears.extend([nn.Linear(n_neurons, n_neurons, bias = True) for i in range(self.n_layers )])
            self.linears.append(nn.Linear(n_neurons, self.final_latent_dim, bias = True))        

            for i in self.linears:
                nn.init.kaiming_uniform_(i.weight)
        else:
            raise ValueError("Wrong name of the type of parameter information")

    def forward(self, x:tc.tensor, parameter:tc.tensor):
        """forward pass of the f function, which takes as input latent vectors x and parameters parameter. It takes (T-1) snapshots, all of them besides the last one to predict the next one

        Args:
            x (torch.tensor): a tensor of latent vectors of dimension [B*(T-1), final_latent_dim], where B is batch size, T is the len of the time series and latent dim is the dimension of the latent space.
            parameter (torch.tensor): a tensor of dimension [B*(T-1), dim_param], where B is batch size, T is the len of the time series and dim_param is the number of parameters.

        Returns:
            torch.tensor: a tensor of latent vectors of dimension [B*T, final_latent_dim], where B is batch size, T is the len of the time series and latent dim is the dimension of the latent space.
            It is the output of the function f.
        """        
        
        if self.parameter_information == 'concatenation':
            if self.param_dim > 0:
                x = tc.cat((x, parameter), dim=1)
            
            for count, i in enumerate(self.linears[0:-1]):
                x = i(x)
                x = self.activation(x) 
            x = self.linears[-1](x)
            return x

        elif self.parameter_information == 'FiLM':
            if self.param_dim > 0:
                parameter_vector_gamma = self.param_FiLM_gamma[0](parameter)
                parameter_vector_beta = self.param_FiLM_beta[0](parameter)
                x = parameter_vector_gamma * x + parameter_vector_beta

            for count, i in enumerate(self.linears[0:-1]):
                x = i(x)
                if (count+1) < self.n_FiLM_conditioning:
                    parameter_vector_gamma = self.param_FiLM_gamma[count+1](parameter)
                    parameter_vector_beta = self.param_FiLM_beta[count+1](parameter)
                    x = parameter_vector_gamma * x + parameter_vector_beta
                x = self.activation(x) 
                
            x = self.linears[-1](x)
            return x

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
        self.encoder_scalar_variables = Fully_Connected_Encoder(config_training, model_information['auto_encoding']['auto_encoder_scalar'])
        self.encoder_plenum_variables = Fully_Connected_Encoder(config_training, model_information['auto_encoding']['auto_encoder_plenum'])
        self.encoder_boundaries_variables = Fully_Connected_Encoder(config_training, model_information['auto_encoding']['auto_encoder_boundaries'])
        self.encoder_core_variables = Convolutional_Encoder(config_training, model_information['auto_encoding']['auto_encoder_core'])
        self.encoder_vessel_variables = Convolutional_Encoder(config_training, model_information['auto_encoding']['auto_encoder_vessel'])
        self.encoder_faces_variables = Convolutional_Encoder(config_training, model_information['auto_encoding']['auto_encoder_faces'])
        self.final_reduction = Fully_Connected_Encoder(config_training, model_information['auto_encoding']['final_reduction_and_initial_increase'])
        self.lambda_regularization = model_information['loss_coefficients']['lambda_regularization'] if model_information['is_coupled'][0] else model_information['loss_coefficients_not_coupled']['lambda_regularization'] 
        
    def forward(self, fields:list, is_AE_frozen: bool, boundaries: tc.tensor = None):
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
        latent_in_variables_separated = [latent_scalar_variables, latent_plenum_variables, latent_core_variables, latent_vessel_variables, latent_faces_variables, latent_boundaries_variables] #useful at testing
        latent_in_variables = tc.concatenate((latent_scalar_variables, latent_plenum_variables, latent_core_variables, latent_vessel_variables, latent_faces_variables), axis = -1)
        regularization_latent = self.l1_latent_regularization(latent_in_variables, self.lambda_regularization, latent_boundaries_variables) #regularization latent space 
        definitive_latent = self.final_reduction(latent_in_variables) #final reduced vector of inner fields
        
        if is_AE_frozen:
            definitive_latent = definitive_latent.detach()
            for count, i in enumerate(latent_in_variables_separated):
                if i != None:
                    latent_in_variables_separated[count] = i.detach()
            if latent_boundaries_variables != None:      
                latent_boundaries_variables = latent_boundaries_variables.detach()
            regularization_latent = regularization_latent.detach()
            
        return definitive_latent, latent_in_variables_separated, latent_boundaries_variables, regularization_latent
    
    def l1_latent_regularization(self, latent_fields: tc.tensor, lambda_l1: float, latent_boundaries: tc.tensor = None):
        
        if lambda_l1 != 0:
            if latent_boundaries is not None:
                l1_norm = (tc.mean(tc.abs(latent_fields)) + tc.mean(tc.abs(latent_boundaries)))/2
            else:
                l1_norm = tc.mean(tc.abs(latent_fields))
                
            return lambda_l1 * l1_norm
        else:
            return tc.tensor(0.0, device=latent_fields.device)
    
class Fully_Connected_Encoder(nn.Module):
    def __init__(self, config_training:dict, fully_connected_information:dict):
        super().__init__()
        
        self.input_dimension = fully_connected_information['input_dimension_encoder']
        self.output_dimension = fully_connected_information['output_dimension_encoder'] 
        self.list_of_neurons = fully_connected_information['list_of_neurons_encoder']
        self.number_of_layers = len(self.list_of_neurons)
        self.last_activation = fully_connected_information['last_activation_encoder']
        self.layers_norm = tc.nn.ModuleList()
        self.layer_norm_encoder = fully_connected_information['layer_norm_encoder']
        max_number_layer_norm = self.number_of_layers + 2 if self.number_of_layers > 0 else 1
        
        if max_number_layer_norm != len(fully_connected_information['layer_norm_encoder']):
            raise TypeError(f'Length layer_norm_encoder is wrong. It is {len(fully_connected_information["layer_norm_encoder"])}, should be {max_number_layer_norm}.')
        
        #get dimensions of the output of each layer
        dimension_outputs_layers = []
        if self.number_of_layers > 0:
            dimension_outputs_layers.append(self.list_of_neurons[0])
        for i in self.list_of_neurons:
            dimension_outputs_layers.append(i)
        dimension_outputs_layers.append(self.output_dimension)
        
        #build list of layers norm where needed
        for count, l in enumerate(fully_connected_information['layer_norm_encoder']):
            if l:
                self.layers_norm.append(tc.nn.LayerNorm(dimension_outputs_layers[count]))
            else:
                self.layers_norm.append(None)
        self.ELU = nn.ELU()
        self.activation = self.ELU
        
        #build list of layers
        if len(self.list_of_neurons) != 0:
            self.layers = tc.nn.ModuleList([nn.Linear(self.input_dimension, dimension_outputs_layers[0])])
            for count, i in enumerate(dimension_outputs_layers[:-1]):
                self.layers.append(nn.Linear(dimension_outputs_layers[count], dimension_outputs_layers[count+1]))
        else:
            self.layers = tc.nn.ModuleList([tc.nn.Linear(self.input_dimension, self.output_dimension, bias=True)])

    def forward(self, x:tc.tensor):
        # first layer
        x = self.layers[0](x)
        if self.layer_norm_encoder[0]:
            x = self.layers_norm[0](x)
        if len(self.list_of_neurons) != 0:
            x = self.activation(x)
            for count, i in enumerate(self.layers[1:-1]):
                x = i(x)
                if self.layer_norm_encoder[count+1]:
                    x = self.layers_norm[count+1](x)
                x = self.activation(x)
                
            x = self.layers[-1](x)
            if self.layer_norm_encoder[-1]:
                x = self.layers_norm[-1](x)
                
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
        self.group_norm_encoder = convolutional_information['group_norm_encoder']
        max_number_layer_norm = len(self.kernel) + 1 #last one is dense layer
        if max_number_layer_norm != len(convolutional_information['group_norm_encoder']):
            raise TypeError(f'Length group_norm_encoder is wrong. It is {len(convolutional_information["group_norm_encoder"])}, should be {max_number_layer_norm}.')
        
        self.input_dfnn = self.dim_input.copy()
        
        # Activation function 
        self.ELU = nn.ELU()
        self.activation = self.ELU

        # Convolutional layers and BatchNorm layers
        self.convolutionals = nn.ModuleList()
        self.group_norm_layers = nn.ModuleList()
        self.len_kernel = len(self.kernel)
        self.last_activation = convolutional_information['last_activation_encoder']
    
        for i in range(self.len_kernel):
            self.convolutionals.append(nn.Conv2d(self.channels[i], self.filters[i], self.kernel[i], stride=self.strides[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
            if self.group_norm_encoder[i]:
                self.group_norm_layers.append(tc.nn.GroupNorm(self.filters[i]//2, self.filters[i]))
            else:
                self.group_norm_layers.append(None)
            nn.init.xavier_uniform_(self.convolutionals[i].weight)
            
            self.input_dfnn[0] = self.filters[i]
            self.input_dfnn[1] = np.ceil(self.input_dfnn[1]/2) if self.strides[i][0] == 2 else self.input_dfnn[1]
            self.input_dfnn[2] = np.ceil(self.input_dfnn[2]/2) if self.strides[i][1] == 2 else self.input_dfnn[2]
            
        self.shape_before_mlp_encoder = self.input_dfnn
        convolutional_information['shape_before_mlp_encoder'] = self.shape_before_mlp_encoder
        self.input_dfnn = int(np.prod(self.input_dfnn))
        
        # Dense fully-connected layer
        self.dfnn = nn.Linear(self.input_dfnn, self.output_dfnn, bias=True)
        nn.init.xavier_uniform_(self.dfnn.weight)
        if self.group_norm_encoder[-1]:
            self.group_norm_layers.append(tc.nn.LayerNorm(self.output_dfnn))
        else:
            self.group_norm_layers.append(None)
    
    def forward(self, x:tc.tensor):
        for i, conv_layer in enumerate(self.convolutionals):
            x = conv_layer(x)
            if self.group_norm_encoder[i]:
                x = self.group_norm_layers[i](x)
            x = self.activation(x)
        x = tc.flatten(x, 1)  # Flatten across the batch dimension
        x = self.dfnn(x)
        if self.group_norm_encoder[-1]:
            x = self.group_norm_layers[-1](x)
        if self.last_activation:
            x = self.activation(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, config_training:dict, model_information:dict):
        super().__init__()
        self.device = config_training['device']
        self.initial_increase = Fully_Connected_Decoder(config_training, model_information['auto_encoding']['final_reduction_and_initial_increase'])
        self.decoder_scalar_variables = Fully_Connected_Decoder(config_training, model_information['auto_encoding']['auto_encoder_scalar'])
        self.decoder_plenum_variables = Fully_Connected_Decoder(config_training, model_information['auto_encoding']['auto_encoder_plenum'])
        self.decoder_core_variables = Convolutional_Decoder(config_training, model_information['auto_encoding']['auto_encoder_core'])
        self.decoder_vessel_variables = Convolutional_Decoder(config_training, model_information['auto_encoding']['auto_encoder_vessel'])
        self.decoder_faces_variables = Convolutional_Decoder(config_training, model_information['auto_encoding']['auto_encoder_faces'])
        self.indeces = self.get_indeces_reconstruction_latent_vectors(model_information)
                
    def get_indeces_reconstruction_latent_vectors(self, model_information:dict):
        indeces = {}
        index = 0
        for i in model_information['auto_encoding']:
            if i != 'final_reduction_and_initial_increase' and i != 'auto_encoder_boundaries': 
                indeces[i] = (int( model_information['auto_encoding'][i]['output_dimension_encoder'])+index)
                index += int( model_information['auto_encoding'][i]['output_dimension_encoder'])
        return indeces
           
    def forward(self, definitive_latent:tc.tensor, is_AE_frozen: bool):
            
        concatenated_latents = self.initial_increase(definitive_latent)
        latent_scalar_variables = concatenated_latents[:,0:self.indeces['auto_encoder_scalar']]
        latent_plenum_variables = concatenated_latents[:,self.indeces['auto_encoder_scalar']:self.indeces['auto_encoder_plenum']]
        latent_core_variables = concatenated_latents[:,self.indeces['auto_encoder_plenum']:self.indeces['auto_encoder_core']]
        latent_vessel_variables = concatenated_latents[:,self.indeces['auto_encoder_core']:self.indeces['auto_encoder_vessel']]
        latent_faces_variables = concatenated_latents[:,self.indeces['auto_encoder_vessel']:self.indeces['auto_encoder_faces']]
        
        latent_in_variables_separated = [latent_scalar_variables, latent_plenum_variables, latent_core_variables, latent_vessel_variables, latent_faces_variables] #useful at testing
        reconstructed_scalar_variables = self.decoder_scalar_variables(latent_scalar_variables)
        reconstructed_plenum_variables = self.decoder_plenum_variables(latent_plenum_variables)
        reconstructed_core_variables = self.decoder_core_variables(latent_core_variables)
        reconstructed_vessel_variables = self.decoder_vessel_variables(latent_vessel_variables)
        reconstructed_faces_variables = self.decoder_faces_variables(latent_faces_variables)
        
        if is_AE_frozen:
            reconstructed_scalar_variables = reconstructed_scalar_variables.detach()
            reconstructed_core_variables = reconstructed_core_variables.detach()
            reconstructed_vessel_variables = reconstructed_vessel_variables.detach()
            reconstructed_plenum_variables = reconstructed_plenum_variables.detach()
            reconstructed_faces_variables = reconstructed_faces_variables.detach()
            for count, i in enumerate(latent_in_variables_separated):
                latent_in_variables_separated[count] = i.detach()

        return [reconstructed_scalar_variables, reconstructed_core_variables, reconstructed_vessel_variables, reconstructed_plenum_variables , reconstructed_faces_variables], latent_in_variables_separated
    
class Fully_Connected_Decoder(nn.Module):
    def __init__(self, config_training:dict, fully_connected_information:dict):
        super().__init__()
        self.list_of_neurons = fully_connected_information['list_of_neurons_decoder']
        self.number_of_layers = len(self.list_of_neurons)
        self.last_activation = fully_connected_information['last_activation_decoder']
        self.input_dimension = fully_connected_information['output_dimension_encoder']
        self.output_dimension = fully_connected_information['input_dimension_encoder']
        self.layers_norm = tc.nn.ModuleList()
        self.layer_norm_decoder = fully_connected_information['layer_norm_decoder']
        max_number_layer_norm = self.number_of_layers + 2 if self.number_of_layers > 0 else 1
        self.ELU = nn.ELU()
        self.activation = self.ELU
        
        if max_number_layer_norm != len(fully_connected_information['layer_norm_decoder']):
            raise TypeError(f'Length layer_norm_decoder is wrong. It is {len(fully_connected_information["layer_norm_decoder"])}, should be {max_number_layer_norm}.')
        
        #get dimensions of the output of each layer
        dimension_outputs_layers = []
        if self.number_of_layers > 0:
            dimension_outputs_layers.append(self.list_of_neurons[0])
        for i in self.list_of_neurons:
            dimension_outputs_layers.append(i)
        dimension_outputs_layers.append(self.output_dimension)
        
        #build list of layers norm where needed
        for count, l in enumerate(fully_connected_information['layer_norm_decoder']):
            if l:
                self.layers_norm.append(tc.nn.LayerNorm(dimension_outputs_layers[count]))
            else:
                self.layers_norm.append(None)
        #build list of layers
        if len(self.list_of_neurons) != 0:
            self.layers = tc.nn.ModuleList([nn.Linear(self.input_dimension, dimension_outputs_layers[0])])
            for count, i in enumerate(dimension_outputs_layers[:-1]):
                self.layers.append(nn.Linear(dimension_outputs_layers[count], dimension_outputs_layers[count+1]))
        else:
            self.layers = tc.nn.ModuleList([tc.nn.Linear(self.input_dimension, self.output_dimension, bias=True)])

    def forward(self, x:tc.tensor):
        # first layer
        x = self.layers[0](x)
        if self.layer_norm_decoder[0]:
            x = self.layers_norm[0](x)
            
        # process middle layers    
        if len(self.list_of_neurons) != 0:
            x = self.activation(x)
            for count, i in enumerate(self.layers[1:-1]):
                x = i(x)
                if self.layer_norm_decoder[count+1]:
                    x = self.layers_norm[count+1](x)
                x = self.activation(x)
                
            # last layer
            x = self.layers[-1](x)
            if self.layer_norm_decoder[-1]:
                x = self.layers_norm[-1](x)
                
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
        self.convolutional_information = convolutional_information["group_norm_decoder"]
        
        # Activation function (use only one for now)
        self.ELU = nn.ELU()
        self.activation = self.ELU

        # Convolutional layers and BatchNorm layers
        self.transposed_convolutionals = nn.ModuleList()
        self.group_norm_layers = nn.ModuleList()
        self.len_filters = len(self.filters) 
        self.last_activation = convolutional_information['last_activation_decoder']
        max_number_layer_norm = len(self.kernel) + 1
        if max_number_layer_norm != len(convolutional_information['group_norm_decoder']):
            raise TypeError(f'Length group_norm_decoder is wrong. It is {len(convolutional_information["group_norm_decoder"])}, should be {max_number_layer_norm}.')
        
        # Dense fully-connected layer
        self.dfnn = nn.Linear(self.input_dfnn, self.output_dfnn, bias=True)
        if convolutional_information["group_norm_decoder"][0]:
            self.group_norm_layers.append(tc.nn.LayerNorm(self.output_dfnn))
        else:
            self.group_norm_layers.append(None)
        nn.init.xavier_uniform_(self.dfnn.weight)
        # convolutional layers
        
        for i in range(self.len_filters-1):
            self.transposed_convolutionals.extend([nn.ConvTranspose2d(self.filters[i],self.filters[i+1],self.kernel[i],self.strides[i],padding=self.size_kernel[i], output_padding = 0)])
            if convolutional_information["group_norm_decoder"][i+1]:
                self.group_norm_layers.append(tc.nn.GroupNorm(self.filters[i+1]//2, self.filters[i+1]))
            else:
                self.group_norm_layers.append(None)
            nn.init.xavier_uniform_(self.transposed_convolutionals[i].weight)

    def forward(self, x:tc.tensor):
        x = self.dfnn(x)
        if self.convolutional_information[0]:
            x = self.group_norm_layers[0](x)
            
        if self.last_activation:
            x = self.activation(x) 
        x = tc.reshape(x,(x.size()[0], self.first_channel, int(self.shape_before_mlp_encoder[-2]), int(self.shape_before_mlp_encoder[-1])))
        
        length = len(self.transposed_convolutionals)
        for i in range(length-1):
            x = self.transposed_convolutionals[i](x)
            if self.convolutional_information[i+1]:
                x = self.group_norm_layers[i+1](x)
            x = self.activation(x)
        x = self.transposed_convolutionals[-1](x)
        return x
    
class F_Latent(nn.Module): 
    def __init__(self, config_training:dict, model_information:dict):
        super().__init__()
        self.param_dim = model_information['auto_encoding']['auto_encoder_boundaries']['output_dimension_encoder']
        self.relu = nn.ReLU()
        self.ELU = nn.ELU()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.leaky = nn.LeakyReLU()
        self.activation = self.ELU
        self.number_of_layers = model_information['number_of_layers_f']
        self.parameter_information = model_information['parameter_information']
        self.n_FiLM_conditioning = model_information['n_FiLM_conditioning']
        self.scaling_output_factor = model_information['scaling_output_factor']
        n_neurons = model_information['n_neurons_f']
        self.final_latent_dim = model_information['auto_encoding']['final_reduction_and_initial_increase']['output_dimension_encoder']
        self.layer_norm_node = model_information['layer_norm_node']
        self.layers_norm = tc.nn.ModuleList()
        max_number_layer_norm = self.number_of_layers + 2 if self.number_of_layers > 0 else 1
        
        if max_number_layer_norm != len(model_information['layer_norm_node']):
            raise TypeError(f'Length layer_norm_f is wrong. It is {len(model_information["layer_norm_node"])}, should be {max_number_layer_norm}.')
        
        #get dimensions of the output of each layer
        dimension_outputs_layers = []
        if self.number_of_layers > 0:
            dimension_outputs_layers.append(n_neurons)
        for i in range(self.number_of_layers):
            dimension_outputs_layers.append(n_neurons)
        dimension_outputs_layers.append(self.final_latent_dim)
        
        #build list of layers norm where needed
        for count, l in enumerate(model_information['layer_norm_node']):
            if l:
                self.layers_norm.append(tc.nn.LayerNorm(dimension_outputs_layers[count]))
            else:
                self.layers_norm.append(None)
                
        if self.parameter_information == 'concatenation':
            if self.number_of_layers !=0:

                if self.param_dim > 0:
                    self.linears = nn.ModuleList([nn.Linear(self.final_latent_dim + self.param_dim, n_neurons, bias = True)])
                    self.layers_norm.append(tc.nn.LayerNorm(n_neurons))
                else:
                    self.linears = nn.ModuleList([nn.Linear(self.final_latent_dim, n_neurons, bias = True)])
                    self.layers_norm.append(tc.nn.LayerNorm(n_neurons))

                self.linears.extend([nn.Linear(n_neurons, n_neurons, bias = True) for i in range(self.number_of_layers)])
                self.layers_norm.extend([ tc.nn.LayerNorm(n_neurons) for i in range(self.number_of_layers)])
                self.linears.append(nn.Linear(n_neurons, self.final_latent_dim, bias = True))
                self.layers_norm.append(tc.nn.LayerNorm(self.final_latent_dim))

                for i in self.linears:
                    nn.init.xavier_uniform_(i.weight)
                
                nn.init.xavier_uniform_(self.linears[-1].weight)
                
            else:
                
                if self.param_dim > 0:
                    self.dfnn = nn.Linear(self.final_latent_dim + self.param_dim, self.final_latent_dim, bias = True)
                    self.layers_norm.append(tc.nn.LayerNorm(self.final_latent_dim))
                    nn.init.xavier_uniform_(i.weight)
                else:
                    self.dfnn = nn.Linear(self.final_latent_dim, self.final_latent_dim, bias = True)
                    self.layers_norm.append(tc.nn.LayerNorm(self.final_latent_dim))
                    nn.init.xavier_uniform_(i.weight)


        elif self.parameter_information == 'FiLM':

            if self.param_dim > 0:
                self.param_FiLM_gamma = nn.ModuleList([nn.Linear(self.param_dim, self.final_latent_dim, bias = True)])
                self.param_FiLM_beta = nn.ModuleList([nn.Linear(self.param_dim, self.final_latent_dim, bias = True)])
                self.param_FiLM_gamma.extend([nn.Linear(self.param_dim, n_neurons, bias = True) for i in range(self.n_FiLM_conditioning)])
                self.param_FiLM_beta.extend([nn.Linear(self.param_dim, n_neurons, bias = True) for i in range(self.n_FiLM_conditioning)])

            self.linears = nn.ModuleList([nn.Linear(self.final_latent_dim, n_neurons, bias = True)])
            self.layers_norm.append(tc.nn.LayerNorm(n_neurons))
            self.linears.extend([nn.Linear(n_neurons, n_neurons, bias = True) for i in range(self.number_of_layers )])
            self.layers_norm.extend([tc.nn.LayerNorm(n_neurons) for i in range(self.number_of_layers )])
            self.linears.append(nn.Linear(n_neurons, self.final_latent_dim, bias = True))   
            self.layers_norm.append(tc.nn.LayerNorm(self.final_latent_dim))     

            for i in self.linears:
                nn.init.xavier_uniform_(i.weight)
                
            nn.init.xavier_uniform_(self.linears[-1].weight)
            
        else:
            raise ValueError("Wrong name of the type of parameter information")

    def forward(self, x:tc.tensor, parameter:tc.tensor):
        
        if self.parameter_information == 'concatenation':
            if self.param_dim > 0:
                x = tc.cat((x, parameter), dim=1)
            
            # First layer
            x = self.linears[0](x)
            if self.layer_norm_node[0]:
                x = self.layers_norm[0](x)
            x = self.activation(x)
            
            # Middle layers with residual connections
            for count, i in enumerate(self.linears[1:-1]):
                x = i(x)
                if self.layer_norm_node[count+1]:
                    x = self.layers_norm[count+1](x)
                x = self.activation(x)
            
            # Final layer
            x = self.linears[-1](x)
            if self.layer_norm_node[-1]:
                x = self.layers_norm[-1](x)
            
            return x * self.scaling_output_factor
            

        elif self.parameter_information == 'FiLM':
            if self.param_dim > 0:
                parameter_vector_gamma = self.param_FiLM_gamma[0](parameter)
                parameter_vector_beta = self.param_FiLM_beta[0](parameter)
                x = parameter_vector_gamma * x + parameter_vector_beta

            # First layer
            x = self.linears[0](x)
            if 0 < self.n_FiLM_conditioning:
                parameter_vector_gamma = self.param_FiLM_gamma[1](parameter)
                parameter_vector_beta = self.param_FiLM_beta[1](parameter)
                x = parameter_vector_gamma * x + parameter_vector_beta
            if self.layer_norm_node[0]:
                x = self.layers_norm[0](x)
            x = self.activation(x)
            
            # Middle layers with residual connections
            for count, i in enumerate(self.linears[1:-1], start=1):
                x = i(x)
                if (count+1) < self.n_FiLM_conditioning:
                    parameter_vector_gamma = self.param_FiLM_gamma[count+1](parameter)
                    parameter_vector_beta = self.param_FiLM_beta[count+1](parameter)
                    x = parameter_vector_gamma * x + parameter_vector_beta
                if self.layer_norm_node[count+1]:
                    x = self.layers_norm[count+1](x)
                x = self.activation(x)
                
            # Final layer
            x = self.linears[-1](x)
            if self.layer_norm_node[-1]:
                x = self.layers_norm[-1](x)
                
            return x * self.scaling_output_factor
import os
import numpy as np
import time
import torch as tc
from src.models.AE_NODE.training.data_functions import *
from src.models.AE_NODE.training.method_functions import Training_Losses
from torch.cuda.amp import autocast

class Training():
    def __init__(self, astec_instance):
        
        self.__dict__.update(astec_instance.__dict__)
        self.training_losses = Training_Losses(self)
        
    def check_if_NaN(self, l1_mean:tc.tensor, l1_latent:tc.tensor, l2_TF: tc.tensor,l2_AR: tc.tensor,l3: tc.tensor,regularization_latent: tc.tensor):
        if tc.isnan(l1_mean):
            print("NaN in l1_mean (reconstruction loss)")
            self.optim.zero_grad()
            return True
        if tc.isnan(l1_latent):
            print("NaN in l1_latent (latent loss)")
            self.optim.zero_grad()
            return True
        if tc.isnan(l2_TF):
            print("NaN in l2_TF (teacher forcing)")
            self.optim.zero_grad()
            return True
        if tc.isnan(l2_AR):
            print("NaN in l2_AR (autoregressive)")
            self.optim.zero_grad()
            return True
        if tc.isnan(l3):
            print("NaN in l3")
            self.optim.zero_grad()
            return True
        if tc.isnan(regularization_latent):
            print("NaN in regularization")
            self.optim.zero_grad()
            return True
        return False
            
            
    def train_epoch(self, loss_coefficients):
        l1_loss = tc.tensor(0.0, device = self.device)
        l1_loss_per_shape = tc.zeros(self.number_of_different_domains, device = self.device)
        l1_loss_latent =  tc.tensor(0.0, device = self.device)
        l2_TF_loss =  tc.tensor(0.0, device = self.device)
        l2_AR_loss =  tc.tensor(0.0, device = self.device)
        l3_loss =  tc.tensor(0.0, device = self.device)
        loss =  tc.tensor(0.0, device = self.device)
        count = 0.0
        regularization_loss =  tc.tensor(0.0, device = self.device)
        self.encoder.train()
        self.f.train()
        self.decoder.train()
        for fields, boundary_conditions, dt, length_of_padding in self.training_loader:
            self.optim.zero_grad()
            with tc.amp.autocast('cuda', enabled=(self.scaler is not None)):
                
                l1,l2_TF,l2_AR,l3, _,regularization_latent = self.training_losses.loss_sup_mixed(fields, boundary_conditions, dt, length_of_padding, loss_coefficients, True)
                
                l1_mean = l1[0]
                l1_mean_per_shape = l1[1]
                l1_latent = l1[2]
                
                #is_there_nan = self.check_if_NaN(l1_mean, l1_latent,l2_TF ,l2_AR, l3, regularization_latent)
                #if is_there_nan:
                #    continue
                if self.scaler is not None:
                    self.scaler.scale(l1_mean+l1_latent+l2_TF+l2_AR+l3+regularization_latent).backward()
                    if self.clipping[0]:
                        self.scaler.unscale_(self.optim)
                        all_params = (list(self.encoder.parameters()) + list(self.f.parameters()) + list(self.decoder.parameters()))
                        tc.nn.utils.clip_grad_norm_(all_params, max_norm=self.clipping[1])
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    
                else:
                    (l1_mean+l1_latent+l2_TF+l2_AR+l3+regularization_latent).backward()
                    
                    if self.clipping[0]:
                        all_params = (list(self.encoder.parameters()) + list(self.f.parameters()) + list(self.decoder.parameters()))
                        tc.nn.utils.clip_grad_norm_(all_params, max_norm=self.clipping[1])
                    
                    self.optim.step()
                 
                loss += (l1_mean +l1_latent+ l2_TF+l2_AR+l3).detach()
                l1_loss += (l1_mean).detach()
                l1_loss_per_shape += (l1_mean_per_shape).detach()
                l1_loss_latent += (l1_latent).detach()
                l2_TF_loss += l2_TF.detach()
                l2_AR_loss += l2_AR.detach()
                l3_loss += l3.detach()
                
                regularization_loss += regularization_latent.detach()
                count += 1

        return l1_loss.cpu().item()/count, l1_loss_per_shape.cpu().numpy()/count, l1_loss_latent.cpu().item()/count ,l2_TF_loss.cpu().item()/count, l2_AR_loss.cpu().item()/count ,l3_loss.cpu().item()/count, regularization_loss.cpu().item()/count, loss.cpu().item()/count
        

    def valid_epoch(self, loss_coefficients):
        
        l1_loss =  tc.tensor(0.0, device = self.device)
        l1_loss_per_shape = tc.zeros(self.number_of_different_domains, device = self.device)
        l1_loss_unnorm_per_variable = tc.zeros(self.number_of_different_domains, device = self.device)
        l1_loss_unnorm =  tc.tensor(0.0, device = self.device)
        l1_loss_latent =  tc.tensor(0.0, device = self.device)
        l2_TF_loss =  tc.tensor(0.0, device = self.device)
        l2_AR_loss =  tc.tensor(0.0, device = self.device)
        l3_loss =  tc.tensor(0.0, device = self.device)
        loss =  tc.tensor(0.0, device = self.device)
        count = 0.0
        regularization_loss =  tc.tensor(0.0, device = self.device)
        loss_real =  tc.tensor(0.0, device = self.device)
        loss_real_per_shape = tc.zeros(self.number_of_different_domains, device = self.device)
        
        
        self.encoder.eval()
        self.f.eval()
        self.decoder.eval()
            
            
        with tc.no_grad():
            for fields, boundary_conditions, dt, length_of_padding in self.validation_loader:
                t0 = time.time()
                l1,l2_TF,l2_AR,l3, l_final, regularization_latent  = self.training_losses.loss_sup_mixed(fields, boundary_conditions, dt, length_of_padding, loss_coefficients, False)
                l1_mean = l1[0]
                l1_mean_per_shape = l1[1]
                l1_mean_denormalized = l1[2]
                l1_mean_per_denormalized_per_variable = l1[3]
                l1_latent = l1[4]
                
                l_real_mean = l_final[0]
                l_real_per_shape = l_final[1]
                
                loss_real +=  l_real_mean.detach()
                loss += (l1_mean + l1_latent+ l2_TF + l2_AR + l3 + l_real_mean).detach()
                loss_real_per_shape += (l_real_per_shape).detach()
                l1_loss += (l1_mean ).detach()
                l1_loss_per_shape += (l1_mean_per_shape).detach()
                l1_loss_unnorm += (l1_mean_denormalized).detach()
                l1_loss_unnorm_per_variable += (l1_mean_per_denormalized_per_variable).detach()
                l1_loss_latent += (l1_latent).detach()
                l2_TF_loss += l2_TF.detach()
                l2_AR_loss += l2_AR.detach()
                l3_loss += l3.detach()
                regularization_loss += regularization_latent.detach()
                count += 1
                t1 = time.time()
                #print('validation: ', t1-t0)
        return l1_loss.cpu().item()/count, l1_loss_per_shape.cpu().numpy()/count, l1_loss_unnorm.cpu().item()/count, l1_loss_unnorm_per_variable.cpu().numpy()/count, l1_loss_latent.cpu().item()/count, l2_TF_loss.cpu().item()/count, l2_AR_loss.cpu().item()/count , l3_loss.cpu().item()/count, loss_real.cpu().item()/count, loss_real_per_shape.cpu().numpy()/count, regularization_loss.cpu().item()/count , loss.cpu().item()/count


    def training(self):
        if self.checkpoint:
            maximum_loss_coefficient_AR = self.loss_coefficients['AR'] # this line should be before calling load_checkpoint
            self.encoder, self.f , self.decoder, self.optim, self.scheduler, first_epoch, loss_value, self.loss_coefficients['AR'], before_next_window_change, how_many_datasets_creations, self.autoregressive_step, time_of_AE, time_of_only_TF = load_checkpoint(self.encoder, self.f , self.decoder, self.optim, self.scheduler, self.PATH_logs+'/checkpoint/check.pt', self.device)
            
            early_stopping = 0 
            train_l1 = np.load(self.PATH_logs + "/losses/train_l1.npy", allow_pickle=True)
            train_l1_per_shape = np.load(self.PATH_logs + "/losses/train_l1_per_shape.npy", allow_pickle=True)
            train_l1_latent = np.load(self.PATH_logs + "/losses/train_l1_latent.npy", allow_pickle=True)
            train_l2_TF = np.load(self.PATH_logs + "/losses/train_l2_TF.npy", allow_pickle=True)
            train_l2_AR = np.load(self.PATH_logs + "/losses/train_l2_AR.npy", allow_pickle=True)
            train_l3 = np.load(self.PATH_logs + "/losses/train_l3.npy", allow_pickle=True)
            train_regularization = np.load(self.PATH_logs + "/losses/train_regularization.npy", allow_pickle=True)
            train_loss_tot = np.load(self.PATH_logs + "/losses/train_loss_tot.npy", allow_pickle=True)

            valid_l1 = np.load(self.PATH_logs + "/losses/valid_l1.npy", allow_pickle=True)
            valid_l1_per_shape = np.load(self.PATH_logs + "/losses/valid_l1_per_shape.npy", allow_pickle=True)
            valid_l1_unnorm = np.load(self.PATH_logs + "/losses/valid_l1_unnorm.npy", allow_pickle=True)
            valid_l1_unnorm_per_variable = np.load(self.PATH_logs + "/losses/valid_l1_unnorm_per_variable.npy", allow_pickle=True)
            valid_l1_latent = np.load(self.PATH_logs + "/losses/valid_l1_latent.npy", allow_pickle=True)
            valid_l2_TF = np.load(self.PATH_logs + "/losses/valid_l2_TF.npy", allow_pickle=True)
            valid_l2_AR = np.load(self.PATH_logs + "/losses/valid_l2_AR.npy", allow_pickle=True)
            valid_l3 = np.load(self.PATH_logs + "/losses/valid_l3.npy", allow_pickle=True)
            valid_real = np.load(self.PATH_logs + "/losses/valid_real.npy", allow_pickle=True)
            valid_real_per_variable = np.load(self.PATH_logs + "/losses/valid_real_per_variable.npy", allow_pickle=True)
            valid_regularization = np.load(self.PATH_logs + "/losses/valid_regularization.npy", allow_pickle=True)
            valid_loss_tot = np.load(self.PATH_logs + "/losses/valid_loss_tot.npy", allow_pickle=True)
            
            self.training_loader, self.validation_loader = build_dataset(self.batch_sizes[how_many_datasets_creations], self.time_windows[how_many_datasets_creations],
                                                                                    self.data_training_path_dynamic, self.data_validation_path_dynamic, 
                                                                                    self.number_of_workers, self.data_path, self.where_to_save_data, 
                                                                                    self.which_normalization, self.device, 
                                                                                    self.config_training['indeces_training_boundaries'], self.config_training['indeces_validation_boundaries'],
                                                                                    self.all_on_gpu, self.pin_memory, self.indeces_training_boundaries, self.indeces_validation_boundaries)
            how_many_datasets_creations += 1
            for fields, _, _, _ in self.validation_loader:
                self.number_of_different_domains = len(fields)
                break
        
        else:
            for fields, _, _, _ in self.validation_loader:
                self.number_of_different_domains = len(fields)
                break
            
            loss_value = 100
            early_stopping = 0 
            first_epoch = 0

            train_l1 = np.zeros(self.epochs)
            train_l1_per_shape = np.zeros((self.epochs, self.number_of_different_domains))
            train_l1_latent = np.zeros(self.epochs)
            train_l2_TF = np.zeros(self.epochs)
            train_l2_AR = np.zeros(self.epochs)
            train_l3 = np.zeros(self.epochs)
            train_regularization = np.zeros(self.epochs)
            train_loss_tot = np.zeros(self.epochs)

            valid_l1 = np.zeros(self.epochs)
            valid_l1_per_shape = np.zeros((self.epochs, self.number_of_different_domains))
            valid_l1_unnorm = np.zeros(self.epochs)
            valid_l1_unnorm_per_variable = np.zeros((self.epochs, self.number_of_different_domains))
            valid_l1_latent = np.zeros(self.epochs)
            valid_l2_TF = np.zeros(self.epochs)
            valid_l2_AR = np.zeros(self.epochs)
            valid_l3 = np.zeros(self.epochs)
            valid_real = np.zeros(self.epochs)
            valid_real_per_variable = np.zeros((self.epochs, self.number_of_different_domains))
            valid_regularization = np.zeros(self.epochs)
            valid_loss_tot = np.zeros(self.epochs)
            
            maximum_loss_coefficient_AR = self.loss_coefficients['AR']
            self.loss_coefficients['AR'] = 0.0
            before_next_window_change = self.waiting_epochs_before_new_dataset_creation[0]
            how_many_datasets_creations = 1
            time_of_AE = True
            time_of_only_TF = True
            
        # create losses file
        os.makedirs(self.PATH_logs+'/losses/',exist_ok=True)
        os.makedirs(self.PATH_logs+'/checkpoint/',exist_ok=True)

        print("------------------TRAINING STARTS------------------")
        # cycle over epochs

        for i in np.arange(first_epoch, self.epochs+1, 1):
            early_stopping += 1
            if early_stopping == self.early_stopping:
                print('Training stopped due to early stopping')
                #writer.close()
                break
            time1 = time.time()
            if i < self.time_of_AE: #use only AE
                before_training = time.time()
                train_l1_data, train_l1_per_shape_data, train_l1_latent_data , train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.loss_coefficients['AE'],0,0,0])
                before_validation = time.time()
                valid_l1_data, valid_l1_per_shape_data, valid_l1_unnorm_data, valid_l1_unnorm_per_variable_data, valid_l1_latent_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_real_per_variable_data, valid_regularization_data, valid_loss_data = self.valid_epoch([[1,1],1,1,1])
                if self.is_coupled[0]:
                    valid_loss_data = valid_l1_data
            elif i >=self.time_of_AE and i < (self.time_only_TF+ self.time_of_AE): #use only TF
                if time_of_AE:
                    initialize_model_to_last_checkpoint(self.encoder, self.f, self.decoder, self.device, self.PATH_logs+'/checkpoint/check.pt')
                    loss_value = 100
                    time_of_AE = False
                before_training = time.time()
                train_l1_data, train_l1_per_shape_data, train_l1_latent_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.loss_coefficients['AE'],self.loss_coefficients['TF'],0, self.loss_coefficients['Random_DT']])
                before_validation = time.time()
                valid_l1_data, valid_l1_per_shape_data, valid_l1_unnorm_data, valid_l1_unnorm_per_variable_data, valid_l1_latent_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_real_per_variable_data, valid_regularization_data, valid_loss_data = self.valid_epoch([[1,1],1,1,1])
                valid_loss_data = valid_l1_data + valid_l1_latent_data + valid_l2_TF_data + valid_l3_data + valid_regularization_data
            else:
                print('AR')
                print(self.loss_coefficients['AR'])
                if time_of_AE:
                    initialize_model_to_last_checkpoint(self.encoder, self.f, self.decoder, self.device, self.PATH_logs+'/checkpoint/check.pt')
                    time_of_AE = False
                    time_of_only_TF = False
                    loss_value = 100
                    
                if time_of_only_TF:
                    initialize_model_to_last_checkpoint(self.encoder, self.f, self.decoder, self.device, self.PATH_logs+'/checkpoint/check.pt')
                    time_of_only_TF = False
                    loss_value = 100
                    
                if self.loss_coefficients['AR'] >= maximum_loss_coefficient_AR:
                    self.loss_coefficients['AR'] = maximum_loss_coefficient_AR
                else:
                    self.loss_coefficients['AR'] += self.loss_coefficients['AR_strength']
                    
                before_training = time.time()
                train_l1_data, train_l1_per_shape_data, train_l1_latent_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.loss_coefficients['AE'],self.loss_coefficients['TF'],self.loss_coefficients['AR'],self.loss_coefficients['Random_DT']])
                before_validation = time.time()
                
                valid_l1_data, valid_l1_per_shape_data, valid_l1_unnorm_data, valid_l1_unnorm_per_variable_data, valid_l1_latent_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_real_per_variable_data, valid_regularization_data, valid_loss_data = self.valid_epoch([[1,1],1,1,1])
                
            time2 = time.time()
            
            if i > self.time_of_lr_war_up:
                self.scheduler.step()
            else:
                self.pre_scheduler.step()
            train_l1[i] = train_l1_data
            train_l1_per_shape[i] = train_l1_per_shape_data
            train_l1_latent[i] = train_l1_latent_data
            train_l2_TF[i] = train_l2_TF_data
            train_l2_AR[i] = train_l2_AR_data
            train_l3[i] = train_l3_data
            train_regularization[i] = train_regularization_data
            train_loss_tot[i] = train_loss_data

            valid_l1[i] = valid_l1_data
            valid_l1_per_shape[i] = valid_l1_per_shape_data
            valid_l1_unnorm[i] = valid_l1_unnorm_data
            valid_l1_unnorm_per_variable[i] = valid_l1_unnorm_per_variable_data
            valid_l1_latent[i] = valid_l1_latent_data
            valid_l2_TF[i] = valid_l2_TF_data
            valid_l2_AR[i] = valid_l2_AR_data
            valid_l3[i] = valid_l3_data
            valid_real[i] = valid_real_data
            valid_real_per_variable[i] = valid_real_per_variable_data
            valid_regularization[i] = valid_regularization_data
            valid_loss_tot[i] = valid_loss_data

            np.save(self.PATH_logs + "/losses/train_l1.npy", train_l1)
            np.save(self.PATH_logs + "/losses/train_l1_per_shape.npy", train_l1_per_shape)
            np.save(self.PATH_logs + "/losses/train_l1_latent.npy", train_l1_latent)
            np.save(self.PATH_logs + "/losses/train_l2_TF.npy", train_l2_TF)
            np.save(self.PATH_logs + "/losses/train_l2_AR.npy", train_l2_AR)
            np.save(self.PATH_logs + "/losses/train_l3.npy", train_l3)
            np.save(self.PATH_logs + "/losses/train_regularization.npy", train_regularization)
            np.save(self.PATH_logs + "/losses/train_loss_tot.npy", train_loss_tot)

            np.save(self.PATH_logs + "/losses/valid_l1.npy", valid_l1)
            np.save(self.PATH_logs + "/losses/valid_l1_per_shape.npy", valid_l1_per_shape)
            np.save(self.PATH_logs + "/losses/valid_l1_unnorm.npy", valid_l1_unnorm_per_variable)
            np.save(self.PATH_logs + "/losses/valid_l1_unnorm_per_variable.npy", valid_l1_unnorm_per_variable)
            np.save(self.PATH_logs + "/losses/valid_l1_latent.npy", valid_l1_latent)
            np.save(self.PATH_logs + "/losses/valid_l2_TF.npy", valid_l2_TF)
            np.save(self.PATH_logs + "/losses/valid_l2_AR.npy", valid_l2_AR)
            np.save(self.PATH_logs + "/losses/valid_l3.npy", valid_l3)
            np.save(self.PATH_logs + "/losses/valid_real.npy", valid_real)
            np.save(self.PATH_logs + "/losses/valid_real_per_variable.npy", valid_real_per_variable)
            np.save(self.PATH_logs + "/losses/valid_regularization.npy", valid_regularization)
            np.save(self.PATH_logs + "/losses/valid_loss_tot.npy", valid_loss_tot)


            print("Epoch: " +str(i)+', ' + str(time2-time1)+ ' s')
            print('Time of training:', before_validation - before_training)
            print('Time of validation:', time2 - before_validation)
            print('Time window:', self.time_windows[how_many_datasets_creations-1])
            
            if self.loss_coefficients['AR'] != 0.0 and self.autoregressive_step['which_technique'] == 'TBPP_from_end':
                print(" for TBPP: " +str(self.autoregressive_step['TBPP_from_end_config'][0]))
                
            elif self.loss_coefficients['AR'] != 0.0 and self.autoregressive_step['which_technique'] == 'TBPP_from_start':
                print(" for TBPP: " +str(self.autoregressive_step['TBPP_from_start_config'][0]))
            
            elif self.loss_coefficients['AR'] != 0.0:
                print("Strength of autoregressive step: ", self.loss_coefficients['AR'])
                
            print('')
            print('Train_loss_data = ' + str(train_loss_data) + 
                '\nAE train loss = ' + str(train_l1_data) + 
                '\n AE train loss per variable' + str(train_l1_per_shape_data)+
                '\nAE train latent loss = ' + str(train_l1_latent_data) + 
                '\nTF train loss = ' + str(train_l2_TF_data) + 
                '\nAR train loss = ' + str(train_l2_AR_data) + 
                '\nRandom dt train loss = ' + str(train_l3_data) + 
                '\nregularization loss = ' + str(train_regularization_data))
            print(' ')
            print('Valid_loss_data = ' + str(valid_loss_data) + 
                '\nvalid Real loss = ' + str(valid_real_data) + 
                '\nvalid Real per variable loss = ' + str(valid_real_per_variable_data) + 
                '\nAE valid loss = ' + str(valid_l1_data) + 
                '\n AE valid loss per variable' + str(valid_l1_per_shape_data)+
                '\nAE valid latent loss = ' + str(valid_l1_latent_data) + 
                '\nAE valid unnorm  = ' + str(valid_l1_unnorm_data) + 
                '\nAE valid unnorm per variable  = ' + str(valid_l1_unnorm_per_variable_data) + 
                '\nvalid TF loss = ' + str(valid_l2_TF_data) + 
                '\nvalid AR loss = ' + str(valid_l2_AR_data) + 
                '\nRandom dt valid loss = ' + str(valid_l3_data) + 
                '\nvalid regularization = ' + str(valid_regularization_data))
            print('The validation loss has not decreased for ' + str(early_stopping) + ' self.epochs!')
            
            print('------------------------------------------------------')

            #check if training a noncoupled system and adjust accordingly the validatin losses to be checked for early stopping
            if not self.is_coupled[0] and self.is_coupled[1] == 'AE' and i >=self.time_of_AE:
                valid_loss_data = valid_l1_data + valid_regularization_data + valid_l1_latent_data
            elif not self.is_coupled[0] and self.is_coupled[1] == 'NODE':
                valid_loss_data = valid_real_data + valid_l2_TF_data + valid_l2_AR_data + valid_l3_data

            if np.mean(valid_loss_data) < loss_value: #careful valid loss tot!!
                loss_value = np.mean(valid_loss_data)
                print('Models saved!')
                save_checkpoint(self.encoder, self.f , self.decoder, self.optim, self.scheduler, i, loss_value, self.loss_coefficients['AR'] , before_next_window_change, how_many_datasets_creations-1, self.autoregressive_step, time_of_AE, time_of_only_TF, self.PATH_logs+'/checkpoint/check.pt')
                early_stopping = 0
                
            # check if it is needed to change the lenght of time series of the dataset.
            if self.dynamic_dataset_generation_during_training and i > (np.max([self.time_only_TF + self.time_of_AE, self.time_of_lr_war_up])) and how_many_datasets_creations < len(self.time_windows):
                
                if before_next_window_change == 0:
                    self.training_loader, self.validation_loader = build_dataset(self.batch_sizes[how_many_datasets_creations], self.time_windows[how_many_datasets_creations],
                                                                                    self.data_training_path_dynamic, self.data_validation_path_dynamic, 
                                                                                    self.number_of_workers, self.data_path, self.where_to_save_data, 
                                                                                    self.which_normalization, self.device, 
                                                                                    self.config_training['indeces_training_boundaries'], self.config_training['indeces_validation_boundaries'],
                                                                                    self.all_on_gpu, self.pin_memory, self.indeces_training_boundaries, self.indeces_validation_boundaries)
                    before_next_window_change = self.waiting_epochs_before_new_dataset_creation[how_many_datasets_creations]
                    how_many_datasets_creations+=1
                    os.remove(f"{self.data_training_path_dynamic}{str(self.time_windows[how_many_datasets_creations-2])}{self.indeces_training_boundaries}.h5")
                    os.remove(f"{self.data_validation_path_dynamic}{str(self.time_windows[how_many_datasets_creations-2])}{self.indeces_validation_boundaries}.h5")
                    checkpoint = tc.load(self.PATH_logs+'/checkpoint/check.pt', map_location=self.device, weights_only=False)
                    loss_value = 100
                    
                    #fetch the best model of previous iteration
                    if self.reinitialize_model_at_each_dataset_reshape:
                        initialize_model_to_last_checkpoint(self.encoder, self.f, self.decoder, self.device, self.PATH_logs+'/checkpoint/check.pt')
                    
                before_next_window_change-=1
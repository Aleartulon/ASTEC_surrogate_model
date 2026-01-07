import os
import numpy as np
import time
import torch as tc
from src.models.AE_NODE.training.data_functions import *
from src.models.AE_NODE.training.method_functions import Training_Losses
from torch.cuda.amp import autocast

class Training():
    def __init__(self, ae_node_instance):
        
        self.parent = ae_node_instance 
        self.losses = Training_Losses(ae_node_instance) 
            
    def train_epoch(self, loss_coefficients):
        l1_loss = tc.tensor(0.0, device = self.parent.device)
        l1_loss_per_shape = tc.zeros(self.parent.number_of_different_domains, device = self.parent.device)
        l1_loss_latent =  tc.tensor(0.0, device = self.parent.device)
        l2_TF_loss =  tc.tensor(0.0, device = self.parent.device)
        l2_AR_loss =  tc.tensor(0.0, device = self.parent.device)
        l3_loss =  tc.tensor(0.0, device = self.parent.device)
        loss =  tc.tensor(0.0, device = self.parent.device)
        count = 0.0
        regularization_loss =  tc.tensor(0.0, device = self.parent.device)
        self.parent.encoder.train()
        self.parent.f.train()
        self.parent.decoder.train()
        for fields, boundary_conditions, dt, length_of_padding in self.parent.training_loader:
            self.parent.optim.zero_grad()
            with tc.amp.autocast('cuda', enabled=(self.parent.scaler is not None)):
                
                l1,l2_TF,l2_AR,l3, _,regularization_latent = self.losses.loss_sup_mixed(fields, boundary_conditions, dt, length_of_padding, loss_coefficients, True)
                
                l1_mean = l1[0]
                l1_mean_per_shape = l1[1]
                l1_latent = l1[2]

                if self.parent.scaler is not None:
                    self.parent.scaler.scale(l1_mean+l1_latent+l2_TF+l2_AR+l3+regularization_latent).backward()
                    if self.parent.clipping[0]:
                        self.parent.scaler.unscale_(self.parent.optim)
                        all_params = (list(self.parent.encoder.parameters()) + list(self.parent.f.parameters()) + list(self.parent.decoder.parameters()))
                        tc.nn.utils.clip_grad_norm_(all_params, max_norm=self.parent.clipping[1])
                    self.parent.scaler.step(self.parent.optim)
                    self.parent.scaler.update()
                    
                else:
                    (l1_mean+l1_latent+l2_TF+l2_AR+l3+regularization_latent).backward()
                    
                    if self.parent.clipping[0]:
                        all_params = (list(self.parent.encoder.parameters()) + list(self.parent.f.parameters()) + list(self.parent.decoder.parameters()))
                        tc.nn.utils.clip_grad_norm_(all_params, max_norm=self.parent.clipping[1])
                    
                    self.parent.optim.step()
                 
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
        
        l1_loss =  tc.tensor(0.0, device = self.parent.device)
        l1_loss_per_shape = tc.zeros(self.parent.number_of_different_domains, device = self.parent.device)
        l1_loss_unnorm_per_variable = tc.zeros(self.parent.number_of_different_domains, device = self.parent.device)
        l1_loss_unnorm =  tc.tensor(0.0, device = self.parent.device)
        l1_loss_latent =  tc.tensor(0.0, device = self.parent.device)
        l2_TF_loss =  tc.tensor(0.0, device = self.parent.device)
        l2_AR_loss =  tc.tensor(0.0, device = self.parent.device)
        l3_loss =  tc.tensor(0.0, device = self.parent.device)
        loss =  tc.tensor(0.0, device = self.parent.device)
        count = 0.0
        regularization_loss =  tc.tensor(0.0, device = self.parent.device)
        loss_real =  tc.tensor(0.0, device = self.parent.device)
        loss_real_per_shape = tc.zeros(self.parent.number_of_different_domains, device = self.parent.device)
        
        
        self.parent.encoder.eval()
        self.parent.f.eval()
        self.parent.decoder.eval()
            
            
        with tc.no_grad():
            for fields, boundary_conditions, dt, length_of_padding in self.parent.validation_loader:
                t0 = time.time()
                l1,l2_TF,l2_AR,l3, l_final, regularization_latent  = self.losses.loss_sup_mixed(fields, boundary_conditions, dt, length_of_padding, loss_coefficients, False)
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
        if self.parent.checkpoint:
            maximum_loss_coefficient_AR = self.parent.loss_coefficients['AR'] # this line should be before calling load_checkpoint
            first_epoch, loss_value, self.parent.loss_coefficients['AR'], before_next_window_change, how_many_datasets_creations, self.parent.autoregressive_step, time_of_AE, time_of_only_TF, self.parent.is_AE_frozen = load_checkpoint(
                self.parent.encoder, 
                self.parent.f, 
                self.parent.decoder, 
                self.parent.optim, 
                self.parent.scheduler, 
                self.parent.PATH_logs+'/checkpoint/check.pt', 
                self.parent.device
            )
            
            early_stopping = 0 
            train_l1 = np.load(self.parent.PATH_logs + "/losses/train_l1.npy", allow_pickle=True)
            train_l1_per_shape = np.load(self.parent.PATH_logs + "/losses/train_l1_per_shape.npy", allow_pickle=True)
            train_l1_latent = np.load(self.parent.PATH_logs + "/losses/train_l1_latent.npy", allow_pickle=True)
            train_l2_TF = np.load(self.parent.PATH_logs + "/losses/train_l2_TF.npy", allow_pickle=True)
            train_l2_AR = np.load(self.parent.PATH_logs + "/losses/train_l2_AR.npy", allow_pickle=True)
            train_l3 = np.load(self.parent.PATH_logs + "/losses/train_l3.npy", allow_pickle=True)
            train_regularization = np.load(self.parent.PATH_logs + "/losses/train_regularization.npy", allow_pickle=True)
            train_loss_tot = np.load(self.parent.PATH_logs + "/losses/train_loss_tot.npy", allow_pickle=True)

            valid_l1 = np.load(self.parent.PATH_logs + "/losses/valid_l1.npy", allow_pickle=True)
            valid_l1_per_shape = np.load(self.parent.PATH_logs + "/losses/valid_l1_per_shape.npy", allow_pickle=True)
            valid_l1_unnorm = np.load(self.parent.PATH_logs + "/losses/valid_l1_unnorm.npy", allow_pickle=True)
            valid_l1_unnorm_per_variable = np.load(self.parent.PATH_logs + "/losses/valid_l1_unnorm_per_variable.npy", allow_pickle=True)
            valid_l1_latent = np.load(self.parent.PATH_logs + "/losses/valid_l1_latent.npy", allow_pickle=True)
            valid_l2_TF = np.load(self.parent.PATH_logs + "/losses/valid_l2_TF.npy", allow_pickle=True)
            valid_l2_AR = np.load(self.parent.PATH_logs + "/losses/valid_l2_AR.npy", allow_pickle=True)
            valid_l3 = np.load(self.parent.PATH_logs + "/losses/valid_l3.npy", allow_pickle=True)
            valid_real = np.load(self.parent.PATH_logs + "/losses/valid_real.npy", allow_pickle=True)
            valid_real_per_variable = np.load(self.parent.PATH_logs + "/losses/valid_real_per_variable.npy", allow_pickle=True)
            valid_regularization = np.load(self.parent.PATH_logs + "/losses/valid_regularization.npy", allow_pickle=True)
            valid_loss_tot = np.load(self.parent.PATH_logs + "/losses/valid_loss_tot.npy", allow_pickle=True)
            
            self.parent.training_loader, self.parent.validation_loader = build_dataset(self.parent.batch_sizes[how_many_datasets_creations], self.parent.time_windows[how_many_datasets_creations],
                                                                                    self.parent.data_training_path_dynamic, self.parent.data_validation_path_dynamic, 
                                                                                    self.parent.number_of_workers, self.parent.data_path, self.parent.where_to_save_data, 
                                                                                    self.parent.which_normalization, self.parent.device, 
                                                                                    self.parent.config_training['indeces_training_boundaries'], self.parent.config_training['indeces_validation_boundaries'],
                                                                                    self.parent.all_on_gpu, self.parent.pin_memory, self.parent.indeces_training_boundaries, self.parent.indeces_validation_boundaries)
            how_many_datasets_creations += 1
            for fields, _, _, _ in self.parent.validation_loader:
                self.parent.number_of_different_domains = len(fields)
                break
            
            if self.parent.is_AE_frozen:
                for param in self.parent.encoder.parameters():
                    param.requires_grad = False
                    
                for param in self.parent.decoder.parameters():
                    param.requires_grad = False
                
                self.parent.is_AE_frozen = True
                print('AE has been frozen!')
        
        else:
            for fields, _, _, _ in self.parent.validation_loader:
                self.parent.number_of_different_domains = len(fields)
                break
            
            loss_value = 100
            early_stopping = 0 
            first_epoch = 0

            train_l1 = np.zeros(self.parent.epochs)
            train_l1_per_shape = np.zeros((self.parent.epochs, self.parent.number_of_different_domains))
            train_l1_latent = np.zeros(self.parent.epochs)
            train_l2_TF = np.zeros(self.parent.epochs)
            train_l2_AR = np.zeros(self.parent.epochs)
            train_l3 = np.zeros(self.parent.epochs)
            train_regularization = np.zeros(self.parent.epochs)
            train_loss_tot = np.zeros(self.parent.epochs)

            valid_l1 = np.zeros(self.parent.epochs)
            valid_l1_per_shape = np.zeros((self.parent.epochs, self.parent.number_of_different_domains))
            valid_l1_unnorm = np.zeros(self.parent.epochs)
            valid_l1_unnorm_per_variable = np.zeros((self.parent.epochs, self.parent.number_of_different_domains))
            valid_l1_latent = np.zeros(self.parent.epochs)
            valid_l2_TF = np.zeros(self.parent.epochs)
            valid_l2_AR = np.zeros(self.parent.epochs)
            valid_l3 = np.zeros(self.parent.epochs)
            valid_real = np.zeros(self.parent.epochs)
            valid_real_per_variable = np.zeros((self.parent.epochs, self.parent.number_of_different_domains))
            valid_regularization = np.zeros(self.parent.epochs)
            valid_loss_tot = np.zeros(self.parent.epochs)
            
            maximum_loss_coefficient_AR = self.parent.loss_coefficients['AR']
            self.parent.loss_coefficients['AR'] = 0.0
            before_next_window_change = self.parent.waiting_epochs_before_new_dataset_creation[0]
            how_many_datasets_creations = 1
            time_of_AE = True
            time_of_only_TF = True
            
        # create losses file
        os.makedirs(self.parent.PATH_logs+'/losses/',exist_ok=True)
        os.makedirs(self.parent.PATH_logs+'/checkpoint/',exist_ok=True)

        print("------------------TRAINING STARTS------------------")
        # cycle over epochs

        for i in np.arange(first_epoch, self.parent.epochs+1, 1):
            early_stopping += 1
            if early_stopping == self.parent.early_stopping:
                print('Training stopped due to early stopping')
                #writer.close()
                break
            time1 = time.time()
            if i < self.parent.time_of_AE: #use only AE
                before_training = time.time()
                train_l1_data, train_l1_per_shape_data, train_l1_latent_data , train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.parent.loss_coefficients['AE'],0,0,0])
                before_validation = time.time()
                valid_l1_data, valid_l1_per_shape_data, valid_l1_unnorm_data, valid_l1_unnorm_per_variable_data, valid_l1_latent_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_real_per_variable_data, valid_regularization_data, valid_loss_data = self.valid_epoch([[1,1],1,1,1])
                if self.parent.is_coupled[0]:
                    valid_loss_data = valid_l1_data
            elif i >=self.parent.time_of_AE and i < (self.parent.time_only_TF+ self.parent.time_of_AE): #use only TF
                if time_of_AE and not (self.parent.is_coupled[0] == False and self.parent.is_coupled[1] == 'NODE'):
                    initialize_model_to_last_checkpoint(self.parent.encoder, self.parent.f, self.parent.decoder, self.parent.device, self.parent.PATH_logs+'/checkpoint/check.pt')
                    loss_value = 100
                    time_of_AE = False
                before_training = time.time()
                train_l1_data, train_l1_per_shape_data, train_l1_latent_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.parent.loss_coefficients['AE'],self.parent.loss_coefficients['TF'],0, self.parent.loss_coefficients['Random_DT']])
                before_validation = time.time()
                valid_l1_data, valid_l1_per_shape_data, valid_l1_unnorm_data, valid_l1_unnorm_per_variable_data, valid_l1_latent_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_real_per_variable_data, valid_regularization_data, valid_loss_data = self.valid_epoch([[1,1],1,1,1])
                valid_loss_data = valid_l1_data + valid_l1_latent_data + valid_l2_TF_data + valid_l3_data + valid_regularization_data
            else:
                if time_of_AE:
                    initialize_model_to_last_checkpoint(self.parent.encoder, self.parent.f, self.parent.decoder, self.parent.device, self.parent.PATH_logs+'/checkpoint/check.pt')
                    time_of_AE = False
                    time_of_only_TF = False
                    loss_value = 100
                    
                if time_of_only_TF:
                    initialize_model_to_last_checkpoint(self.parent.encoder, self.parent.f, self.parent.decoder, self.parent.device, self.parent.PATH_logs+'/checkpoint/check.pt')
                    time_of_only_TF = False
                    loss_value = 100
                    
                if self.parent.loss_coefficients['AR'] >= maximum_loss_coefficient_AR:
                    self.parent.loss_coefficients['AR'] = maximum_loss_coefficient_AR
                else:
                    self.parent.loss_coefficients['AR'] += self.parent.loss_coefficients['AR_strength']
                    
                before_training = time.time()
                train_l1_data, train_l1_per_shape_data, train_l1_latent_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.parent.loss_coefficients['AE'],self.parent.loss_coefficients['TF'],self.parent.loss_coefficients['AR'],self.parent.loss_coefficients['Random_DT']])
                before_validation = time.time()
                
                valid_l1_data, valid_l1_per_shape_data, valid_l1_unnorm_data, valid_l1_unnorm_per_variable_data, valid_l1_latent_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_real_per_variable_data, valid_regularization_data, valid_loss_data = self.valid_epoch([[1,1],1,1,1])
                
            time2 = time.time()
            
            if i > self.parent.time_of_lr_war_up:
                self.parent.scheduler.step()
            else:
                self.parent.pre_scheduler.step()
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

            np.save(self.parent.PATH_logs + "/losses/train_l1.npy", train_l1)
            np.save(self.parent.PATH_logs + "/losses/train_l1_per_shape.npy", train_l1_per_shape)
            np.save(self.parent.PATH_logs + "/losses/train_l1_latent.npy", train_l1_latent)
            np.save(self.parent.PATH_logs + "/losses/train_l2_TF.npy", train_l2_TF)
            np.save(self.parent.PATH_logs + "/losses/train_l2_AR.npy", train_l2_AR)
            np.save(self.parent.PATH_logs + "/losses/train_l3.npy", train_l3)
            np.save(self.parent.PATH_logs + "/losses/train_regularization.npy", train_regularization)
            np.save(self.parent.PATH_logs + "/losses/train_loss_tot.npy", train_loss_tot)

            np.save(self.parent.PATH_logs + "/losses/valid_l1.npy", valid_l1)
            np.save(self.parent.PATH_logs + "/losses/valid_l1_per_shape.npy", valid_l1_per_shape)
            np.save(self.parent.PATH_logs + "/losses/valid_l1_unnorm.npy", valid_l1_unnorm_per_variable)
            np.save(self.parent.PATH_logs + "/losses/valid_l1_unnorm_per_variable.npy", valid_l1_unnorm_per_variable)
            np.save(self.parent.PATH_logs + "/losses/valid_l1_latent.npy", valid_l1_latent)
            np.save(self.parent.PATH_logs + "/losses/valid_l2_TF.npy", valid_l2_TF)
            np.save(self.parent.PATH_logs + "/losses/valid_l2_AR.npy", valid_l2_AR)
            np.save(self.parent.PATH_logs + "/losses/valid_l3.npy", valid_l3)
            np.save(self.parent.PATH_logs + "/losses/valid_real.npy", valid_real)
            np.save(self.parent.PATH_logs + "/losses/valid_real_per_variable.npy", valid_real_per_variable)
            np.save(self.parent.PATH_logs + "/losses/valid_regularization.npy", valid_regularization)
            np.save(self.parent.PATH_logs + "/losses/valid_loss_tot.npy", valid_loss_tot)


            print("Epoch: " +str(i)+', ' + str(time2-time1)+ ' s')
            print('Time of training:', before_validation - before_training)
            print('Time of validation:', time2 - before_validation)
            print('Time window:', self.parent.time_windows[how_many_datasets_creations-1])
            print('AE is frozen : ', self.parent.is_AE_frozen)
            
            if self.parent.loss_coefficients['AR'] != 0.0 and self.parent.autoregressive_step['which_technique'] == 'TBPP_from_end':
                print(" for TBPP: " +str(self.parent.autoregressive_step['TBPP_from_end_config'][0]))
                
            elif self.parent.loss_coefficients['AR'] != 0.0 and self.parent.autoregressive_step['which_technique'] == 'TBPP_from_start':
                print(" for TBPP: " +str(self.parent.autoregressive_step['TBPP_from_start_config'][0]))
            
            elif self.parent.loss_coefficients['AR'] != 0.0:
                print("Strength of autoregressive step: ", self.parent.loss_coefficients['AR'])
                
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
            print('The validation loss has not decreased for ' + str(early_stopping) + ' self.parent.epochs!')
            
            print('------------------------------------------------------')

            #check if training a noncoupled system and adjust accordingly the validatin losses to be checked for early stopping
            if not self.parent.is_coupled[0] and self.parent.is_coupled[1] == 'AE' and i >=self.parent.time_of_AE:
                valid_loss_data = valid_l1_data + valid_regularization_data + valid_l1_latent_data
            elif not self.parent.is_coupled[0] and self.parent.is_coupled[1] == 'NODE':
                valid_loss_data = valid_real_data + valid_l2_TF_data + valid_l2_AR_data + valid_l3_data

            if np.mean(valid_loss_data) < loss_value: #careful valid loss tot!!
                loss_value = np.mean(valid_loss_data)
                print('Models saved!')
                save_checkpoint(self.parent.encoder, self.parent.f , self.parent.decoder, self.parent.optim, self.parent.scheduler, i, loss_value, self.parent.loss_coefficients['AR'] , before_next_window_change, how_many_datasets_creations-1, self.parent.autoregressive_step, time_of_AE, time_of_only_TF, self.parent.is_AE_frozen, self.parent.PATH_logs+'/checkpoint/check.pt')
                early_stopping = 0
                if i > (self.parent.time_only_TF+ self.parent.time_of_AE) and self.parent.freeze_AE_after_a_while[0] and valid_l1_data < float(self.parent.freeze_AE_after_a_while[1]) and self.parent.time_windows[how_many_datasets_creations-1] >= int(self.parent.freeze_AE_after_a_while[2]):
                    if not self.parent.is_AE_frozen:
                        for param in self.parent.encoder.parameters():
                            param.requires_grad = False
                            
                        for param in self.parent.decoder.parameters():
                            param.requires_grad = False
                        
                        self.parent.is_AE_frozen = True
                        print('AE has been frozen!')
                        
                
            # check if it is needed to change the lenght of time series of the dataset.
            if self.parent.dynamic_dataset_generation_during_training and i > (np.max([self.parent.time_only_TF + self.parent.time_of_AE, self.parent.time_of_lr_war_up])) and how_many_datasets_creations < len(self.parent.time_windows):
                
                if before_next_window_change == 0:
                    self.parent.training_loader, self.parent.validation_loader = build_dataset(self.parent.batch_sizes[how_many_datasets_creations], self.parent.time_windows[how_many_datasets_creations],
                                                                                    self.parent.data_training_path_dynamic, self.parent.data_validation_path_dynamic, 
                                                                                    self.parent.number_of_workers, self.parent.data_path, self.parent.where_to_save_data, 
                                                                                    self.parent.which_normalization, self.parent.device, 
                                                                                    self.parent.config_training['indeces_training_boundaries'], self.parent.config_training['indeces_validation_boundaries'],
                                                                                    self.parent.all_on_gpu, self.parent.pin_memory, self.parent.indeces_training_boundaries, self.parent.indeces_validation_boundaries)
                    before_next_window_change = self.parent.waiting_epochs_before_new_dataset_creation[how_many_datasets_creations]
                    how_many_datasets_creations+=1
                    os.remove(f"{self.parent.data_training_path_dynamic}{str(self.parent.time_windows[how_many_datasets_creations-2])}{self.parent.indeces_training_boundaries}.h5")
                    os.remove(f"{self.parent.data_validation_path_dynamic}{str(self.parent.time_windows[how_many_datasets_creations-2])}{self.parent.indeces_validation_boundaries}.h5")
                    checkpoint = tc.load(self.parent.PATH_logs+'/checkpoint/check.pt', map_location=self.parent.device, weights_only=False)
                    loss_value = 100
                    
                    #fetch the best model of previous iteration
                    if self.parent.reinitialize_model_at_each_dataset_reshape:
                        initialize_model_to_last_checkpoint(self.parent.encoder, self.parent.f, self.parent.decoder, self.parent.device, self.parent.PATH_logs+'/checkpoint/check.pt')
                    
                before_next_window_change-=1
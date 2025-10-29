import os
import numpy as np
import time
import torch as tc
from src.models.AE_NODE.training.data_functions import *
from src.models.AE_NODE.training.method_functions import Training_Losses

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
        l1_loss = 0
        l1_loss_per_shape = np.zeros(self.number_of_different_domains)
        l1_loss_latent = 0
        l2_TF_loss = 0
        l2_AR_loss = 0
        l3_loss = 0
        loss = 0
        count = 0
        regularization_loss = 0
        self.encoder.train()
        self.f.train()
        self.decoder.train()
            
        for fields, boundary_conditions, dt, length_of_padding in self.training_loader:
            
            l1,l2_TF,l2_AR,l3, _,regularization_latent = self.training_losses.loss_sup_mixed(fields, boundary_conditions, dt, length_of_padding, loss_coefficients, True)
            
            l1_mean = l1[0]
            l1_mean_per_shape = l1[1]
            l1_latent = l1[2]
            
            #is_there_nan = self.check_if_NaN(l1_mean, l1_latent,l2_TF ,l2_AR, l3, regularization_latent)
            #if is_there_nan:
            #    continue
            (l1_mean+l1_latent+l2_TF+l2_AR+l3+regularization_latent).backward()
            if self.clipping[0]:
                
                all_params = (list(self.encoder.parameters()) + list(self.f.parameters()) + list(self.decoder.parameters()))
                tc.nn.utils.clip_grad_norm_(all_params, max_norm=self.clipping[1])
                
            self.optim.step()
            self.optim.zero_grad() 
    
            loss += (l1_mean +l1_latent+ l2_TF+l2_AR+l3).detach().cpu().item()
            l1_loss += (l1_mean).detach().cpu().numpy()
            l1_loss_per_shape += (l1_mean_per_shape).detach().cpu().numpy()
            l1_loss_latent += (l1_latent).detach().cpu().numpy()
            l2_TF_loss += l2_TF.detach().cpu().item()
            l2_AR_loss += l2_AR.detach().cpu().item()
            l3_loss += l3.detach().cpu().item()
            
            regularization_loss += regularization_latent.detach().cpu().item()
            count += 1
        return l1_loss/count, l1_loss_per_shape/count, l1_loss_latent/count ,l2_TF_loss/count, l2_AR_loss/count ,l3_loss/count, regularization_loss/count, loss/count
        

    def valid_epoch(self, loss_coefficients):
        
        l1_loss = 0
        l1_loss_per_shape = np.zeros(self.number_of_different_domains)
        l1_loss_unnorm_per_variable = np.zeros(self.number_of_different_domains)
        l1_loss_unnorm = 0
        l1_loss_latent = 0
        l2_TF_loss = 0
        l2_AR_loss = 0
        l3_loss = 0
        loss = 0
        count = 0
        regularization_loss = 0
        loss_real = 0
        loss_real_per_shape = np.zeros(self.number_of_different_domains-1)
        
        
        self.encoder.eval()
        self.f.eval()
        self.decoder.eval()
            
            
        with tc.no_grad():
            for fields, boundary_conditions, dt, length_of_padding in self.validation_loader:
                l1,l2_TF,l2_AR,l3, l_final, regularization_latent  = self.training_losses.loss_sup_mixed(fields, boundary_conditions, dt, length_of_padding, loss_coefficients, False)
                l1_mean = l1[0]
                l1_mean_per_shape = l1[1]
                l1_mean_denormalized = l1[2]
                l1_mean_per_denormalized_per_variable = l1[3]
                l1_latent = l1[4]
                
                l_real_mean = l_final[0]
                l_real_per_shape = l_final[1]
                
                loss_real +=  l_real_mean.detach().item()
                loss += (l1_mean + l1_latent+ l2_TF + l2_AR + l3 + l_real_mean).detach().cpu().item()
                loss_real_per_shape += (l_real_per_shape).detach().cpu().numpy()
                l1_loss += (l1_mean ).detach().cpu().numpy()
                l1_loss_per_shape += (l1_mean_per_shape).detach().cpu().numpy()
                l1_loss_unnorm += (l1_mean_denormalized).detach().cpu().item()
                l1_loss_unnorm_per_variable += (l1_mean_per_denormalized_per_variable).detach().cpu().numpy()
                l1_loss_latent += (l1_latent).detach().cpu().item()
                l2_TF_loss += l2_TF.detach().cpu().item()
                l2_AR_loss += l2_AR.detach().cpu().item()
                l3_loss += l3.detach().cpu().item()
                regularization_loss += regularization_latent.detach().cpu().item()
                count += 1
        return l1_loss/count, l1_loss_per_shape/count, l1_loss_unnorm/count, l1_loss_unnorm_per_variable/count, l1_loss_latent/count, l2_TF_loss/count, l2_AR_loss/count , l3_loss/count, loss_real/count, loss_real_per_shape/count, regularization_loss/count , loss/count


    def training(self):
        
        if not self.checkpoint:
            # create losses file
            os.makedirs(self.PATH+'/losses/',exist_ok=True)
            os.makedirs(self.PATH+'/checkpoint/',exist_ok=True)
            
            #start the training

            #check for coupled_system
            if not self.is_coupled[0] and self.is_coupled[1] == 'NODE':
                self.time_of_AE = 0

            print("------------------TRAINING STARTS------------------")
            loss_value = 100
            early_stopping = 0 
            full_training_count = 1

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
            valid_real_per_variable = np.zeros((self.epochs, self.number_of_different_domains-1))
            valid_regularization = np.zeros(self.epochs)
            valid_loss_tot = np.zeros(self.epochs)
            
            maximum_loss_coefficient_AR = self.loss_coefficients['AR']
            self.loss_coefficients['AR'] = 0.0
            before_next_window_change = self.waiting_epochs_before_new_dataset_creation[0]
            how_many_datasets_creations = 1
            for i in range(self.epochs):

                early_stopping += 1
                if early_stopping == self.early_stopping:
                    print('Training stopped due to early stopping')
                    #writer.close()
                    break
                time1 = time.time()
                if i < self.time_of_AE: #use only AR
                    before_training = time.time()
                    train_l1_data, train_l1_per_shape_data, train_l1_latent_data , train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.loss_coefficients['AE'],0,0,0])
                    before_validation = time.time()
                    valid_l1_data, valid_l1_per_shape_data, valid_l1_unnorm_data, valid_l1_unnorm_per_variable_data, valid_l1_latent_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_real_per_variable_data, valid_regularization_data, valid_loss_data = self.valid_epoch([[1,1],1,1,1])
                    if self.is_coupled[0]:
                        valid_loss_data = 100.0
                elif i >=self.time_of_AE and i < self.time_only_TF: #use only TF
                    before_training = time.time()
                    train_l1_data, train_l1_per_shape_data, train_l1_latent_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.loss_coefficients['AE'],self.loss_coefficients['TF'],0, self.loss_coefficients['Random_DT']])
                    valid_l1_data, valid_l1_per_shape_data, valid_l1_unnorm_data, valid_l1_unnorm_per_variable_data, valid_l1_latent_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_real_per_variable_data, valid_regularization_data, valid_loss_data = self.valid_epoch([[1,1],1,1,1])
                else:
                    if self.loss_coefficients['AR'] >= maximum_loss_coefficient_AR:
                        self.loss_coefficients['AR'] = maximum_loss_coefficient_AR
                    else:
                        self.loss_coefficients['AR'] += self.loss_coefficients['AR_strength']
                        
                    before_training = time.time()
                    train_l1_data, train_l1_per_shape_data, train_l1_latent_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.loss_coefficients['AE'],self.loss_coefficients['TF'],self.loss_coefficients['AR'],self.loss_coefficients['Random_DT']])
                    before_validation = time.time()
                    
                    valid_l1_data, valid_l1_per_shape_data, valid_l1_unnorm_data, valid_l1_unnorm_per_variable_data, valid_l1_latent_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_real_per_variable_data, valid_regularization_data, valid_loss_data = self.valid_epoch([[1,1],1,1,1])
                    
                time2 = time.time()
                if self.dynamic_dataset_generation_during_training and i > (self.time_only_TF + self.time_of_AE) and how_many_datasets_creations < len(self.time_windows):
                    
                    if before_next_window_change == 0:
                        self.training_loader, self.validation_loader = build_dataset(self.batch_sizes[how_many_datasets_creations], self.time_windows[how_many_datasets_creations], self.data_training_path_dynamic, self.data_validation_path_dynamic, self.number_of_workers, self.data_path)
                        before_next_window_change = self.waiting_epochs_before_new_dataset_creation[how_many_datasets_creations]
                        how_many_datasets_creations+=1
                        os.remove(self.data_training_path_dynamic + str(self.time_windows[how_many_datasets_creations-2]) + '.h5')
                        os.remove(self.data_validation_path_dynamic + str(self.time_windows[how_many_datasets_creations-2]) + '.h5')
                        loss_value = 100
                        
                    before_next_window_change-=1
                
                
                if i > self.time_of_AE:
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

                np.save(self.PATH + "/losses/train_l1.npy", train_l1)
                np.save(self.PATH + "/losses/train_l1_per_shape.npy", train_l1_per_shape)
                np.save(self.PATH + "/losses/train_l1_latent.npy", train_l1_latent)
                np.save(self.PATH + "/losses/train_l2_TF.npy", train_l2_TF)
                np.save(self.PATH + "/losses/train_l2_AR.npy", train_l2_AR)
                np.save(self.PATH + "/losses/train_l3.npy", train_l3)
                np.save(self.PATH + "/losses/train_regularization.npy", train_regularization)
                np.save(self.PATH + "/losses/train_loss_tot.npy", train_loss_tot)

                np.save(self.PATH + "/losses/valid_l1.npy", valid_l1)
                np.save(self.PATH + "/losses/valid_l1_per_shape.npy", valid_l1_per_shape)
                np.save(self.PATH + "/losses/valid_l1_unnorm.npy", valid_l1_unnorm_per_variable)
                np.save(self.PATH + "/losses/valid_l1_unnorm_per_variable.npy", valid_l1_unnorm_per_variable)
                np.save(self.PATH + "/losses/valid_l1_latent.npy", valid_l1_latent)
                np.save(self.PATH + "/losses/valid_l2_TF.npy", valid_l2_TF)
                np.save(self.PATH + "/losses/valid_l2_AR.npy", valid_l2_AR)
                np.save(self.PATH + "/losses/valid_l3.npy", valid_l3)
                np.save(self.PATH + "/losses/valid_real.npy", valid_real)
                np.save(self.PATH + "/losses/valid_regularization.npy", valid_regularization)
                np.save(self.PATH + "/losses/valid_loss_tot.npy", valid_loss_tot)


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
                    valid_loss_data = valid_l1_data + valid_regularization_data
                elif not self.is_coupled[0] and self.is_coupled[1] == 'NODE':
                    valid_loss_data = valid_real_data + valid_l2_TF_data + valid_l2_AR_data + valid_l3_data

                if np.mean(valid_loss_data) < loss_value: #careful valid loss tot!!
                    loss_value = np.mean(valid_loss_data)
                    print('Models saved!')
                    save_checkpoint(self.encoder, self.f , self.decoder, self.optim, self.scheduler, i, loss_value, self.loss_coefficients['AR'] , self.autoregressive_step, full_training_count,self.PATH+'/checkpoint/check.pt')
                    early_stopping = 0
        
        else:

            self.encoder, self.f, self.decoder, self.optim, scheduler, start_epoch, loss, self.loss_coefficients['AR'], self.start_backprop, full_training_count = load_checkpoint(self.encoder, self.f , self.decoder, self.optim, scheduler, self.PATH+'/checkpoint/check.pt', self.device)
            self.encoder.to(self.device)
            self.f.to(self.device)
            self.decoder.to(self.device) 

            #start the training
            print("------------------TRAINING STARTS------------------")
            loss_value = loss
            early_stopping = 0 

            train_l1 = np.load(self.PATH + "/losses/train_l1.npy", allow_pickle=True)
            train_l2_TF = np.load(self.PATH + "/losses/train_l2_TF.npy",allow_pickle=True)
            train_l2_AR = np.load(self.PATH + "/losses/train_l2_AR.npy",allow_pickle=True)
            train_l3 = np.load(self.PATH + "/losses/train_l3.npy",allow_pickle=True)
            train_regularization = np.load(self.PATH + "/losses/train_regularization.npy",allow_pickle=True)
            train_loss_tot = np.load(self.PATH + "/losses/train_loss_tot.npy",allow_pickle=True)
            
            valid_l1 = np.load(self.PATH + "/losses/valid_l1.npy",allow_pickle=True)
            valid_l1_unnorm = np.load(self.PATH + "/losses/valid_l1_unnorm.npy",allow_pickle=True)
            valid_l2_TF = np.load(self.PATH + "/losses/valid_l2_TF.npy",allow_pickle=True)
            valid_l2_AR = np.load(self.PATH + "/losses/valid_l2_AR.npy",allow_pickle=True)
            valid_l3 = np.load(self.PATH + "/losses/valid_l3.npy",allow_pickle=True)
            valid_regularization = np.load(self.PATH + "/losses/valid_regularization.npy",allow_pickle=True)
            valid_loss_tot = np.load(self.PATH + "/losses/valid_loss_tot.npy",allow_pickle=True)
            valid_real = np.load(self.PATH + "/losses/valid_real.npy",allow_pickle=True)

            for i in np.arange(start_epoch+1, self.epochs+1, 1):

                early_stopping += 1
                if early_stopping == 200:
                    print('Training stopped due to early stopping')
                    #writer.close()
                    break
                time1 = time.time()
                if i < self.time_of_AE: #use only AR
                    train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(self.encoder, self.f, self.decoder,self.device, self.optim, training_data, [self.loss_coefficients[0],0,0,0], self.RK, self.k, self.start_backprop, self.lambda_regularization, self.clipping, self.is_coupled)
                    valid_l1_data, valid_l1_unnorm_per_variable_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(self.encoder, self.f, self.decoder,  self.device, validation_data,[1,1,1,1], self.RK, self.k, self.start_backprop, self.lambda_regularization, self.is_coupled)
                    if self.is_coupled[0]:
                        valid_loss_data = 100.0
                elif i >=self.time_of_AE and i < time_only_TF: #use only TF
                    train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(self.encoder, self.f, self.decoder,  self.device, self.optim, training_data,[self.loss_coefficients[0],self.loss_coefficients[1],0,self.loss_coefficients[3]], self.RK, self.k, self.start_backprop, self.lambda_regularization, self.clipping, self.is_coupled)
                    valid_l1_data, valid_l1_unnorm_per_variable_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(self.encoder, self.f, self.decoder,  self.device, validation_data,[1,1,1,1], self.RK, self.k, self.start_backprop, self.lambda_regularization, self.is_coupled)
                else:
                    self.loss_coefficients['AR'] = self.loss_coefficients[2] * AR_strength * full_training_count
                    full_training_count +=1
                    if self.loss_coefficients['AR'] >= self.loss_coefficients[2]:
                        self.loss_coefficients['AR'] = self.loss_coefficients[2]

                    if self.TBPP_dynamic[0]  and self.start_backprop[1] < self.TBPP_dynamic[2] and full_training_count%self.TBPP_dynamic[1] == 0:
                        self.start_backprop[1] += 1
                        
                    train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(self.encoder, self.f, self.decoder, self.device, self.optim, training_data,[self.loss_coefficients[0],self.loss_coefficients[1],self.loss_coefficients['AR'],self.loss_coefficients[3]], self.RK, self.k, self.start_backprop,self.lambda_regularization, self.clipping, self.is_coupled)
                    valid_l1_data, valid_l1_unnorm_per_variable_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(self.encoder, self.f, self.decoder, self.device, validation_data,[1,1,1,1], self.RK, self.k,self.start_backprop, self.lambda_regularization, self.is_coupled)

                time2 = time.time()

                if i > self.time_of_AE:
                    scheduler.step()
                else:
                    pre_scheduler.step()
                train_l1[i] = train_l1_data
                train_l2_TF[i] = train_l2_TF_data
                train_l2_AR[i] = train_l2_AR_data
                train_l3[i] = train_l3_data
                train_regularization[i] = train_regularization_data
                train_loss_tot[i] = train_loss_data

                valid_l1[i] = valid_l1_data
                valid_l1_unnorm[i] = valid_l1_unnorm_per_variable_data
                valid_l2_TF[i] = valid_l2_TF_data
                valid_l2_AR[i] = valid_l2_AR_data
                valid_l3[i] = valid_l3_data
                valid_real[i] = valid_real_data
                valid_regularization[i] = valid_regularization_data
                valid_loss_tot[i] = valid_loss_data

                np.save(self.PATH + "/losses/train_l1.npy", train_l1)
                np.save(self.PATH + "/losses/train_l2_TF.npy", train_l2_TF)
                np.save(self.PATH + "/losses/train_l2_AR.npy", train_l2_AR)
                np.save(self.PATH + "/losses/train_l3.npy", train_l3)
                np.save(self.PATH + "/losses/train_regularization.npy", train_regularization)
                np.save(self.PATH + "/losses/train_loss_tot.npy", train_loss_tot)

                np.save(self.PATH + "/losses/valid_l1.npy", valid_l1)
                np.save(self.PATH + "/losses/valid_l1_unnorm.npy", valid_l1_unnorm)
                np.save(self.PATH + "/losses/valid_l2_TF.npy", valid_l2_TF)
                np.save(self.PATH + "/losses/valid_l2_AR.npy", valid_l2_AR)
                np.save(self.PATH + "/losses/valid_l3.npy", valid_l3)
                np.save(self.PATH + "/losses/valid_real.npy", valid_real)
                np.save(self.PATH + "/losses/valid_regularization.npy", valid_regularization)
                np.save(self.PATH + "/losses/valid_loss_tot.npy", valid_loss_tot)


                print("Epoch: " +str(i)+', ' + str(time2-time1)+ ' s')
                if self.TBPP_dynamic[0]:
                    print("self.start_backprop for TBPP: " +str(self.start_backprop[1]))
                print("Loss coefficient 2: " +str(self.loss_coefficients['AR']))
                print('Train_loss_data = ' + str(train_loss_data) + ', l1 train loss = ' +str(train_l1_data) + ', l2 TF train loss = ' + str(train_l2_TF_data)+ ', l2 AR train loss = ' + str(train_l2_AR_data)+ ', l3 train loss = ' + str(train_l3_data)+ ', regularization loss = ' + str(train_regularization_data))
                print('Valid_loss_data = ' + str(valid_loss_data)+ ', valid Real loss = ' +str(valid_real_data)  + ', l1 valid loss = ' +str(valid_l1_data) +  ', l1 valid unnorm loss = ' +str(valid_l1_unnorm_per_variable_data) + ', l2 valid TF loss = ' + str(valid_l2_TF_data)+ ', l2 valid AR loss = ' + str(valid_l2_AR_data)+ ', l3 valid loss = ' + str(valid_l3_data)+ ', valid regularization = ' + str(valid_regularization_data))
                print('The validation loss has not decreased for ' + str(early_stopping) + ' epochs!')
                
                print('------------------------------------------------------')

                #check if training a noncoupled system and adjust accordingly the validatin losses to be checked for early stopping
                if not self.is_coupled[0] and self.is_coupled[1] == 'AE' and i >=self.time_of_AE:
                    valid_loss_data = valid_l1_data + valid_regularization_data
                elif not self.is_coupled[0] and self.is_coupled[1] == 'NODE':
                    valid_loss_data = valid_real_data + valid_l2_TF_data + valid_l2_AR_data + valid_l3_data

                if valid_loss_data < loss_value: #careful valid loss tot!!
                    loss_value = valid_loss_data
                    print('Models saved!')
                    save_checkpoint(self.encoder, self.f , self.decoder, self.optim, scheduler, i, loss_value, self.loss_coefficients['AR'] , self.start_backprop, full_training_count,self.PATH+'/checkpoint/check.pt')
                    early_stopping = 0


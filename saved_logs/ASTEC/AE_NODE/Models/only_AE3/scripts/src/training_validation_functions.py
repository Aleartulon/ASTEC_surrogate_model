import os
import numpy as np
import time
from src.models.AE_NODE.training.data_functions import *
from src.models.AE_NODE.training.method_functions import Training_Losses

class Training():
    def __init__(self, astec_instance):
        
        self.__dict__.update(astec_instance.__dict__)
        self.training_losses = Training_Losses(self)
    
    def train_epoch(self, loss_coefficients):
        l1_loss = 0
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
            
            l1,l2_TF,l2_AR,l3, _,regularization_latent  = self.training_losses.loss_sup_mixed(fields, boundary_conditions, dt, length_of_padding, loss_coefficients, True)
            
            (l1+l2_TF+l2_AR+l3+regularization_latent).backward()
            if self.clipping[0]:
                tc.nn.utils.clip_grad_norm_(self.f.parameters(), max_norm=self.clipping[1])
                
            self.optim.step()
            self.optim.zero_grad() 
    
                
            loss += (l1+l2_TF+l2_AR+l3).detach().cpu().item()
            
            l1_loss += l1.detach().cpu().item()
            l2_TF_loss += l2_TF.detach().cpu().item()
            l2_AR_loss += l2_AR.detach().cpu().item()
            l3_loss += l3.detach().cpu().item()
            regularization_loss += regularization_latent.detach().cpu().item()
            count += 1
        
        return l1_loss/count, l2_TF_loss/count, l2_AR_loss/count ,l3_loss/count, regularization_loss/count, loss/count
        

    def valid_epoch(self, loss_coefficients):
        
        l1_loss = 0
        l2_TF_loss = 0
        l2_AR_loss = 0
        l3_loss = 0
        loss = 0
        count = 0
        regularization_loss = 0
        l_real = 0
        l1_loss_unnorm = np.zeros(self.number_of_different_domains)
        
        self.encoder.eval()
        self.f.eval()
        self.decoder.eval()
            
            
        with tc.no_grad():
            for fields, boundary_conditions, dt, length_of_padding in self.validation_loader:
                l1,l2_TF,l2_AR,l3, l_final, regularization_latent  = self.training_losses.loss_sup_mixed(fields, boundary_conditions, dt, length_of_padding, loss_coefficients, False)
                    
                loss += (l1[0]+l2_TF+l2_AR+l3).detach().cpu().item()
                l_real +=  l_final.detach().item()
                l1_loss += l1[0].detach().cpu().item()
                l1_loss_unnorm += l1[1].detach().cpu().numpy()
                l2_TF_loss += l2_TF.detach().cpu().item()
                l2_AR_loss += l2_AR.detach().cpu().item()
                l3_loss += l3.detach().cpu().item()
                regularization_loss += regularization_latent.detach().cpu().item()
                count += 1
                
        return l1_loss/count, l1_loss_unnorm/count, l2_TF_loss/count, l2_AR_loss/count , l3_loss/count, l_real/count, regularization_loss/count , loss/count


    def training(self):
        
        if not self.checkpoint:
            # create losses file
            os.makedirs(self.PATH+'/losses/',exist_ok=True)
            os.makedirs(self.PATH+'/checkpoint/',exist_ok=True)

            loss_coeff_2 = -10
            #start the training

            #check for coupled_system
            if not self.is_coupled[0] and self.is_coupled[1] == 'NODE':
                self.time_of_AE = 0

            print("------------------TRAINING STARTS------------------")
            loss_value = 100
            early_stopping = 0 
            full_training_count = 1

            train_l1 = np.zeros(self.epochs)
            train_l2_TF = np.zeros(self.epochs)
            train_l2_AR = np.zeros(self.epochs)
            train_l3 = np.zeros(self.epochs)
            train_regularization = np.zeros(self.epochs)
            train_loss_tot = np.zeros(self.epochs)

            valid_l1 = np.zeros(self.epochs)
            valid_l1_unnorm = np.zeros((self.epochs, self.number_of_different_domains))
            valid_l2_TF = np.zeros(self.epochs)
            valid_l2_AR = np.zeros(self.epochs)
            valid_l3 = np.zeros(self.epochs)
            valid_real = np.zeros(self.epochs)
            valid_regularization = np.zeros(self.epochs)
            valid_loss_tot = np.zeros(self.epochs)
            
            if self.TBPP_dynamic[0]:
                self.start_backprop.append(1)
        
            for i in range(self.epochs):

                early_stopping += 1
                if early_stopping == self.early_stopping:
                    print('Training stopped due to early stopping')
                    #writer.close()
                    break
                time1 = time.time()
                if i < self.time_of_AE: #use only AR
                    before_training = time.time()
                    train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.loss_coefficients[0],0,0,0])
                    before_validation = time.time()
                    valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = self.valid_epoch([1,1,1,1])
                    if self.is_coupled[0]:
                        valid_loss_data = 100.0
                elif i >=self.time_of_AE and i < self.time_only_TF: #use only TF
                    before_training = time.time()
                    train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.loss_coefficients[0],self.loss_coefficients[1],0, self.loss_coefficients[3]])
                    valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = self.valid_epoch([1,1,1,1])
                else:
                    loss_coeff_2 = self.loss_coefficients[2] * self.AR_strength * full_training_count
                    full_training_count +=1
                    if loss_coeff_2 >= self.loss_coefficients[2]:
                        loss_coeff_2 = self.loss_coefficients[2]

                    if self.TBPP_dynamic[0]  and self.start_backprop[1] < self.TBPP_dynamic[2] and full_training_count%self.TBPP_dynamic[1] == 0:
                        self.start_backprop[1] += 1
                    before_training = time.time()
                    train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = self.train_epoch([self.loss_coefficients[0],self.loss_coefficients[1],loss_coeff_2,self.loss_coefficients[3]])
                    before_validation = time.time()
                    valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = self.valid_epoch([1,1,1,1])
                
                time2 = time.time()
                
                print('Time of training:', before_validation - before_training)
                print('Time of validation:', time2 - before_validation)
                
                if i > self.time_of_AE:
                    self.scheduler.step()
                else:
                    self.pre_scheduler.step()
                train_l1[i] = train_l1_data
                train_l2_TF[i] = train_l2_TF_data
                train_l2_AR[i] = train_l2_AR_data
                train_l3[i] = train_l3_data
                train_regularization[i] = train_regularization_data
                train_loss_tot[i] = train_loss_data

                valid_l1[i] = valid_l1_data
                valid_l1_unnorm[i] = valid_l1_unnorm_data
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
                print('')
                print("Loss coefficient 2: " +str(loss_coeff_2))
                print('Train_loss_data = ' + str(train_loss_data) + 
                    '\nl1 train loss = ' + str(train_l1_data) + 
                    '\nl2 TF train loss = ' + str(train_l2_TF_data) + 
                    '\nl2 AR train loss = ' + str(train_l2_AR_data) + 
                    '\nl3 train loss = ' + str(train_l3_data) + 
                    '\nregularization loss = ' + str(train_regularization_data))
                print(' ')
                print('Valid_loss_data = ' + str(valid_loss_data) + 
                    '\nvalid Real loss = ' + str(valid_real_data) + 
                    '\nl1 valid loss = ' + str(valid_l1_data) + 
                    '\nl1 valid unnorm loss = ' + str(valid_l1_unnorm_data) + 
                    '\nl2 valid TF loss = ' + str(valid_l2_TF_data) + 
                    '\nl2 valid AR loss = ' + str(valid_l2_AR_data) + 
                    '\nl3 valid loss = ' + str(valid_l3_data) + 
                    '\nvalid regularization = ' + str(valid_regularization_data))
                print('The validation loss has not decreased for ' + str(early_stopping) + ' self.epochs!')
                
                print('------------------------------------------------------')

                #check if training a noncoupled system and adjust accordingly the validatin losses to be checked for early stopping
                if not self.is_coupled[0] and self.is_coupled[1] == 'AE' and i >=self.time_of_AE:
                    valid_loss_data = valid_l1_data + valid_regularization_data
                elif not self.is_coupled[0] and self.is_coupled[1] == 'NODE':
                    valid_loss_data = valid_real_data + valid_l2_TF_data + valid_l2_AR_data + valid_l3_data

                if valid_loss_data < loss_value: #careful valid loss tot!!
                    loss_value = valid_loss_data
                    print('Models saved!')
                    save_checkpoint(self.encoder, self.f , self.decoder, self.optim, self.scheduler, i, loss_value, loss_coeff_2 , self.start_backprop, full_training_count,self.PATH+'/checkpoint/check.pt')
                    early_stopping = 0
        
        else:

            self.encoder, self.f, self.decoder, self.optim, scheduler, start_epoch, loss, loss_coeff_2, self.start_backprop, full_training_count = load_checkpoint(self.encoder, self.f , self.decoder, self.optim, scheduler, self.PATH+'/checkpoint/check.pt', self.device)
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
                    valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(self.encoder, self.f, self.decoder,  self.device, validation_data,[1,1,1,1], self.RK, self.k, self.start_backprop, self.lambda_regularization, self.is_coupled)
                    if self.is_coupled[0]:
                        valid_loss_data = 100.0
                elif i >=self.time_of_AE and i < time_only_TF: #use only TF
                    train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(self.encoder, self.f, self.decoder,  self.device, self.optim, training_data,[self.loss_coefficients[0],self.loss_coefficients[1],0,self.loss_coefficients[3]], self.RK, self.k, self.start_backprop, self.lambda_regularization, self.clipping, self.is_coupled)
                    valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(self.encoder, self.f, self.decoder,  self.device, validation_data,[1,1,1,1], self.RK, self.k, self.start_backprop, self.lambda_regularization, self.is_coupled)
                else:
                    loss_coeff_2 = self.loss_coefficients[2] * AR_strength * full_training_count
                    full_training_count +=1
                    if loss_coeff_2 >= self.loss_coefficients[2]:
                        loss_coeff_2 = self.loss_coefficients[2]

                    if self.TBPP_dynamic[0]  and self.start_backprop[1] < self.TBPP_dynamic[2] and full_training_count%self.TBPP_dynamic[1] == 0:
                        self.start_backprop[1] += 1
                        
                    train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(self.encoder, self.f, self.decoder, self.device, self.optim, training_data,[self.loss_coefficients[0],self.loss_coefficients[1],loss_coeff_2,self.loss_coefficients[3]], self.RK, self.k, self.start_backprop,self.lambda_regularization, self.clipping, self.is_coupled)
                    valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(self.encoder, self.f, self.decoder, self.device, validation_data,[1,1,1,1], self.RK, self.k,self.start_backprop, self.lambda_regularization, self.is_coupled)

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
                valid_l1_unnorm[i] = valid_l1_unnorm_data
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
                print("Loss coefficient 2: " +str(loss_coeff_2))
                print('Train_loss_data = ' + str(train_loss_data) + ', l1 train loss = ' +str(train_l1_data) + ', l2 TF train loss = ' + str(train_l2_TF_data)+ ', l2 AR train loss = ' + str(train_l2_AR_data)+ ', l3 train loss = ' + str(train_l3_data)+ ', regularization loss = ' + str(train_regularization_data))
                print('Valid_loss_data = ' + str(valid_loss_data)+ ', valid Real loss = ' +str(valid_real_data)  + ', l1 valid loss = ' +str(valid_l1_data) +  ', l1 valid unnorm loss = ' +str(valid_l1_unnorm_data) + ', l2 valid TF loss = ' + str(valid_l2_TF_data)+ ', l2 valid AR loss = ' + str(valid_l2_AR_data)+ ', l3 valid loss = ' + str(valid_l3_data)+ ', valid regularization = ' + str(valid_regularization_data))
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
                    save_checkpoint(self.encoder, self.f , self.decoder, self.optim, scheduler, i, loss_value, loss_coeff_2 , self.start_backprop, full_training_count,self.PATH+'/checkpoint/check.pt')
                    early_stopping = 0


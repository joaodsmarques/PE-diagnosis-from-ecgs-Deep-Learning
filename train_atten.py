# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 17:18:14 2022

@author: Utilizador
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 00:41:21 2022

@author: Utilizador
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader,Dataset
from bs4 import BeautifulSoup
import numpy as np
import glob
import os.path
import sys, os
#For gpu on regular code
from numba import jit, cuda
import gc
import pandas as pd
from resnet18_with_attention import ResNet18_with_att
import random
from focal_loss import FocalLoss
import warnings
from torchmetrics.classification import MulticlassPrecisionRecallCurve
import metrics_and_performance as mp
import data_augmentation as dtaug
import wandb

# setting device on GPU if available, else CPU
cuda0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(cuda0)

#To not display useless warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

##############################################################################
# Hyperparameters
lr_step = 0.0001
lr_factor =0.001
min_lr = 0.00001
batch_size = 32
epoch = 200
schedular_step = 30
schedular_gamma = 0.1
scheduler_milestones = [5,10,30,50]

norm_factor = 0.0025

#Focal loss related:
alpha_zero = 0.3
alpha_one = 0.7

#############################################################################

wandb_key = 'insert here your key!'
# start a new wandb run to track this script
wandb.login(key = wandb_key)



##############################################################################

#Init network weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
        

#creation of the Network

#model = Special_Net()
model = ResNet18_with_att(2, channels=12)
model.apply(init_weights)


###############################################################################

# Pytorch requires a class like this as input to dataloader
# More memory efficient
class ECG_Dataset(Dataset):
    
    def __init__(self,csv_file,root_dir, transform = None, test = False):
        #Normalizing the ecg
        self.root_dir = root_dir #Normalizing
        self.transform = transform
        self.labels = pd.read_csv(csv_file,delimiter = ";", header = None)
        self.test = test
        self.ecg_path = ''

    #Get the len of the datasett
    def __len__(self):
        return len(self.labels)
    

    #Atualizar isto de acordo com as praticas do documento do cluster
    def __getitem__(self,idx):
        self.ecg_path = os.path.join(self.root_dir,self.labels.iloc[idx,0])
        label_tensor = torch.tensor(int(self.labels.iloc[idx,1]))
        ecg = get_dataset(self.ecg_path,self.test)
        
        #Operations
        #Normalization:
        
        #If the data needs to be normalized - from standart unit to mV
        #ecg = torch.mul(ecg, norm_factor)
        
        #Apply z-score normalization method
        #signal = (signal-mean)/standard deviation
        #print(ecg)
        ecg = torch.sub(ecg,torch.mean(ecg))
        ecg = torch.div(ecg, torch.std(ecg, unbiased=False))
                
        #print(ecg)
        
        return (ecg, label_tensor) 
    
    def get_name(self):
        return self.ecg_path


#####################################################################

     
#####################################################################


#As the signal has a length of 5000, we can choose the length we want to analyse 
def pick_signal_section(flag, all_values):
    
    #pick from start
    if flag == 0:
        #return all_values[0:4096]
        return all_values[0:4096]
    #middle
    elif flag == 1:
        return all_values[451:4547]
    
    else:
        return all_values[904:5000]
    
#creates the tensor correspondent to the raw data from the extracted values
def get_signal_values(all_leads, leads_labels, test_flag):
  
  #Variable to save all leads info
  leads_final_info = []
  
  #Choose the data_augmentation to perform
  data_augmentation = random.randint(0,6)
  
  #keep test set coherent between different runs
  if test_flag == False:
      flag = random.randint(0,2)
      #flag = 0
  else:
      flag = 1

  #print(len(leads_labels))
  #Loop between all leads
  for i in range(len(leads_labels)):
    #print(leads_labels[i])
    lead_unit = all_leads.find('channel', {'name':leads_labels[i]})
    
    #Not needed for general purposes
    '''
    amplitude_unit = list(lead_I["amplitude_unit"])
    amplitude_unit[1] = "."
    amplitude_unit = "".join(amplitude_unit)
    amplitude_unit = float(amplitude_unit)
    freq = int(lead_I["freq"])
    '''
    
    # Split by number into a list of values
    all_values = lead_unit.text.split(" ")
  
    #Remove from beginning and end enough values to have a signal with 4096 points
    all_values = pick_signal_section(flag, all_values)
    
    #perform a data augmentation technique - flag is random
    lead = torch.tensor(list(map(float, all_values)))
    

    
    if i == 0:
        leads_final_info = torch.unsqueeze(lead, dim = 0)
        
    else:
        leads_final_info = torch.cat((leads_final_info, torch.unsqueeze(lead, dim = 0)), 0)
  
  #print(data_augmentation)
  #print(leads_final_info)
    
  augmented_leads = dtaug.choose_and_perform_dtaug(leads_final_info, data_augmentation) 
  
  return augmented_leads


def get_dataset(input_path, test_flag):
    
    
    leads_labels = ['I','II','III','aVR','aVL','aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    #leads_labels = ['I', 'III','V1', 'V2', 'V3', 'V4']
    leads_final_info = []
    
    with open(input_path, 'r', encoding="ISO-8859-1") as f:
        data = f.read()
        
    #Open xml file
    bs_data = BeautifulSoup(data, "lxml")
    
    #Get all the infos about all the leads
    all_leads = bs_data.find('rhythm')
    
    #Function to extract each value from the ecg by lead
    return get_signal_values(all_leads, leads_labels, test_flag)


#########################################################
#################################################################

def train(train_loader, test_loader, step_size = 0):
    #Prevents the program from stopping
    #torch.multiprocessing.freeze_support()
    
    
    #Classes for metrics
    train_metrics = mp.plots_and_metrics(len(train_loader.dataset),len(train_loader))
    test_metrics = mp.test_plots_and_metrics(len(test_loader.dataset),len(test_loader))
    mcprc = MulticlassPrecisionRecallCurve(num_classes=2, thresholds = 100).to(cuda0)
    
    #Move model to GPU if available
    if torch.cuda.is_available():
       model.to(cuda0)
    
    #Optimizer and loss functions definition
    optimizer = torch.optim.AdamW(model.parameters(), lr=step_size)
    
    criterion = FocalLoss(gamma = 2, alpha = torch.tensor([alpha_zero, alpha_one]).to(cuda0))
        
    
    wandb.watch(model, criterion, log = "all", log_freq = 10)
    #Toggle train mode
    model.train()   
    ########################################
    #            Training loop             #
    ########################################
    for iteration in tqdm(range(epoch)):
        #test = 0
        ###   Trainning Part    ####
        for data in train_loader:
            images, labels = data
            
            if torch.cuda.is_available():
               images = images.to(cuda0)
               labels = labels.to(cuda0)
            
            output = model(images)
            
            if torch.cuda.is_available():
                output = output.to(cuda0)
           
            loss = criterion(output,labels)
            
            # log metrics to wandb
            wandb.log({"epoch": iteration, "train loss": loss})
                            
            #To plot the training loss at this epoch
            train_metrics.add_loss(loss.item())
            #test = loss.item() + test
            #BackPropagation
            optimizer.zero_grad() # zero the gradient buffers
            loss.backward()
            optimizer.step() # Does the update

        
            #Get all predicted classes and calculate the correct prevision rate      
            _ , previsions = torch.max(output,1)
            #print(previsions)
            #print(labels)
            train_metrics.update_correct_outputs(previsions, labels)
        
        #Statistics - average loss over batches and accuracy rate
        train_metrics.get_average_loss()
        
        #print(test)
        #print(train_metrics.return_average_loss())
        
        wandb.log({"training avg loss": train_metrics.return_average_loss()})
        train_metrics.append_loss(increase = True)
        train_metrics.get_accuracy()
        
##################################################################################################
        #Evaluation set
################################################################################
        #toggle model to evaluation mode
        model.eval()     # As we dont want to use dropout while testing
        
        #For precision recall curves
        #flag for checking if it is starting a new evaluation routine
        first = -1
        
        for data, label in test_loader:
            
            if torch.cuda.is_available():
               data = data.to(cuda0)
               label = label.to(cuda0)
            
            # Network results for the validation set
            test_output = model(data)
                       
            if torch.cuda.is_available():
                test_output = test_output.to(cuda0)
           
            # Find the Loss
            loss = criterion(test_output,label)
            
            #Add loss item
            test_metrics.add_loss(loss.item())
     
            #Calculate accuracy - one hot encoded, the biggest value
            _,test_output = test_output.max(1)  # THIS IS FOR ONE HOT
            
            #For printing the classification differences
            #print('test_output: %f and labels:%f ' %(test_output, label))
            
            #Substitution - adds new prevision result (correct or incorrect)
            test_metrics.update_previsions(test_output, label, test_loader)
          
        
        #Get average loss and append loss to vector and increase the epoch axis for the plot
        test_metrics.get_average_loss()
        
        #Load average loss to wandb
        wandb.log({"epoch": iteration, "eval loss": test_metrics.return_average_loss(), "F1-score": test_metrics.get_f1_score(),"ppv": test_metrics.get_precision(), "recall": test_metrics.get_recall()})
        
        test_metrics.append_loss(increase = True)
        
        #substitute for performance metrics calculation
        test_metrics.update_performance()
                            
        #Print and/or plot the network results
        test_metrics.get_performance()
        #train_metrics.plot_loss('Training loss', 'Training Statistics: ')
        #test_metrics.plot_loss('Eval Loss', 'Eval Statistics:')
        
        #Reset epoch related variables
        test_metrics.clean_before_new_epoch()
        train_metrics.clean_before_new_epoch()
        
        #scheduler.step()
        model.train()
        #Print current learning rate
        #for param_group in optimizer.param_groups:
        #    print('Current learning rate: ', param_group['lr'])
        
        #print('Accuracy increased!!!')
        #torch.save(model.state_dict(), './parameters_last_epoch.pth')
    
    #Print performance
    test_metrics.print_performance()
    
    #Finish wandb
    wandb.finish()
    #CHANGE THIS!!!!!!!!!!!!    
    return [0,0]

################################################################
    
#################################################################
if __name__ == '__main__':
    
    
     #Data locations
    path_to_trainset = "./split_train_data/all"
    path_to_train_labels = "./split_train_data/all/labels_train_all.csv"
    path_to_testset = "./split_test_data/all"
    path_to_test_labels = "./split_test_data/all/labels_test_all.csv"
    
    #Convert data into dataset class
    train_set = ECG_Dataset(path_to_train_labels, path_to_trainset, test = False)
    test_set = ECG_Dataset(path_to_test_labels, path_to_testset, test = True)
    
    
    test_loader = DataLoader(
                            test_set, 
                            batch_size = 1, 
                            shuffle = False )

    
    print('Current lr: ', lr_step)

    wandb.init(
    # set the wandb project where this run will be logged
    project="final_model_thesis",
    name = 'with log ' + str(batch_size) + ' lr: ' + str(lr_step) + 'focal: ' + str(alpha_one),
    tags=["final_model"],
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr_step,
    "batch size": batch_size,
    "architecture": "ResNet with attention ",
    "dataset": "PE HSM dataset",
    "epochs": epoch,
    "run_name": "testing my library",
    "dropout": '0.5',
    "focal loss": [alpha_zero,alpha_one],
    "scheduler": 'no'
    
        }
    )
    #config file
    config = wandb.config
    
    #Get the values printed
    print(config)


    #train_loader
    train_loader = DataLoader(
                dataset = train_set, 
                batch_size = batch_size, 
                pin_memory = True, 
                prefetch_factor = 2, 
                shuffle = True, 
                drop_last=True,
                num_workers=4 
                )


    results.append(train(train_loader, test_loader, step_size = lr_step))

    #Finish wandb
    wandb.finish()
    
    #print(results)

    #pass train loader and test loader to the training loop
    
    
    











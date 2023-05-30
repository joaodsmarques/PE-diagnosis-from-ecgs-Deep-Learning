# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 21:24:36 2023

@author: Utilizador
"""

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
#Plot of the evolution of the training



#Self-made library to ease the metrics evaluation, plots and more
###############################################################################
def plot(epochs, plottable, name, ylabel=''):
    plt.clf()
    plt.xlabel(name)
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.show()
    #plt.savefig('%s.pdf' % (name), bbox_inches='tight')


class plots_and_metrics:
  
    def __init__(self, size_of_dataset, size_of_dataloader):
        self.min_loss = 1000
        self.curr_epoch = 1
        self.dataset_size = size_of_dataset
        self.data_loader_size = size_of_dataloader
        self.epoch_axis = []
        self.loss_axis = []
        self.loss_sum = 0
        self.correct_outputs = 0
        self.accuracy = 0
        self.average_loss = 0
       
        
    #get number of all elements of the dataset
    def get_dataset_size(self):
        return self.dataset_size
    
    #get number of batch size chunks
    def get_data_loader_size(self):
        return self.data_loader_size
    
    #get average loss over the batches
    def get_average_loss(self):
        self.average_loss = self.loss_sum / self.data_loader_size
       
        
    def return_average_loss(self):
        return self.average_loss
        
    #call at the end of each epoch
    def increase_epoch(self):
        self.epoch_axis.append(self.curr_epoch)
        self.curr_epoch +=1        
        
    def update_correct_outputs(self, previsions, labels):
        self.correct_outputs = (previsions == labels).sum().item() + self.correct_outputs
    
    #Save results on a file every 5 epochs
    def save_results(self):
    
        if self.curr_epoch % 5 == 0:
            print('saving results')
    
    #add last batch loss
    def add_loss(self, loss):
    
        self.loss_sum +=loss
        
    #append loss and update epoch count
    def append_loss(self, increase):
        self.loss_axis.append(self.average_loss)
        #o problema esta em acresvcentar 2 x um epoch
        if (increase == True):
            self.increase_epoch()
        
        
    #clean variables before going to a new epoch
    def clean_before_new_epoch(self):
        self.loss_sum = 0
        self.correct_outputs = 0
        self.accuracy = 0
        
    #Returns percentage of well classified outputs
    def get_accuracy(self):
        self.accuracy = self.correct_outputs/self.dataset_size
        #print(self.correct_outputs)
        #print(self.dataset_size)
        
            
    #plot loss graphically
    def plot_loss(self, plot_name,text_title):
        print(text_title)
        print('Average Loss: %f - Accuracy: %f' %( self.average_loss,self.accuracy))
        #Uncomment if needed for loss plot
        plot(self.epoch_axis, self.loss_axis, name = plot_name)
        
        
###############################################################################

#Class with all the content from the super class
class test_plots_and_metrics(plots_and_metrics):
    
    def __init__(self, size_of_dataset, size_of_dataloader):
        
        super().__init__(size_of_dataset, size_of_dataloader)
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.class1_correct = 0
        self.class1_incorrect = 0
        self.class0_correct = 0
        self.class0_incorrect = 0
        #keeps record of all the information per epoch
        #0 - accuracy, 1 - precision, 2 - recall, 3 - f1 score
        self.performance = []
        
    def get_precision(self):
        return self.precision
    
    def get_recall(self):
        return self.recall
    
    def get_f1_score(self):
        return self.f1_score
    
    def print_performance(self):
        print('Epoch, Accuracy, Precision, Recall, F1')
        for i in range(len(self.performance)):
            print('Epoch: ', str(i), ' - Accuracy: %f - Precision: %f - Recall: %f - F1_score: %f'
                 % (self.performance[i][0],self.performance[i][1],
                    self.performance[i][2],self.performance[i][3]))
        
        #print(self.performance)     
        
    def update_previsions(self, output, label, data_loader):
        
        #If the class was correct
        if torch.equal(output, label):
            
            if label.item() == 1:
                self.class1_correct +=1
                #print('correct 1: ',data_loader.dataset.get_name())
            else:
                self.class0_correct +=1
        
        else:
            #It was class 0 but it classified as 1
            if label.item() == 1:
                self.class1_incorrect +=1 #number of false negatives
                
            else:
                self.class0_incorrect += 1 #number of false positives
                #print('incorrect 0: ',data_loader.dataset.get_name())

    #Updates all the metrics after epoch
    def update_performance(self):
        
        #Get number of correct previsions
        self.correct_outputs  = self.class0_correct + self.class1_correct
        #get accuracy
        self.get_accuracy()
        
        '''
        Important metrics formulas:
            Precision = TruePositives / (TruePositives + FalsePositives)
            Recall = TruePositives / (TruePositives + FalseNegatives)
            F1 score = (2 * Precision * Recall) / (Precision + Recall)
        '''
        
        #Precision
        if (self.class0_incorrect + self.class1_correct) != 0:
            self.precision = self.class1_correct / (self.class1_correct + self.class0_incorrect)
        else:
            self.precision = 0
            
        #Recall (sensitivity)
        if (self.class1_incorrect + self.class1_correct) != 0:
            self.recall = self.class1_correct / (self.class1_correct + self.class1_incorrect)
        else:
            self.recall = 0
            
        #F1 score
        if (self.precision + self.recall) !=0 :
            self.f1_score = 2*self.precision*self.recall / (self.precision + self.recall)
            
        else:
            self.f1_score = 0
            
        #save into the performance vector
        self.performance.append([self.accuracy, self.precision, self.recall, self.f1_score])
        
    
    #Print all the acquired statistical metrics
    def get_performance(self):
        print('\n----------------------------------------')
        print('Test metrics and Overall Results')
        print('correct0: %d - incorrect0: %d - correct1: %d - incorrect1: %d' %(self.class0_correct, self.class0_incorrect,self.class1_correct,self.class1_incorrect))
        print('-------- Important Metrics ----------------')
        print('Correct previsions: ||| %d out of %d |||' %( self.correct_outputs, self.dataset_size))
        print('Accuracy: %f'% self.accuracy)
        print('Precision: %f'% self.precision)
        print('Recall: %f' % self.recall)
        print('f1 score: %f' % self.f1_score)
        print('-------------------------------------------')
        #print(' Accuracy, precision, recall, f1: ', self.performance)
        
    #Resets local variables
    def clean_before_new_epoch(self):
        super().clean_before_new_epoch()
        
        self.class0_correct = 0
        self.class1_correct = 0
        self.class0_incorrect = 0
        self.class1_incorrect = 0
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:00:48 2023

@author: thedges
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import transforms
import os
from CEDAR2019_params_simple import facilParams, cLight
import time
import h5py
from postpro_cnn_results import postprocess_cnn_results
from split_data import determine_chunk_startpts
from get_false_neg_rate_vs_snr import get_false_neg_rate
import pandas as pd

# Close out any lingering Matplotlib windows
plt.close("all")

# Batch size of images to load when testing CNN (separate from batch size used for the training process!)
TEST_BATCH_SIZE = 500

# Padding mode to use for edges of image in convolutional layers.
padding = "valid"
#padding = "same"

class RadarImageDatasetHDF5(Dataset):
    """Dataset object for PyTorch to load radar images for meteors.
    Loads .hdf5 file containing set of radar images."""

    def __init__(self, data_file, transform=None):
        self.data_file = data_file
        self.transform = transform

        # Get number of positive and negatively labelled examples
        self.pos_labels = 0
        self.neg_labels = 0
        self.unlabelled = 0

        # Open the HDF5 file and loop through all examples (HDF5 groups). Keep HDF5 file open while class exists.
        self.h5file = h5py.File(self.data_file, "r")
        for example in self.h5file.keys():

            obj = self.h5file[example]

            if obj.attrs["label"] == 1:
                self.pos_labels += 1
            elif obj.attrs["label"] == 0:
                self.neg_labels += 1
            else:
                self.unlabelled += 1
        
        # Get total number of examples
        self.num_examples = self.pos_labels + self.neg_labels + self.unlabelled
    
    # Function to get length of set, needed for PyTorch
    def __len__(self):
        return(self.num_examples)

    # Function to get a specific rawdata example, and its corresponding label, as a PyTorch tensor. Needed for PyTorch.
    def __getitem__(self, i):
        
        group_name = f"{i:07d}-rawdata"
        radar_image = torch.tensor(self.h5file[group_name]["data"][:,:], dtype=torch.float32)
        label = torch.tensor(self.h5file[group_name].attrs["label"], dtype=torch.int64)
        id = torch.tensor(self.h5file[group_name].attrs["id"])

        return(radar_image, label, id)

    def __del__(self):

        # Close hdf5 file when the class goes out of scope
        self.h5file.close()


class RadarImageDatasetHDF5Specific(Dataset):
    """Dataset object for PyTorch to load radar images for meteors.
    Loads .hdf5 files containing radar images.
    Has functionality to include only specific examples from HDF5 file in the dataset.
    Useful for specifying interesting examples to create plots for, etc. """

    def __init__(self, data_file, ids_to_include=None, transform=None):
        self.data_file = data_file
        self.transform = transform
        self.ids_to_include = ids_to_include  # A list of image IDs to include in the data set to train/validate/test

        # Get number of positive and negatively labelled examples
        self.pos_labels = 0
        self.neg_labels = 0
        self.unlabelled = 0

        # Open the HDF5 file
        self.h5file = h5py.File(self.data_file, "r")

        # If the list of specific IDs exists, account for only these IDs
        if ids_to_include is not None:
            for id in self.ids_to_include:
                example = f"{id:07d}-rawdata"

                obj = self.h5file[example]

                if obj.attrs["label"] == 1:
                    self.pos_labels += 1
                elif obj.attrs["label"] == 0:
                    self.neg_labels += 1
                else:
                    self.unlabelled += 1
        # Otherwise, include all examples
        else:
            for example in self.h5file.keys():

                obj = self.h5file[example]

                if obj.attrs["label"] == 1:
                    self.pos_labels += 1
                elif obj.attrs["label"] == 0:
                    self.neg_labels += 1
                else:
                    self.unlabelled += 1
        
        # Get total number of examples
        self.num_examples = self.pos_labels + self.neg_labels + self.unlabelled
    
    # Function to get length of set, needed for PyTorch
    def __len__(self):
        return(self.num_examples)

    # Function to get a specific rawdata example, and its corresponding label, as a PyTorch tensor. Needed for PyTorch.
    def __getitem__(self, i):

        if self.ids_to_include is not None:
            id = self.ids_to_include[i]
        else:
            id = i
        
        group_name = f"{id:07d}-rawdata"
        radar_image = torch.tensor(self.h5file[group_name]["data"][:,:], dtype=torch.float32)
        label = torch.tensor(self.h5file[group_name].attrs["label"], dtype=torch.int64)
        id = torch.tensor(self.h5file[group_name].attrs["id"])

        return(radar_image, label, id)

    def __del__(self):

        # Close hdf5 file when the class goes out of scope
        self.h5file.close()



class RadarImageDatasetDirect(Dataset):
    """ Loads radar images directly from formatted radar data files (no subdivision step).
    Examples are not labelled. TODO: not sure if this dataset was properly tested.
    """

    def __init__(self, data_file, chunk_size_x, chunk_size_y, min_overlap, start_sample,
                 end_sample, ids_to_include=None, transform=None):
        self.data_file = data_file
        self.transform = transform
        self.ids_to_include = ids_to_include
        self.chunk_size_x = chunk_size_x
        self.chunk_size_y = chunk_size_y

        # Open the HDF5 file
        self.h5file = h5py.File(self.data_file, "r")

        # Determine start points for each tile.
        Lx = np.shape(self.h5file["rawdata"])[0]
        Ly = end_sample-start_sample #np.shape(rawdata)[1]
        self.tile_start_x = determine_chunk_startpts(Lx, chunk_size_x, min_overlap)
        self.tile_start_y = determine_chunk_startpts(Ly, chunk_size_y, min_overlap,
                                                start_coord=start_sample)
        
        self.ntiles_x = len(self.tile_start_x)
        self.ntiles_y = len(self.tile_start_y)


    def __getitem__(self, i):

        if self.ids_to_include is not None:
            id = self.ids_to_include[i]
        else:
            id = i
        
        i_start = id // self.ntiles_y
        j_start = id % self.ntiles_y
        p_start = self.tile_start_x[i_start]
        s_start = self.tile_start_y[j_start]

        radar_image = np.empty((2, self.chunk_size_x, self.chunk_size_y))
        radar_image[0,:,:] = np.real(self.h5file["rawdata"][p_start:p_start+self.chunk_size_x, s_start:s_start+self.chunk_size_y])
        radar_image[1,:,:] = np.imag(self.h5file["rawdata"][p_start:p_start+self.chunk_size_x, s_start:s_start+self.chunk_size_y])

        radar_image = torch.tensor(radar_image, dtype=torch.float32)
        label = torch.tensor(-1)
        id = torch.tensor(id)

        return(radar_image, label, id)
    
    def __len__(self):
        return(self.ntiles_x*self.ntiles_y)

    def __del__(self):
        
        # Close hdf5 file
        self.h5file.close()


class cnn_3conv_1full_2mp_mod4(nn.Module):
    """PyTorch class that defines and implements CNN architecture used for ML head echoes 2023 paper.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define all functions that the neural network will use for each layer
        self.convlay1 = nn.Conv2d(2, 15, 3, padding=padding) # Convolutional layers...
        self.pool = nn.MaxPool2d(2, 2)  # 2-by-2 maxpool
        self.convlay2 = nn.Conv2d(15, 50, 3, padding=padding)
        self.convlay3 = nn.Conv2d(50, 300, 3, padding=padding)
        self.full1 = nn.Linear(300, 2) # Fully connected layer

        # Define each convolutional layer
        self.layer1 = nn.Sequential(self.convlay1, nn.LeakyReLU(), self.pool)
        self.layer2 = nn.Sequential(self.convlay2, nn.LeakyReLU(), self.pool)
        self.layerFinal = nn.Sequential(self.convlay3, nn.LeakyReLU())  # Must be named "layerFinal" for CAM-generating functions to work properly


    def forward(self, x):
        
        x = self.layer1(x)  # Conv. 1 + activation + max-pool
        x = self.layer2(x)  # Conv. 2 + activation + max-pool
        x = self.layerFinal(x)  # Conv. 3 + activation + max-pool
        x = func.max_pool2d(x, kernel_size=x.size()[2:])
        #x = func.avg_pool2d(x, kernel_size=x.size()[2:])  # Uncomment to try avg pool instead of maxpool (never seemed to work)
        x = torch.flatten(x,1)  # Flatten to 1D vector
        x = self.full1(x) # Fully connected layer 1
        
        return x


class cnn_4conv_1full_v2(nn.Module):
    """4-layer architecture tested. Did not seem to produce any better results than a 3-layer architecture.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define all functions that the neural network will use for each layer
        self.convlay1 = nn.Conv2d(2, 10, 3, padding=padding) # Convolutional layers...
        self.pool = nn.MaxPool2d(2, 2)
        self.convlay2 = nn.Conv2d(10, 30, 3, padding=padding)
        self.convlay3 = nn.Conv2d(30, 100, 3, padding=padding)
        self.convlay4 = nn.Conv2d(100, 300, 3, padding=padding)
        self.full1 = nn.Sequential(nn.Linear(300, 30), nn.LeakyReLU()) # Two fully connected layers...
        self.full2 = nn.Linear(30, 2)

        self.layer1 = nn.Sequential(self.convlay1, nn.LeakyReLU(), self.pool)
        self.layer2 = nn.Sequential(self.convlay2, nn.LeakyReLU(), self.pool)
        self.layer3 = nn.Sequential(self.convlay3, nn.LeakyReLU(), self.pool)
        self.layerFinal = nn.Sequential(self.convlay4, nn.LeakyReLU())


    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layerFinal(x)
        x = func.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x,1)
        x = self.full1(x)
        x = self.full2(x)
        
        return x


def train_net(nnet, data, optimizer, loss_func, device, loss_tot=0.0):
    """Function to perform one optimization step of training nnet (an instance of a CNN
    architecture class as defined above) using one batch of training data. The data
    argument is a single enumeration of a PyTorch data loader class.
    """

    # Put the neural network into training mode
    nnet.train()
    
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels, ids = data
    inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU if needed

    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward propagate + backward propagate + perform optimization step
    outputs = nnet(inputs)
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Update loss based on these iterations
    loss_tot += float(loss.item())
    
    return(loss_tot)



def test_net(nnet, data_loader, loss_func, device, eval_frac=1, thresh=0.5):
    """
    Function to test neural net performance given some data + loss fcn

    eval_frac is the fraction of the dataset to test
    """
    
    # Put the neural network into evaluation mode
    nnet.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs    
    with torch.no_grad():
        
        loss_tot = 0.0
        correct_labels = 0
        total_labels = 0

        # Get number of batches to evaluate
        num_batches_eval = int(np.round(len(data_loader)*eval_frac))

        i = 0
        for data in data_loader:
            
            images, labels, ids = data
            images, labels = images.to(device), labels.to(device)
            
            # Forward propagate test data
            outputs = nnet(images)
            loss = loss_func(outputs, labels) # Calculate loss
            loss_tot += float(loss.item())
            
            # Determine prediction via given threshold
            meteor_prob = outputs.data[:,1]
            predicted = meteor_prob >= thresh
            total_labels += labels.size(0)
            correct_labels += (predicted == labels).sum().item()

            i += 1
            if i >= num_batches_eval:
                break

    
    return(loss_tot, correct_labels, total_labels)


def predict_new_data(nnet, test_set_path, device, assess_correctness=True, thresh=0.5, batch_size=None, ids_to_include=None, load_directly=False,
                     chunk_size_x=None, chunk_size_y=None, min_overlap=None, facil_spec_params=None, verbose=False):
    """Uses neural network to predict based on input data with no labels """

    # Put neural network in evaluation mode
    nnet.eval()

    # Specify whether we should load examples directly from larger sections of rawdata (i.e. split a larger section into 
    #   the appropriate segment sizes in real time), or load pre-defined segments of the appropriate size
    if load_directly:
        test_set = RadarImageDatasetDirect(test_set_path, chunk_size_x, chunk_size_y, min_overlap=min_overlap, start_sample=facil_spec_params["start_sample"],
                                           end_sample=facil_spec_params["end_sample"])
    else:
        test_set = RadarImageDatasetHDF5Specific(test_set_path, ids_to_include=ids_to_include, transform=transforms.ToTensor())

    # If batch_size for testing not specified, have it just be the total length of the test set
    if batch_size is None:
        print("Batch size ", batch_size)
        batch_size=len(test_set)

    # Get data loader object for the test set, where test_set is the PyTorch dataset object
    data_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size)

    # Get total number of examples to predict on
    num_batches = len(data_loader)
    
    predicts_list = []
    corrects_list = []
    probs_meteor = []
    ids_full = []
    labels_full = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(data_loader):
            
            images, labels, ids = data
            images, labels = images.to(device), labels.to(device)

            # Forward propagate test data
            outputs = nnet(images)
            
            # Determine prediction via given threshold
            meteor_prob = torch.softmax(outputs.data, dim=1)[:,1]  # outputs.data[:,1]
            predicted = meteor_prob >= thresh

            if assess_correctness:
                correctness = (predicted == labels) # TODO: need option to disable correctness (or return dummy value) when labels are unavailable

            predicts_list.append(predicted)
            corrects_list.append(correctness)
            probs_meteor.append(meteor_prob)
            ids_full.append(ids)
            labels_full.append(labels)
            
            if verbose:
                print(f"Finished predicting on batch {i+1} of {num_batches}.")
            
            
    predicted = torch.cat(predicts_list)
    correctness = torch.cat(corrects_list)
    prob_meteor = torch.cat(probs_meteor)
    ids_full = torch.cat(ids_full)
    labels_full = torch.cat(labels_full)
    
    # Sort ids in ascending order (since the dataloader shuffles them)
    ids_sort, indices = torch.sort(ids_full)
    predicted = predicted[indices]
    prob_meteor = prob_meteor[indices]
    correctness = correctness[indices]
    labels_full = labels_full[indices]
    
    return(predicted, prob_meteor, ids_sort, correctness, labels_full)

    
def get_num_params(model):
    """Function to get the total number of parameters optimized in the neural network."""
    
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    
    return(total_params)


def train_validate(data_dir, train_file, valid_file, model, device, model_name, num_epochs, valid_real_file=None,
                   lr=0.001, batch_size=20, output_loss_interval=20, weight_decay=0.0001, resume=False):
    """
    Train neural network on synthetic training data set, and evaluate it on synthetic validation data set as 
      a measure of training progress.

      lr is the learning rate
      output_loss_interval is the number of batches/optimization steps performed between validations.
        Must have the total number of training examples be divisible by batch_size*output_loss_interval,
        since this makes it a lot cleaner to plot training progress etc.
    """
    
    ## Make output directories for trained model files (weights) and any plots
    os.makedirs(data_dir + "/trained_models/", exist_ok=True)
    os.makedirs(data_dir + "/plots/", exist_ok=True)

    ## Get training, validation, and test sets
    train_set = RadarImageDatasetHDF5(train_file, transform=transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    
    num_train = len(train_set)
    num_loss_per_epoch = num_train // (batch_size*output_loss_interval)
    assert (num_train / (batch_size*output_loss_interval)).is_integer(), "Need num_train to be divisible by batch_size*output_loss_interval"
    
    valid_set = RadarImageDatasetHDF5(valid_file, transform=transforms.ToTensor())
    valid_dataloader = torch.utils.data.DataLoader(valid_set, shuffle=False, batch_size=TEST_BATCH_SIZE)
    
    # Use the Adam optimization function + cross-entropy loss.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()

    # Move model/weights to GPU
    model.to(device)

    # Specify filenames of intermediate files used for saving/loading training progress
    # TODO: this could be done much more cleanly
    trained_model_dir = f"{data_dir}/trained_models/"
    loss_train_dir = f"{trained_model_dir}{model_name}-loss-train.npy"
    loss_valid_dir = f"{trained_model_dir}{model_name}-loss-valid.npy"
    correctness_train_dir = f"{trained_model_dir}{model_name}-correctness-train.npy"
    correctness_valid_dir = f"{trained_model_dir}{model_name}-correctness-valid.npy"
    current_epoch_dir = f"{trained_model_dir}{model_name}-num_epochs.npy"

    f1_s_dir = f"{trained_model_dir}{model_name}-f1_s.npy"
    f1_r_dir = f"{trained_model_dir}{model_name}-f1_r.npy"
    accuracy_s_dir = f"{trained_model_dir}{model_name}-accuracy_s.npy"
    accuracy_r_dir = f"{trained_model_dir}{model_name}-accuracy_r.npy"
    recall_s_dir = f"{trained_model_dir}{model_name}-recall_s.npy"
    recall_r_dir = f"{trained_model_dir}{model_name}-recall_r.npy"

    # Load all necessary data if we are resuming a previous training session
    if resume:
        loss_arrays_prev = np.load(loss_train_dir)
        loss_arrays_valid_prev = np.load(loss_valid_dir)
        correctness_train_prev = np.load(correctness_train_dir)
        correctness_valid_prev = np.load(correctness_valid_dir)
        epoch = np.load(current_epoch_dir)

        f1_s_prev = np.load(f1_s_dir)
        f1_r_prev = np.load(f1_r_dir)
        accuracy_s_prev = np.load(accuracy_s_dir)
        accuracy_r_prev = np.load(accuracy_r_dir)
        recall_s_prev = np.load(recall_s_dir)
        recall_r_prev = np.load(recall_r_dir)
    # If not resuming, start at epoch zero 
    else:
        epoch = 0
    
    # Define all lists of performance metrics during training
    loss_arrays = []
    loss_arrays_valid = []
    correctness_arrays_train = []
    correctness_arrays_valid = []
    f1_s_arrs = []
    f1_r_arrs = []
    precision_s_arrs = []
    precision_r_arrs = []
    recall_s_arrs = []
    recall_r_arrs = []
    specificity_s_arrs = []
    specificity_r_arrs = []
    accuracy_s_arrs = []
    accuracy_r_arrs = []

    # Determine epoch number at which to end training
    epoch_end = epoch + num_epochs
    
    # Time the training process
    time_start_train_cycle = time.time()

    # Run desired epochs
    while epoch < epoch_end:
    
        # will track loss in this interval of training
        loss_this_interval = 0.0
        
        # Prepare Numpy arrays for performance metrics
        loss_array = np.empty(num_loss_per_epoch)
        loss_array_valid = np.empty(num_loss_per_epoch)
        correctness_array_train = np.empty(num_loss_per_epoch)
        correctness_array_valid = np.empty(num_loss_per_epoch)
        f1_s_arr = np.empty(num_loss_per_epoch)
        f1_r_arr = np.empty(num_loss_per_epoch)
        precision_s_arr = np.empty(num_loss_per_epoch)
        precision_r_arr = np.empty(num_loss_per_epoch)
        recall_s_arr = np.empty(num_loss_per_epoch)
        recall_r_arr = np.empty(num_loss_per_epoch)
        specificity_s_arr = np.empty(num_loss_per_epoch)
        specificity_r_arr = np.empty(num_loss_per_epoch)
        accuracy_s_arr = np.empty(num_loss_per_epoch)
        accuracy_r_arr = np.empty(num_loss_per_epoch)

        # Iterate through batches of training data
        for i,data in enumerate(train_dataloader, 0):
            
            # Perform one training step and get resulting loss from loss function
            loss_this_interval = train_net(model, data, optimizer, loss_func, device, loss_this_interval)
        
            # Perform validation every output_loss_interval mini-batches
            if i % output_loss_interval == (output_loss_interval-1):
                
                j = i // output_loss_interval
                loss_val = loss_this_interval/output_loss_interval
                
                time_end_train_cycle = time.time()

                loss_array[j] = loss_val
                loss_this_interval = 0.0
                
                # Time the training set test
                time_start_train = time.time()
                num_train_eval = 1000
                eval_frac = num_train_eval/num_train
                # Test on synthetic training data
                loss_train, correct_train, total_train = test_net(model, train_dataloader, loss_func, device, eval_frac=eval_frac)
                time_end_train = time.time()

                # Time the validation set test
                time_start_valid = time.time()
                # Test on synthetic validation data (this just gets correctness, no performance metrics)
                loss_valid, correct_labels, total_labels = test_net(model, valid_dataloader, loss_func, device)
                time_end_valid = time.time()
                
                # Do validation on synthetic data via function that returns necessary data for performance metrics
                #   TODO: this repeats some steps done immediately above, can refactor a bit
                predicted_s, prob_meteor_s, ids_s, correctness_s, labels_s = predict_new_data(model, valid_file, device, batch_size=TEST_BATCH_SIZE)
                predicted_s = predicted_s.to(torch.device("cpu")) # Move data back to CPU for postprocessing
                labels_s = labels_s.to(torch.device("cpu"))
                label_table_s = np.array(torch.stack((ids_s, predicted_s)).T)
                truth_table_s = np.array(torch.stack((ids_s, labels_s)).T)

                # Determine performance metrics for the synthetic data tested
                num_true_pos_s, num_true_neg_s, num_false_pos_s, num_false_neg_s, precision_s, \
                    recall_s, specificity_s, f1_s, total_accuracy_s = postprocess_cnn_results(label_table_s, truth_table_s, plot_synth_dir)

                # Do a validation on some real data, and record all the associated parameters
                #   Note that the real "validation" set in this step must be separate from the designated testing set!
                predicted_r, prob_meteor_r, ids_r, correctness_r, labels_r = predict_new_data(model, valid_real_file, device, batch_size=TEST_BATCH_SIZE)
                predicted_r = predicted_r.to(torch.device("cpu"))
                labels_r = labels_r.to(torch.device("cpu"))
                label_table_r = np.array(torch.stack((ids_r, predicted_r)).T)
                truth_table_r = np.array(torch.stack((ids_r, labels_r)).T)
                # Determine performance metrics for the real data tested
                num_true_pos_r, num_true_neg_r, num_false_pos_r, num_false_neg_r, precision_r, \
                    recall_r, specificity_r, f1_r, total_accuracy_r = postprocess_cnn_results(label_table_r, truth_table_r, plot_synth_dir)
                
                # Put performance metrics in numpy arrays to be saved/plotted
                f1_s_arr[j] = f1_s
                f1_r_arr[j] = f1_r
                precision_s_arr[j] = precision_s
                precision_r_arr[j] = precision_r
                recall_s_arr[j] = recall_s
                recall_r_arr[j] = recall_r
                specificity_s_arr[j] = specificity_s
                specificity_r_arr[j] = specificity_r
                accuracy_s_arr[j] = total_accuracy_s
                accuracy_r_arr[j] = total_accuracy_r
                loss_array_valid[j] = loss_valid
                correctness_array_train[j] = correct_train/total_train
                correctness_array_valid[j] = correct_labels/total_labels
                
                # Save model in its current state, for resumability
                model_path = f"{data_dir}/trained_models/model-{model_name}-{epoch}-{j}.pt"
                torch.save(model, model_path)

                # Print training stats/performance metrics to terminal
                print(f"Epoch {epoch + 1}, Minibatch {i + 1:5d}, Loss: {loss_val:.4f}, Validation (synthetic) F1: {f1_s:.4f}, Validation (real) F1: {f1_r:.4f}, \n"
                      f"Training Time: {time_end_train_cycle-time_start_train_cycle:.2f} s" 
                        f", Train test time: {time_end_train-time_start_train:.2f} s, Validation test time: {time_end_valid-time_start_valid:.2f} s")

                # Start the clock on next training cycle
                time_start_train_cycle = time.time()


        # Add performance metrics from this epoch to a list
        loss_arrays.append(loss_array)
        loss_arrays_valid.append(loss_array_valid)
        correctness_arrays_train.append(correctness_array_train)
        correctness_arrays_valid.append(correctness_array_valid)
        f1_s_arrs.append(f1_s_arr)
        f1_r_arrs.append(f1_r_arr)
        precision_s_arrs.append(precision_s_arr)
        precision_r_arrs.append(precision_r_arr)
        recall_s_arrs.append(recall_s_arr)
        recall_r_arrs.append(recall_r_arr)
        specificity_s_arrs.append(specificity_s_arr)
        specificity_r_arrs.append(specificity_r_arr)
        accuracy_s_arrs.append(accuracy_s_arr)
        accuracy_r_arrs.append(accuracy_r_arr)
        
        # Finished with this epoch
        epoch += 1
    
    ## Save model parameters, given hyperparameters
    loss_array = np.array(loss_arrays)
    loss_array_valid = np.array(loss_arrays_valid)
    correctness_array_train = np.array(correctness_arrays_train)
    correctness_array_valid = np.array(correctness_arrays_valid)
    f1_s_arrs = np.array(f1_s_arrs)
    f1_r_arrs = np.array(f1_r_arrs)
    precision_s_arrs = np.array(precision_s_arrs)
    precision_r_arrs = np.array(precision_r_arrs)
    recall_s_arrs = np.array(recall_s_arrs)
    recall_r_arrs = np.array(recall_r_arrs)
    specificity_s_arrs = np.array(specificity_s_arrs)
    specificity_r_arrs = np.array(specificity_r_arrs)
    accuracy_s_arrs = np.array(accuracy_s_arrs)
    accuracy_r_arrs = np.array(accuracy_r_arrs)

    num_epochs = epoch

    # If we are resuming from prior training, add performance metrics to existing lists
    if resume:
        loss_array = np.concatenate((loss_arrays_prev, loss_array), axis=0)
        loss_array_valid = np.concatenate((loss_arrays_valid_prev, loss_array_valid), axis=0)
        correctness_array_train = np.concatenate((correctness_train_prev, correctness_array_train), axis=0)
        correctness_array_valid = np.concatenate((correctness_valid_prev, correctness_array_valid), axis=0)
        f1_s_arrs = np.concatenate((f1_s_prev, f1_s_arrs), axis=0)
        f1_r_arrs = np.concatenate((f1_r_prev, f1_r_arrs), axis=0)
        accuracy_s_arrs = np.concatenate((accuracy_s_prev, accuracy_s_arrs), axis=0)
        accuracy_r_arrs = np.concatenate((accuracy_r_prev, accuracy_r_arrs), axis=0)
        recall_s_arrs = np.concatenate((recall_s_prev, recall_s_arrs), axis=0)
        recall_r_arrs = np.concatenate((recall_r_prev, recall_r_arrs), axis=0)
        
    # Save all performance metrics
    np.save(loss_train_dir, loss_array)
    np.save(loss_valid_dir, loss_array_valid)
    np.save(correctness_train_dir, correctness_array_train)
    np.save(correctness_valid_dir, correctness_array_valid)
    np.save(current_epoch_dir, num_epochs)
    np.save(f1_s_dir, f1_s_arrs)
    np.save(f1_r_dir, f1_r_arrs)
    np.save(accuracy_s_dir, accuracy_s_arrs)
    np.save(accuracy_r_dir, accuracy_r_arrs)
    np.save(recall_s_dir, recall_s_arrs)
    np.save(recall_r_dir, recall_r_arrs)
    

    # Make and save plot of training and validation loss vs epoch
    acc_xaxis = np.arange(0, num_epochs*num_loss_per_epoch)/num_loss_per_epoch
    plt.figure()
    plt.plot(acc_xaxis, np.ndarray.flatten(loss_array), label="train loss")
    plt.plot(acc_xaxis, np.ndarray.flatten(loss_array_valid), label="validation loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim([0,0.5])
    if model_name is not None:
        plt.savefig(data_dir + "/plots/loss-" + model_name + ".png")
    plt.figure()
    plt.plot(acc_xaxis, np.ndarray.flatten(correctness_array_train), label="train accuracy")
    plt.plot(acc_xaxis, np.ndarray.flatten(correctness_array_valid), label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if model_name is not None:
        plt.savefig(data_dir + "/plots/accuracy-" + model_name + ".png")
    
    plt.figure()
    plt.plot(acc_xaxis, np.ndarray.flatten(f1_s_arrs), "r--", label="f1, synthetic")
    plt.plot(acc_xaxis, np.ndarray.flatten(f1_r_arrs), "r", label="f1, real")
    plt.plot(acc_xaxis, np.ndarray.flatten(accuracy_s_arrs), "g--", label="accuracy, synthetic")
    plt.plot(acc_xaxis, np.ndarray.flatten(accuracy_r_arrs), "g", label="accuracy, real")
    plt.plot(acc_xaxis, np.ndarray.flatten(recall_s_arrs), "b--", label="recall, synthetic")
    plt.plot(acc_xaxis, np.ndarray.flatten(recall_r_arrs), "b", label="recall, real")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if model_name is not None:
        plt.savefig(data_dir + "/plots/trainstats-" + model_name + ".png")

    plt.close("all")
    

    # Save model parameters/weights to file
    model_path = data_dir + "/trained_models/" + "/model-" + model_name + ".pt"
    torch.save(model, model_path)

    return(model)


def returnCAM(feature_conv, weight_softmax, class_id):
    """Function to return class activation map image.
    Adapted from https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
    """

    bz, nc, h, w = feature_conv.shape

    # Calculate CAM from forward propagation intercepted before global maxpool
    #   and weights from the fully-connected layer
    cam = weight_softmax[class_id].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)

    return cam


def powerLog(data, noisefloor = 0):
    """
    Helper function to get power in dB of complex data array. Noisefloor is in dB.
    """
    
    power = np.abs(data)
    power[power <= 1e-10] = 1e-10 # Ensure log(nonpositive) does not occur
    power = 20*np.log10(power) - noisefloor
    
    return(power)


def create_cam_plot(CAMs_pos, image, truth, pred, mean_val, std_val, facility, title_string=""):
    """Function to generate final CAM plot as a Matplotlib figure/axis, using CAM data already
    generated, along with the radar image itself
    """
    
    # Convert from PyTorch tensors to Numpy arrays and transpose as necessary for plotting
    CAMs_pos = np.transpose(np.array(CAMs_pos))

    # Get grid of CAM
    nx = np.shape(CAMs_pos)[1]
    ny = np.shape(CAMs_pos)[0]
    x = np.linspace(0,nx-1,nx)
    y = np.linspace(0,ny-1,ny)
    X, Y = np.meshgrid(x,y)

    # Specify colorbar for CAM contours    
    ylorrd = plt.cm.get_cmap('YlOrRd', 100)
    
    # Generate plot objects
    fig, ax = plt.subplots(1,1, figsize=(5,5))

    # Specify contour levels to include in CAM
    levels = np.array([mean_val + std_val, mean_val + 2*std_val, mean_val + 3*std_val, mean_val + 4*std_val])
    
    # Generate contour plot
    contours = ax.contour(X, Y, CAMs_pos, levels, colors=ylorrd(np.linspace(0,1,len(levels))))
    
    # Generate RTI image. Contours are displayed on top of this image.
    rawdata = np.transpose(np.array(image[0,0,:,:] + 1j*image[0,1,:,:]))
    image_data = powerLog(rawdata, noisefloor=facility["NOISE_RAW"])
    ax.imshow(image_data, interpolation='nearest', cmap=plt.cm.plasma, origin='lower', extent=(0, nx-1, 0, ny-1),
            vmin=0, vmax=40)

    # Add title above figure    
    fig.suptitle(f"{title_string}, \n truth={truth}, label={pred}")

    return(fig, ax)


if __name__ == "__main__":

    ### INPUTS ###
    cuda_enabled = True
    test_only = True
    num_epochs_to_run = 4
    batch_size = 100
    output_loss_interval = 50
    learning_rate = 0.001  # Initial learning rate
    #learning_rate = 0.0005
    #learning_rate = 0.0001
    #learning_rate = 0.00005
    #learning_rate = 0.00001  # Final learning rate
    weight_decay = learning_rate/10

    # Uncomment whichever facility is being used
    facility = "RISR-N"
    #facility = "JRO"
    #facility = "MHO"

    ### DO NOT MODIFY ANYTHING BELOW THIS

    # Specify list of test sets depending on facility
    if facility == "RISR-N":
        test_sets = ["day1_hour0_partial", "day1_hour1_partial"]
    elif facility == "MHO":
        test_sets = ["day1_hour0_partial", "day1_hour1_partial"]
    else:
        test_sets = ["day1_hour0_partial", "day1_hour3_partial_EEJ_1"]

    #test_set = "day1_hour3_partial_EEJ_1"
    #test_sets = ["day1_hour0_partial", "day1_hour1_partial"]
    #test_set_main = test_sets[0]
    #test_sets = ["day1_hour0_partial"]
    test_set_main = test_sets[0]

    synthetic_data_dir = f"{facility}/synthetic_data/"
    test_data_dir = f"{facility}/test_sets/{test_set_main}/data/"

    test_data_dirs = []
    test_real_files = []
    for i,test_set in enumerate(test_sets):
        test_data_dir_i = f"{facility}/test_sets/{test_set}/data/"
        test_data_dirs.append(test_data_dir_i)
        test_real_files.append(f"{test_data_dir_i}/test_data.h5")

    model_class = cnn_3conv_1full_2mp_mod4
    model_str = "cnn_3conv_1full_2mp_mod4"
    model_name = f"{facility}-{model_str}"
    truth_file = f"{facility}/test_sets/{test_set_main}/data/truth.csv"
    plot_synth_dir = f"{facility}/test_sets/{test_set_main}/plot_synth/"
    plot_dir = f"{facility}/test_sets/{test_set_main}/plot/"
    

    
    ### Run/evaluate CNN ###

    os.makedirs(plot_dir, exist_ok=True)
    
    model = model_class()
    print("number of parameters in network: ", get_num_params(model))

    train_file = f"{synthetic_data_dir}/train.h5"
    valid_file = f"{synthetic_data_dir}/valid.h5"
    test_file = f"{synthetic_data_dir}/test.h5"
    test_real_file = f"{test_data_dir}/test_data.h5"
    model_path = f"{synthetic_data_dir}/trained_models/model-{model_name}.pt"

    # Specify whether to use CUDA.
    if torch.cuda.is_available() and cuda_enabled:
        device = torch.device("cuda")
    elif not torch.cuda.is_available() and cuda_enabled:
        print("Warning! GPU not detected/available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    if not os.path.exists(model_path):
        model = model_class()
        resume = False
    else:
        model = torch.load(model_path)
        resume = True
    
    if not test_only:
        time_start = time.time()
        model = train_validate(synthetic_data_dir, train_file, valid_file, model, device, model_name=model_name,
                        num_epochs=num_epochs_to_run, batch_size=batch_size, output_loss_interval=output_loss_interval,
                        lr=learning_rate, weight_decay=weight_decay, resume=resume, valid_real_file=test_real_file)
        time_end = time.time()
        print("Time to train model: ", time_end-time_start)
    

    # Test model
    #test_existing_model(model, test_file, hdf5=True)
    

    # Predict on synthetic and real data. Used to generate CAMs.
    # TODO: this is a weird way to do it...
    predicted_s, prob_meteor_s, ids_s, correctness_s, labels_s = predict_new_data(model, test_file, device, batch_size=TEST_BATCH_SIZE)

    # Transfer back to CPU
    predicted_s = predicted_s.to(torch.device("cpu"))
    labels_s = labels_s.to(torch.device("cpu"))

    label_table_s = np.array(torch.stack((ids_s, predicted_s)).T)
    truth_table_s = np.array(torch.stack((ids_s, labels_s)).T)

    # Postprocess results
    print("Final results for synthetic data... ")
    postprocess_cnn_results(label_table_s, truth_table_s, plot_synth_dir, verbose=True)

    # Predict for all real test sets given.
    label_tables = []
    truth_tables = []
    num_above_thresh_tot = 0
    num_below_thresh_tot = 0
    sensitivity_thresh_above_tot = 0
    sensitivity_thresh_below_tot = 0
    for i,test_set in enumerate(test_sets):

        print(f"For test set {test_set}...")

        test_real_file = test_real_files[i]
        test_data_dir = test_data_dirs[i]

        predicted, prob_meteor, ids, correctness, labels_r = predict_new_data(model, test_real_file, device, batch_size=TEST_BATCH_SIZE)

        # Transfer back to CPU
        labels_r = labels_r.to(torch.device("cpu"))
        predicted = predicted.to(torch.device("cpu"))

        # Save a CSV with all positive labels and all negative labels, if it doesn't already exist
        label_file = f"{test_data_dir}/labels.csv"
        label_table = np.array(torch.stack((ids, predicted)).T)
        np.savetxt(label_file, label_table, header="id, label")

        # Append stats to full list of examples
        label_tables.append(label_table)
        truth_tables.append(labels_r)

        # For each test set, perform analysis of false negative rate vs. snr threshold
        data_csv = f"data_new/{facility}/{test_set}/labels_table.csv"

        # Load list of SNR values from Pandas
        data = pd.read_csv(data_csv)
        ids_pos = data["chunk_id"].values
        labels_pos = data["label"].values
        snr = data["avg_snr"].values

        # Get ids_pos and snr only where labels_pos == 1
        ids_pos = ids_pos[labels_pos == 1]
        snr = snr[labels_pos == 1]

        dB_thresh_above = 15
        greater_than = True
        sensitivity_thresh_above, num_above_thresh = get_false_neg_rate(predicted.numpy(), labels_r.numpy(), ids_pos, labels_pos, snr, dB_thresh_above, greater_than=greater_than)
        sensitivity_thresh_above_tot += num_above_thresh*sensitivity_thresh_above
        num_above_thresh_tot += num_above_thresh
        print(f"Sensitivity above {dB_thresh_above}: {sensitivity_thresh_above} with {num_above_thresh} total\n")

        dB_thresh_below = 15
        greater_than = False
        sensitivity_thresh_below, num_below_thresh = get_false_neg_rate(predicted.numpy(), labels_r.numpy(), ids_pos, labels_pos, snr, dB_thresh_below, greater_than=greater_than)
        sensitivity_thresh_below_tot += num_below_thresh*sensitivity_thresh_below
        num_below_thresh_tot += num_below_thresh
        print(f"Sensitivity below {dB_thresh_below}: {sensitivity_thresh_below} with {num_below_thresh} total\n")
    
    # Get total average of each threshold sensitivity
    print(f"For all test sets combined...")
    sensitivity_thresh_above_tot /= num_above_thresh_tot
    sensitivity_thresh_below_tot /= num_below_thresh_tot
    print(f"Sensitivity above {dB_thresh_above}: {sensitivity_thresh_above_tot} with {num_above_thresh_tot} total\n")
    print(f"Sensitivity below {dB_thresh_below}: {sensitivity_thresh_below_tot} with {num_below_thresh_tot} total\n")

    # Put all tables together
    label_table_full = np.concatenate(label_tables, axis=0)[:,1]
    truth_table_full = np.concatenate(truth_tables, axis=0)


    print("Final results for real data, with all test sets combined... ")
    postprocess_cnn_results(label_table_full, truth_table_full, plot_dir, verbose=True)



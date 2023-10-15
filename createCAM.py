import numpy as np
import torch
import os
from cnn_simple import create_cam_plot, returnCAM, RadarImageDatasetHDF5Specific, cnn_3conv_1full_2mp_mod4, predict_new_data
import torch.nn.functional as func
import matplotlib.pyplot as plt
from torch import topk
from torchvision import transforms
from CEDAR2019_params_simple import facilParams


def create_CAM_multiple(model_dir, model_name, test_real_file, ids_to_include, plot_dir, facility, device):
    """Function to create class activation maps (CAMs) with one plot per saved model during training process,
    to show evolution of the CAM as the CNN trains. Given a CNN model base name, test data HDF5, and a list
    of test data example IDs to make CAMs for. Requires CNN models saved during entire training process."""

    model_filename = f"model-{model_name}-"

    # Get list of model files corresponding to each checkpoint
    model_file_list = []
    files_all = os.listdir(model_dir)
    for file in files_all:
        if file.startswith(model_filename):
            model_file_list.append(file)
    
    model_file_list.sort()
    print("List of model files to create CAMs for: ", model_file_list)


    # Create a directory for each example to plot
    for id in ids_to_include:
        os.makedirs(f"{plot_dir}/{id}/", exist_ok=True)

    # Loop over each model
    for model_file in model_file_list:

        # Get model name on its own
        model_name = model_file.split(".")[0]

        # Load model
        model = torch.load(f"{model_dir}/{model_file}")
        model.eval() # Ensure model in evaluation mode

        # Define hook to extract output from convolutional layers
        features_blobs = []
        def hook_feature(module, input, output):
            if len(features_blobs) == 0:
                features_blobs.append(output.data.cpu().numpy())
            else:
                features_blobs[0] = output.data.cpu().numpy()
        model._modules.get("layerFinal").register_forward_hook(hook_feature)
        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
        
        # Loop over each example...
        test_set = RadarImageDatasetHDF5Specific(test_real_file, ids_to_include=ids_to_include, transform=transforms.ToTensor())
        data_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=1)
        for i,data in enumerate(data_loader):

            image, labels, ids = data

            # forward pass through model
            outputs = model(image.to(device)).cpu()
            # get the softmax probabilities
            probs = func.softmax(outputs).data.squeeze()
            # get the class indices of top k probabilities
            class_idx = topk(probs, 1)[1].int()
            truth = labels[0].item()
            pred = class_idx[0].item()

            # generate class activation mapping for the top1 prediction
            CAMs_pos = returnCAM(features_blobs[0], weight_softmax, 1)

            # Convert to Numpy arrays and transpose as necessary for plotting
            CAMs_pos = np.transpose(np.array(CAMs_pos))

            # generate class activation mapping for the top1 prediction
            CAMs_pos = returnCAM(features_blobs[0], weight_softmax, 1)

            # Get "mean value" from this
            mean_val = np.mean(CAMs_pos)
            std_val = np.std(CAMs_pos)

            # Create plot and save
            fig, ax = create_cam_plot(CAMs_pos, image, truth, pred, mean_val, std_val, facility, title_string=f"Model: {model_name}")

            save_name = f"{plot_dir}/{ids[0]}/{model_name}.png"
            fig.savefig(save_name)
            plt.close(fig)

        

facility_name = "RISR-N"
model_str = "cnn_3conv_1full_2mp_mod4"
realdata = True
model_name = f"{facility_name}-{model_str}"
synthetic_data_dir = f"{facility_name}/synthetic_data/"
model_path = f"{synthetic_data_dir}/trained_models/"
ids_to_include = [496]
include_all_interesting = True  # Overrides ids_to_include
facility=facilParams[facility_name]
if realdata:
    test_set_str = "day1_hour1_partial"
    test_real_file = f"{facility_name}/test_sets/{test_set_str}/data/test_data.h5"
    cam_plot_dir = f"{facility_name}/synthetic_data/plots/CAMS/{test_set_str}/{model_name}/"
else:
    filename = "valid.h5"
    test_real_file = f"{facility_name}/synthetic_data/{filename}"
    cam_plot_dir = f"{facility_name}/synthetic_data/plots/CAMS/{model_name}/"

# Specify whether to use CUDA.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif not torch.cuda.is_available():
    print("Warning! GPU not detected/available. Using CPU.")
    device = torch.device("cpu")
else:
    device = torch.device("cpu")


# Only generate CAM if (1) it's a true positive, (2) false positive or (3) false negative.
#   No need for CAMs of true negatives
if include_all_interesting:

    # Load final model
    model = torch.load(f"{model_path}/model-{model_name}.pt")

    # Predict on test set
    predicted, prob_meteor, ids, correctness, labels = predict_new_data(model, test_real_file, device, batch_size=500)
    predicted = predicted.to(torch.device("cpu")).numpy()
    labels = labels.to(torch.device("cpu")).numpy()

    # Get instances where predictions are positive OR labels are positive/inconclusive
    ids_true_positive = np.where(np.all(np.stack((predicted == 1, labels == 1)), axis=0))[0]
    ids_false_positive = np.where(np.all(np.stack((predicted == 1, labels == 0)), axis=0))[0]
    ids_false_negative = np.where(np.all(np.stack((predicted == 0, labels == 1)), axis=0))[0]

    # Create CAMs
    cam_true_pos_dir = f"{cam_plot_dir}/true_pos/"
    cam_false_pos_dir = f"{cam_plot_dir}/false_pos/"
    cam_false_neg_dir = f"{cam_plot_dir}/false_neg/"
    os.makedirs(cam_true_pos_dir, exist_ok=True)
    os.makedirs(cam_false_pos_dir, exist_ok=True)
    os.makedirs(cam_false_neg_dir, exist_ok=True)

    create_CAM_multiple(model_path, model_name, test_real_file, ids_true_positive,
                        cam_true_pos_dir, facility, device)
    create_CAM_multiple(model_path, model_name, test_real_file, ids_false_positive,
                        cam_false_pos_dir, facility, device)
    create_CAM_multiple(model_path, model_name, test_real_file, ids_false_negative,
                        cam_false_neg_dir, facility, device)
else:
    create_CAM_multiple(model_path, model_name, test_real_file, ids_to_include,
                        cam_plot_dir, facility, device)



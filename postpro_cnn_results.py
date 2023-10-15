import numpy as np
import os
import shutil


def postprocess_cnn_results(labels_data, truth_data, plot_dir=None, copy_plots=False, verbose=False):
    """Function to get performance metrics based on labels classified by CNN and truth labels. 
    Labels/truth labels may be loaded from text file or passed in as a Numpy array.
    """

    if type(labels_data) is str:
        labels_data = np.loadtxt(labels_data, skiprows=1)
    if type(truth_data) is str:
        truth_data = np.loadtxt(truth_data, delimiter=",", skiprows=1)

    if labels_data.ndim == 2:
        labels = labels_data[:,1]
        truth_labels = truth_data[:,1]
    else:
        labels = labels_data
        truth_labels = truth_data

    # Count total number of examples being checked
    num_total_truth = len(truth_labels)

    # Eliminate inconclusive cases from count (instead of considering them positive)
    truth_neg = (truth_labels == 0)
    truth_pos = (truth_labels == 1)
    truth_probably = (truth_labels == 2)
    truth_inconclusive = (truth_labels == 3)
    labels = labels[truth_inconclusive == False]
    truth_labels = truth_labels[truth_inconclusive == False]

    num_positive_truth = np.sum(truth_pos) + np.sum(truth_probably)
    num_inconclusive_truth = np.sum(truth_inconclusive)
    num_negative_truth = np.sum(truth_neg)

    # Generate confusion matrix
    true_positives = np.all(np.stack((labels, truth_labels)), axis=0)
    true_negatives = np.all(np.stack((labels==False, truth_labels==False)), axis=0)
    false_positives = np.all(np.stack((labels==True, truth_labels==False)), axis=0)
    false_negatives = np.all(np.stack((labels==False, truth_labels==True)), axis=0)

    num_true_pos = np.sum(true_positives)
    num_true_neg = np.sum(true_negatives)
    num_false_pos = np.sum(false_positives)
    num_false_neg = np.sum(false_negatives)

    if verbose:
        print(f"True positives: {num_true_pos}, \nTrue negatives: {num_true_neg}, \nFalse positives: {num_false_pos}, \nFalse negatives: {num_false_neg}\n")

    num_correct = num_true_pos + num_true_neg
    num_incorrect = num_false_pos + num_false_neg
    num_total = num_correct + num_incorrect

    precision = num_true_pos/(num_true_pos + num_false_pos)
    recall = num_true_pos/(num_true_pos + num_false_neg)
    specificity = num_true_neg/(num_true_neg + num_false_pos)
    f1 = 2*(precision*recall)/(precision + recall)
    total_accuracy = num_correct/num_total

    if verbose:

        print("\n")
        print(f"Number of positive examples: {num_positive_truth}")
        print(f"Number of inconclusive examples: {num_inconclusive_truth}")
        print(f"Number of negative examples: {num_negative_truth}")
        print(f"Total number of examples: {num_total_truth}")
        print("\n")

        print(f"Precision (true positives/all predicted positives) = {precision:.03f}")
        print(f"Recall/sensitivity (true positives/all labelled positives) = {recall:.03f}")
        print(f"Specificity (true negatives/all labelled negatives) = {specificity:.03f}")
        print(f"F1-score = {f1:.03f}")
        print(f"Total accuracy: {total_accuracy:.03f}")
        print("\n")

    if (plot_dir is not None) and copy_plots:
        # Copy plot images according to confusion matrix
        os.makedirs(f"{plot_dir}/true_pos/", exist_ok=True)
        os.makedirs(f"{plot_dir}/true_neg/", exist_ok=True)
        os.makedirs(f"{plot_dir}/false_pos/", exist_ok=True)
        os.makedirs(f"{plot_dir}/false_neg/", exist_ok=True)

        for i in range(num_total):

            if true_positives[i]:
                shutil.copy(f"{plot_dir}/rti-chunk-{str(i)}.png", f"{plot_dir}/true_pos/")
            elif true_negatives[i]:
                shutil.copy(f"{plot_dir}/rti-chunk-{str(i)}.png", f"{plot_dir}/true_neg/")
            elif false_positives[i]:
                shutil.copy(f"{plot_dir}/rti-chunk-{str(i)}.png", f"{plot_dir}/false_pos/")
            elif false_negatives[i]:
                shutil.copy(f"{plot_dir}/rti-chunk-{str(i)}.png", f"{plot_dir}/false_neg/")

    return(num_true_pos, num_true_neg, num_false_pos, num_false_neg, precision, recall, specificity, f1, total_accuracy)



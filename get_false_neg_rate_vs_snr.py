import numpy as np

def get_false_neg_rate(predict_labels, truth_labels, ids_pos, labels_pos, snrs, snr_thresh, greater_than=True):

    true_pos = (truth_labels == 1)
    ids_pos_ = np.where(true_pos)[0]

    if greater_than:
        ids_thresh = ids_pos[snrs >= snr_thresh]
    else:
        ids_thresh = ids_pos[snrs <= snr_thresh]

    predict_labels_thresh = predict_labels[ids_thresh]

    num_thresh = len(ids_thresh)
    sensitivity_thresh = np.sum(predict_labels_thresh)/num_thresh

    return(sensitivity_thresh, num_thresh)


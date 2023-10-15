import numpy as np
import h5py

hdf5 = True

# Specify data set location.
facility = "RISR-N"
dataset_name = "day1_hour1_partial"
dataset_file = f"C:/Users/thedges/Dropbox/research/radar/computer_vision/{facility}/{dataset_name}/data/test_data.h5"
label_file = f"C:/Users/thedges/Dropbox/research/radar/computer_vision/{facility}/{dataset_name}/data/truth.csv"

# Load a list of labels applied to an existing unlabelled dataset. Creates a labelled version of same data.
labels_data = np.loadtxt(label_file, delimiter=",", skiprows=1, dtype=np.int32)
#neg_examples = np.loadtxt(f"{label_dir}/neg_labels.csv", delimiter=",", skiprows=1, dtype=np.int32)

#num_pos = np.shape(pos_examples)[0]
#num_neg = np.shape(neg_examples)[0]

ids = labels_data[:,0]
labels = labels_data[:,1]
clutter = labels_data[:,2]

num_pos = np.sum(labels==1)
num_neg = np.sum(labels==0)
num_probably_pos = np.sum(labels==2)
num_inconclusive = np.sum(labels==3)

print(f"{num_pos} positive labels. {num_neg} negative labels. {num_probably_pos} probably positive. "
      f"{num_inconclusive} inconclusive. {num_pos+num_neg+num_probably_pos+num_inconclusive} total.")


with h5py.File(dataset_file, "r+") as h5file:

    for i,id in enumerate(ids):

        group_name = f"{int(id):07d}-rawdata"
        h5file[group_name].attrs["label"] = labels[i]
        h5file[group_name].attrs["clutter"] = clutter[i]


"""pos_ids = pos_examples[:,0]
pos_labels = pos_examples[:,1]
pos_clutter = pos_examples[:,2]
neg_ids = neg_examples[:,0]
neg_labels = neg_examples[:,1]
neg_clutter = neg_examples[:,2]

def update_file(labels, clutter, ids, hdf5=False):
    
    for i,id in enumerate(ids):
        if hdf5:
            data_file = f"{dataset_dir}/test_data.hdf5"
            group_name = f"{int(id):07d}-rawdata"
            with h5py.File(data_file, "r+") as h5file:
                h5file[group_name].attrs["label"] = labels[i]
                h5file[group_name].attrs["clutter"] = clutter[i]
        else:
            data_file = f"{dataset_dir}/{int(id)}-rawdata.npz"
            file_data = dict(np.load(data_file, allow_pickle=True))
            file_data["label"] = labels[i]
            file_data["clutter"] = clutter[i]
            np.savez(data_file, **file_data)

update_file(pos_labels, pos_clutter, pos_ids, hdf5=hdf5)
update_file(neg_labels, neg_clutter, neg_ids, hdf5=hdf5)
"""




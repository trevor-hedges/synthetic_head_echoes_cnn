import numpy as np
import h5py
import os

def load_data_hdf5_simple(h5filepath, id):
    
    group_name = f"{id:07d}-rawdata"
    data = {}
    with h5py.File(h5filepath, "r") as h5file:
        
        data["data"] = h5file[group_name]["data"][:,:]
        #rawdata = data[0,:,:] + 1j*data[1,:,:]
        if "match" in data:
            data["match"] = h5file[group_name]["match"][:,:]
        data["label"] = h5file[group_name].attrs["label"]
        data["id"] = h5file[group_name].attrs["id"]

        if "clutter" in h5file[group_name].attrs:
            data["clutter"] = h5file[group_name].attrs["clutter"]
        else:
            data["clutter"] = -1
        
        if "avg_snr" in h5file[group_name].attrs:
            data["avg_snr"] = h5file[group_name].attrs["avg_snr"]
        else:
            data["avg_snr"] = None
    
    return(data)

def get_max_id_hdf5(h5filepath):

    with h5py.File(h5filepath, "r") as h5file:
        
        i = 0
        while (f"{i:07d}-rawdata" in h5file):
            i += 1
    
    return(i)

def save_data_hdf5_simple(h5filepath, id, data_dict, attrs_dict):

    id = int(id)

    group_name = f"{id:07d}-rawdata"

    if id == 0:
        file_access = "w-"  # Error if file already exists. Requires user to delete an existing file first.
    else:
        file_access = "r+"

    with h5py.File(h5filepath, file_access) as h5file:
        
        grp = h5file.create_group(group_name)
        
        for key in data_dict.keys():
            grp[key] = data_dict[key]
        
        for key in attrs_dict.keys():
            grp.attrs[key] = attrs_dict[key]


def convert_npz_hdf5(npz_dir, hdf5_filepath):

    # Get list of all npz files
    file_list = os.listdir(npz_dir)

    #files_npz = []
    first_write = True
    id = 0
    for file in file_list:
        if file[-3:] == "npz":

            #id = int(file.split("-")[0])

            data = np.load(f"{npz_dir}/{file}", allow_pickle=True)
            
            if first_write:
                file_access = "w-"
                first_write = False
            else:
                file_access = "r+"

            with h5py.File(hdf5_filepath, file_access) as h5file:

                group_name = f"{id:07d}-rawdata"
                grp = h5file.create_group(group_name)

                for field in list(data):

                    if field == "id" or field=="label" or field=="clutter" or field=="t0" \
                        or field=="start_pulse_in_file" or field=="start_sample_in_file" or field=="source_file":
                        
                        grp.attrs[field] = data[field]
                    else:
                        grp[field] = data[field]

            id += 1

            #files_npz.append(file)

    #print("List of npz files: ", files_npz)


def print_info_hdf5(hdf5_filepath):

    with h5py.File(hdf5_filepath, "r") as h5file:

        print(list(h5file))

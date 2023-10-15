# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:24:43 2023

@author: thedges
"""

import numpy as np
import h5py
import mfAlgs_simple
from CEDAR2019_params_simple import facilParams, cLight
import matplotlib.pyplot as plt
import os
import os.path
import datetime
from load_data import get_max_id_hdf5

def convert_JRO_timestamp(timestamp):
    """ Converts JRO timestamp string in the format YYYY-MM-DDTHH:MM:SS.XXX
    into a Python datetime object
    """
    datetime_object = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
    return(datetime_object)


def determine_chunk_startpts(L, chunk_size, min_overlap, start_coord=0):
    
    num_tiles = int(np.ceil((L-min_overlap)/(chunk_size-min_overlap)))
    T = chunk_size*num_tiles - L
    N = num_tiles - 1
    n = int(np.ceil(T/N))
    a = N*n-T
    b = N - a
    tile_rng_a = np.arange(a)
    tile_rng_b = np.arange(b + 1)
    tile_start_a = tile_rng_a*(chunk_size - (n-1))
    tile_start_b = a*(chunk_size-(n-1)) + tile_rng_b*(chunk_size-n)
    tile_start = start_coord + np.concatenate((tile_start_a, tile_start_b))
    
    return(tile_start)


def load_chunk_info(chunk_file):
    
    chunk = np.load(chunk_file, allow_pickle=True)
    
    print("Source file: ", chunk["source_file"])
    print("Chunk start time: ", chunk["t0"])
    print("Chunk ID: ", chunk["id"])
    print("Chunk start pulse in source file: ", chunk["start_pulse_in_file"])
    print("Chunk start sample: ", chunk["start_sample_in_file"])
    
    

def split_into_chunks(radar_file, out_dir, plot_dir, facility, start_sample, end_sample,
                      chunk_size_x=150, chunk_size_y=150, min_overlap=100, hdf5=False, plot=True):
    
    
    # Strip filepath from radar file
    radar_filename = radar_file.split("/")[-1]
    
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Function to load "formatted" radar data file and split it up into 150x150
    #   "chunks" readable by CS229 final project CNN
    
    # Specify output filename
    out_filename = f"{out_dir}/test_data.h5"

    # Get data example id number to start with
    # TODO: must check if file exists. If it doesn't, set to 0.
    if os.path.isfile(out_filename):
        start_id = get_max_id_hdf5(out_filename)
    else:
        start_id = 0

    # Load radar file
    with h5py.File(radar_file, "r") as rfile:
        
        rawdata = rfile["rawdata"][:,:]
        rng = rfile["rng"][:]
        time = rfile["time"][:]
        #if "dtime" in rfile:
        #    dtime = rfile["dtime"][:]
        t0 = rfile.attrs["t0"]
        
    # Convert the timestamp to a datetime object
    t0_datetime = convert_JRO_timestamp(t0)
        
    # Perform matched filter on rawdata (solely for plotting purposes)
    match = mfAlgs_simple.matchedFilter2D(rawdata, code=facilParams[facility]["CODE"], params=facilParams[facility], dopplerV=30000)
    power = mfAlgs_simple.powerLog(match, noisefloor=facilParams[facility]["NOISE"])
    fig, ax = mfAlgs_simple.plotRTI(power, figsize=(18,12), alt=rng/1000, time=time)
    fig.savefig(plot_dir + f"/rti{start_id}.png")
    plt.close(fig)
    
    # Determine start points for each tile.
    Lx = np.shape(rawdata)[0]
    Ly = end_sample-start_sample #np.shape(rawdata)[1]
    tile_start_x = determine_chunk_startpts(Lx, chunk_size_x, min_overlap)
    tile_start_y = determine_chunk_startpts(Ly, chunk_size_y, min_overlap,
                                            start_coord=start_sample)

    # Cut rawdata into corresponding chunks
    num_tiles_x = len(tile_start_x)
    num_tiles_y = len(tile_start_y)
    k = start_id
    for i in tile_start_x:
        
        # Get the starttime for this "column" of tiles
        starttime = t0_datetime + datetime.timedelta(seconds=time[i])
        time_rel = time[i:i+chunk_size_x] - time[i]
                
        for j in tile_start_y:
            
            
            rawdata_chunk = np.empty((2, chunk_size_x, chunk_size_y))
            rawdata_chunk[0,:,:] = np.real(rawdata[i:i+chunk_size_x, j:j+chunk_size_y])
            rawdata_chunk[1,:,:] = np.imag(rawdata[i:i+chunk_size_x, j:j+chunk_size_y])
            
            # NOTE: added start pulse/sample as metadata so they can be
            #   identified from source file
            if hdf5:
                with h5py.File(f"{out_dir}/test_data.h5", "a") as h5file:
                    
                    data_str = f"{k:07d}-rawdata"
                    if not data_str in h5file:
                        data_grp = h5file.create_group(data_str)
                    else:
                        data_grp = h5file[data_str]

                    data_grp["data"] = rawdata_chunk
                    data_grp.attrs["label"] = -1  # This indicates that the data is unlabelled
                    data_grp.attrs["id"] = k
                    data_grp["match"] = match[i:i+chunk_size_x, j:j+chunk_size_y]
                    data_grp["time"] = time_rel
                    data_grp.attrs["t0"] = str(starttime)
                    data_grp.attrs["start_pulse_in_file"] = i
                    data_grp.attrs["start_sample_in_file"] = j
                    data_grp.attrs["source_file"] = radar_filename
                    data_grp["rng"] = rng[j:j+chunk_size_y]
            else:
                np.savez(f"{out_dir}/{k:.0f}-rawdata.npz", data=rawdata_chunk,
                        match=match[i:i+chunk_size_x, j:j+chunk_size_y],
                        id=k, t0=starttime, start_pulse_in_file=i,
                        start_sample_in_file=j,
                        source_file=radar_filename, time=time_rel,
                        rng=rng[j:j+chunk_size_y])
            
            # Make a plot of just this chunk
            if plot:
                power_chunk_raw = mfAlgs_simple.powerLog(rawdata[i:i+chunk_size_x, j:j+chunk_size_y], noisefloor=facilParams[facility]["NOISE_RAW"])
                power_chunk_match = power[i:i+chunk_size_x, j:j+chunk_size_y]
                
                # Do FFT method on just this chunk
                peakVal, medianVal, dopplerVs = mfAlgs_simple.fftMethod(rawdata[i:i+chunk_size_x, j:j+chunk_size_y], facilParams[facility]["CODE"],
                                                        facilParams[facility]["f"],
                                                        facilParams[facility]["fRX"])
                
                peakVal[dopplerVs <= 10000] = 0.000001
                peakVal[dopplerVs >= 90000] = 0.000001

                #peakValMask = peakVal[np.all(np.stack((dopplerVs >= 10000, dopplerVs <= 100000)), axis=0)]
                peakValPower = mfAlgs_simple.powerLog(peakVal, noisefloor=facilParams[facility]["fftMax_dB_plot"]-15)

                # Plot side-by-side rawdata (left), matched filter (center), and FFT method (right) TODO: actually implement this
                """
                fig, (ax1, ax2, ax3) = mfAlgs.plot2RTI1FFT(power_chunk_raw, power_chunk_match, peakVal, title=f"chunk {k}",
                                        figsize=(15,4), time=time[np.arange(i,i+chunk_size_x)],
                                        alt=rng[np.arange(j,j+chunk_size_y)]/1000,
                                        FFTmax=facilParams[facility]["fftmax"], dBmin=0)
                """
                fig, (ax1, ax2, ax3) = mfAlgs_simple.plot3RTI(power_chunk_raw, power_chunk_match, peakValPower, title=f"chunk {k}",
                                        figsize=(15,4), time=time[np.arange(i,i+chunk_size_x)],
                                        alt=rng[np.arange(j,j+chunk_size_y)]/1000,
                                        dBmin=0, dBmax=20)
                fig.savefig(plot_dir + f"/rti-chunk-{k}.png")
                plt.close(fig)
            
            if (k % 200) == 0:
                print(f"Finished chunk {k+1-start_id} of {num_tiles_x*num_tiles_y}")

            k += 1
    
    
def generate_all_chunks(formatted_data_path, start_num, end_num, out_dir, plot_dir, 
                        start_sample, end_sample, facility, start_part=0, end_part=9, chunk_size_x=150,
                        chunk_size_y=150,
                        min_overlap=100, hdf5=False, plot=True):
    """Generates all the chunk files of data from radar experiment with
    formatted data, such as JRO. """
    
    # Determine what the number of the first example to save should be
    
    # Get a list of all HDF5 files in a directory, sorted alphabetically
    files = os.listdir(formatted_data_path)
    h5_files = []
    h5_files.sort() # Sort alphabetically
    for file in files:
        if file[-3:] == ".h5":
            filestr = file.split(".")[0] # Remove file extension
            filestr = filestr.split("_")[0] # Remove any "part" string
            filestr = filestr.split("data")[1] # Remove "data" prefix
            if int(filestr) >= start_num and int(filestr) <= end_num:

                # Check whether the series of files contains "parts", as is the case for JRO.
                if "part" in file:
                    part_num = int(file.split(".")[0][-1])
                else:
                    part_num = -1

                if (part_num == -1) or (part_num >= start_part) and (part_num <= end_part):
                    split_into_chunks(f"{formatted_data_path}/{file}", out_dir, plot_dir, facility,
                                        start_sample, end_sample, chunk_size_x, chunk_size_y,
                                        min_overlap,
                                        hdf5=hdf5, plot=plot)

                #h5_files.append(file)
    

    """    

    print("splitting the following files: ", h5_files)

    num_files = len(h5_files)    
    k = start_id
    for j,file in enumerate(h5_files):
        print(f"Working on file {j+1} of {num_files}")
        k = split_into_chunks(formatted_data_path + "/" + file, out_dir, plot_dir,
                              start_sample, end_sample, chunk_size, min_overlap, k,
                              hdf5=hdf5, plot=plot)
    """
    
    
    
    
"""
start_sample_JRO = 300
end_sample_JRO = 950
start_sample_RISRN = 960
end_sample_RISRN = 1894
"""

if __name__ == "__main__":
    
    plt.close("all")
    plt.ioff()
    
    #radar_file = r"E:/JROdata/formatted/20191010/data283235_part0.h5"
    
    #dirname = "day1_hour1_partial"
    dirname = "day1_hour3_partial_EEJ_1"
    #facility = "MHO"
    facility = "JRO"

    out_dir = f"C:/Users/thedges/Dropbox/research/radar/computer_vision/{facility}/{dirname}/data/"
    plot_dir = f"C:/Users/thedges/Dropbox/research/radar/computer_vision/{facility}/{dirname}/plot/"
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Analyze all chunks from all formatted .h5 radar data files in a directory
    #formatted_datapath = r"C:/Users/thedges/Dropbox/research/radar/computer_vision/formatted_data/"
    formatted_datapath = r"E:/JROdata/formatted/20191010/"
    #formatted_datapath = r"E:/RISRdata/formatted/20191010.001/"
    #formatted_datapath = r"E:/MHOdata/formatted/20191010/"

    #for meteor_filenum in [1570701600, 1570701802]:
    #for meteor_filenum in [164159, 164199, 164219, 164243, 164251, 164271, 164279, 164283, 164299, 164303, 164363, 164367, 164483, 164503, 164507, 164555, 164563, 164595, 164611, 164627, 164643]:
    #generate_all_chunks(formatted_datapath, 1570701600, 1570701802, out_dir, plot_dir, 
    generate_all_chunks(formatted_datapath, 283274, 283274, out_dir, plot_dir, 
                        start_sample=facilParams[facility]["start_sample"],
                        end_sample=facilParams[facility]["end_sample"],
                        facility=facility,
                        chunk_size_x=150, chunk_size_y=150, min_overlap=30, hdf5=True, plot=True)
        
    # day1_hour0_partial --> 283172
    # day1_hour0_partial_2 --> 283173
    # day1_hour0_partial_3 --> 283174
    # day1_hour1_partial --> 283202
    # day1_hour2_partial --> 283229
    # day1_hour3_partial --> 283259 
    #   all at start of hour
    # EEJ --> need a bunch of examples near the end
    # day1_hour3_partial_EEJ_1 --> 283274
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
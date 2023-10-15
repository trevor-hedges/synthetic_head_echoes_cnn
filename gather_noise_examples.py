import numpy as np
import os
from gather_clutter_examples import linear_comb_n, linear_comb_n_wrapper
from load_data import load_data_hdf5_simple, get_max_id_hdf5, save_data_hdf5_simple
from CEDAR2019_params_simple import facilParams

# Script that grabs examples of non-meteors from test set and turns them into a set of "noise" examples,
#   with data augmentation to turn the small amount of designated examples into a large set
#   Examples of clutter/noise are already pre-designated within the test sets.

# Uncomment whichever facility to use
facility_str = "RISR-N"
#facility_str = "MHO"
#facility_str = "JRO"

files = []
dirsRISR_N = ["day1_hour0_partial", "day1_hour1_partial"]
dirsMHO = ["day1_hour0_partial", "day1_hour1_partial"]
dirsJRO = ["day1_hour0_partial"]

if facility_str == "RISR-N":
    dirs = dirsRISR_N
elif facility_str == "MHO":
    dirs = dirsMHO
else:
    dirs = dirsJRO


exclude_clutter = False
include_all_clutter = True
for dirname in dirs:
    files.append(f"{facility_str}/{dirname}/data/test_data.h5")

num_noise_include = 2000

# Designate new HDF5 file for noise examples
noise_dir = f"{facility_str}/noise_examples/"
noise_h5file = f"{noise_dir}/noise_examples.h5"
facility_p = facilParams[facility_str]

# Create directory to store noise examples
os.makedirs(noise_dir, exist_ok=True)

# Form array that specifies which file a particular example is in and what id it is in the file.
#   Only include examples of noise, i.e. not identified as "clutter" and not a meteor either.
examples_files = []
examples_files_include = []
for j,filepath in enumerate(files):
    
    num_examples = get_max_id_hdf5(filepath)

    i = 0
    for id in range(num_examples):
        data = load_data_hdf5_simple(filepath, id)
        
        label = data["label"]
        cl_type = data["clutter"]
        if exclude_clutter:
            if (cl_type == 0) and (label == 0):
                examples_files.append([j, id])
        else:
            if label == 0:
                examples_files.append([j, id])
        
        if include_all_clutter:
            if (cl_type != 0) and (label == 0):
                examples_files_include.append([j, id])


# 
examples_files = np.array(examples_files)
num_examples_tot = np.shape(examples_files)[0]

examples_files_include = np.array(examples_files_include)
num_examples_must_include = np.shape(examples_files_include)[0]

# Initialize random number generator
rng = np.random.default_rng(seed=42)

# Determine examples to include
noise_avail_rng = np.arange(num_examples_tot)
rng.shuffle(noise_avail_rng)
noise_ids = noise_avail_rng[:num_noise_include]


# Add all instances that we mark specifically for inclusion
must_include_ids = np.arange(num_examples_must_include)

# Save all instances of noise as they are (i.e. no combinations)
start_id = 0
start_id = linear_comb_n_wrapper(noise_ids, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=False)

# Save some instances with random roll enabled (data augmentation)
start_id = linear_comb_n_wrapper(noise_ids, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=True)
start_id = linear_comb_n_wrapper(noise_ids, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=True)
start_id = linear_comb_n_wrapper(noise_ids, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=True)

# Save some linear combinations of instances with random roll

nnoise_2_comb = num_noise_include*5 #num_noise_include*10

# Determine what linear combinations of 2 separate noise examples to make
noise_combs = noise_ids[rng.integers(num_noise_include, size=(nnoise_2_comb, 2))]
start_id = linear_comb_n_wrapper(noise_combs, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=True)

# Include examples that are marked specifically for inclusion
start_id = linear_comb_n_wrapper(must_include_ids, examples_files_include, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=False)
start_id = linear_comb_n_wrapper(must_include_ids, examples_files_include, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=True)







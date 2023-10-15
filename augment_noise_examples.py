import numpy as np
import os
from load_data import load_data_hdf5_simple, get_max_id_hdf5, save_data_hdf5_simple
from CEDAR2019_params_simple import facilParams

def linear_comb_n(files, ids, rng, gmin=0.95, gmax=1.05, random_roll = False):
    """Produces linear combination of N data examples specified in the list ids, using random mixture between each example
    and also including a random gain factor.
    """

    N = len(ids)

    # Get random coefficients that add up to 1
    c = rng.dirichlet(np.ones(N), size=1)[0]

    # Get a random gain factor by which to multiply all data.
    #   If N=1, just let this be 1, since in this case we only want to save the example itself.
    if N > 1:
        g = rng.uniform(gmin, gmax)
    else:
        g = 1.0

    # Get first example
    data_comb = load_data_hdf5_simple(files[0], ids[0])["data"]

    # Roll this example if desired (a form of augmentation to maximize utility of few noise examples)
    if random_roll:
        shape_x = np.shape(data_comb)[1]
        shape_y = np.shape(data_comb)[2]
        data_comb = np.roll(data_comb, shift=(0, rng.integers(0,shape_x), rng.integers(0,shape_y)))

    # Multiply by first mix coefficient
    data_comb *= c[0]

    # Add next N-1 examples
    for i in np.arange(1,N):
        data_add = load_data_hdf5_simple(files[i], ids[i])["data"]
        if random_roll:
            data_add = np.roll(data_add, shift=(0, rng.integers(0,shape_x), rng.integers(0,shape_y)))
        data_comb += c[i]*data_add

    # Multiply by total gain
    data_comb *= g

    return(data_comb)


def linear_comb_n_wrapper(comb_list, examples_files, h5filepaths, h5filepath_save, start_id, rng, gmin=0.95, gmax=1.05, random_roll=False):

    num_combs = np.shape(comb_list)[0]
    if comb_list.ndim == 1:
        comb_list = np.reshape(comb_list,(num_combs,1))

    for i,id in enumerate(np.arange(start_id, start_id + num_combs)):

        # Get appropriate files and ID in the file
        h5files = list(h5filepaths[l] for l in examples_files[comb_list[i,:]][:,0])
        ids_in_file = examples_files[comb_list[i,:]][:,1]

        data_comb = linear_comb_n(h5files, ids_in_file, rng, gmin, gmax, random_roll)
        label = 0
        clutter = 2

        data_dict = {"data": data_comb}#, "match": match}
        attrs_dict = {"label": label, "clutter": clutter, "id": id}

        save_data_hdf5_simple(h5filepath_save, id, data_dict, attrs_dict)
    
    return(start_id + num_combs)



if __name__ == "__main__":

    # Script that grabs examples of non-meteors from test set and turns them into a set of "noise" examples,
    #   with data augmentation to turn the small amount of designated examples into a large set
    #   Examples of clutter/noise are already pre-designated within the test sets.

    # Uncomment whichever facility to use
    #facility_str = "RISR-N"
    facility_str = "MHO"
    #facility_str = "JRO"
    files = []

    # Add noise file to list (normally this script can account for multiple files but here it's hardcoded for just one)
    noise_dir = f"{facility_str}/noise_examples/"
    files.append(f"{noise_dir}/noise_examples_small.h5")

    # Generate new HDF5 file where all augmented noise examples will be generated
    noise_h5file = f"{noise_dir}/noise_examples.h5"

    # Get parameters for the chosen facility
    facility_p = facilParams[facility_str]


    # Form array that specifies which file a particular example is in and what id it is in the file.
    # Needed when gathering examples from multiple HDF5 files (not used here but useful)
    examples_files = []
    examples_files_include = []
    for j,filepath in enumerate(files):
        
        num_examples = get_max_id_hdf5(filepath)

        i = 0
        for id in range(num_examples):
            data = load_data_hdf5_simple(filepath, id)
            examples_files.append([j, id])


    # Turn this into Numpy array
    examples_files = np.array(examples_files)
    num_examples_tot = np.shape(examples_files)[0]

    # Initialize random number generator
    rng = np.random.default_rng(seed=42)

    # Determine examples to include 
    noise_avail_rng = np.arange(num_examples_tot)
    rng.shuffle(noise_avail_rng)
    noise_ids = noise_avail_rng[:num_examples_tot]

    # Save all instances of noise as they are (i.e. no combinations)
    start_id = 0
    start_id = linear_comb_n_wrapper(noise_ids, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=False)

    # Save some instances with random roll enabled (data augmentation)
    start_id = linear_comb_n_wrapper(noise_ids, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=True)
    start_id = linear_comb_n_wrapper(noise_ids, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=True)
    start_id = linear_comb_n_wrapper(noise_ids, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=True)

    # Save some linear combinations of instances with random roll
    nnoise_2_comb = num_examples*6 #num_noise_include*10

    # Determine what linear combinations of 2 separate noise examples to make
    noise_combs = noise_ids[rng.integers(num_examples, size=(nnoise_2_comb, 2))]
    start_id = linear_comb_n_wrapper(noise_combs, examples_files, files, noise_h5file, start_id, rng, gmin=1, gmax=1, random_roll=True)




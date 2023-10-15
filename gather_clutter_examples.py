import numpy as np
import os
from load_data import load_data_hdf5_simple, get_max_id_hdf5, save_data_hdf5_simple
import mfAlgs
from CEDAR2019_params import facilParams, cLight

def linear_comb_n(files, ids, rng, gmin=0.95, gmax=1.05, random_roll = False):
    """Produces linear combination of N data examples specified in the list ids, using random mixture between each example
    and also including a random gain factor.
    """

    N = len(ids)

    # Get random coefficients that add up to 1
    c = rng.dirichlet(np.ones(N), size=1)[0]
    #c = rng.uniform(0, 1, size=N)
    #c[N-1] = 1-np.sum(c[:N])

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
        #match = mfAlgs.matchedFilter2D(data_comb[0,:,:] + 1j*data_comb[1,:,:], facility["CODE"])
        label = 0
        clutter = 2

        data_dict = {"data": data_comb}#, "match": match}
        attrs_dict = {"label": label, "clutter": clutter, "id": id}

        save_data_hdf5_simple(h5filepath_save, id, data_dict, attrs_dict)
    
    return(start_id + num_combs)





# Function to choose some labelled data examples designated as clutter,
#   and add them to a separate data set to include in the synthetic data.
if __name__ == "__main__":

    facil_str = "MHO"

    labelled_datasets = []

    if facil_str == "JRO":
        labelled_datasets.append(r"C:/Users/thedges/Dropbox/research/radar/computer_vision/rawdata_cleaned_partial_10_10_JRO/data/test_data.h5")
        labelled_datasets.append(r"C:/Users/thedges/Dropbox/research/radar/computer_vision/day1_hour0_partial/data/test_data.h5")
        labelled_datasets.append(r"C:/Users/thedges/Dropbox/research/radar/computer_vision/day1_hour1_partial/data/test_data.h5")
        labelled_datasets.append(r"C:/Users/thedges/Dropbox/research/radar/computer_vision/day1_hour2_partial/data/test_data.h5")
    elif facil_str == "MHO":
        labelled_datasets.append(f"C:/Users/thedges/Dropbox/research/radar/computer_vision/MHO/day1_hour1_partial/data/test_data.h5")


    clutter_dir = f"C:/Users/thedges/Dropbox/research/radar/computer_vision/{facil_str}/clutter_examples/"
    clutter_h5file = f"{clutter_dir}/clutter_examples.h5"
    facility = facilParams[facil_str]

    os.makedirs(clutter_dir, exist_ok=True)

    # Specify number of trails and number of fluff from each dataset to put together into clutter examples
    num_trails_to_use = 1000000
    num_fluffs_to_use = 1000000
    num_eejs_to_use = 1000000
    num_glitches_to_use = 1000000
    num_nonmeteors_to_use = 1000000
    fluff_2_mult = 10
    fluff_3_mult = 10
    fluff_2_3_mult = 10
    glitches_2_mult = 20
    nonmeteors_2_mult = 20

    # Load labels, get random sampling of different kinds of each clutter without a meteor, and save in separate file
    start_id = 0


    # Form array that specifies which file a particular example is in, what id it is in the file,
    #   and what type of clutter it is
    examples_files = []

    for j,dataset_dir in enumerate(labelled_datasets):
        
        num_examples = get_max_id_hdf5(dataset_dir)

        i = 0
        for id in range(num_examples):
            data = load_data_hdf5_simple(dataset_dir, id)
            
            label = data["label"]
            cl_type = data["clutter"]
            if (cl_type > 0) and (label == 0):
                examples_files.append([j,id,cl_type,label])


    # Convert to Numpy arrays
    examples_files = np.array(examples_files)

    # Get indices of each type of clutter

    #trails = np.where(np.all(np.stack((examples_files[:,2] == 1, examples_files[:,3] == 0)), axis=1))[0]
    trails = np.where(examples_files[:,2] == 1)[0]
    fluffs = np.where(examples_files[:,2] == 2)[0]
    eejs = np.where(examples_files[:,2] == 3)[0]
    glitches = np.where(examples_files[:,2] == 4)[0]
    nonmeteors = np.where(examples_files[:,2] == 5)[0]

    num_trails = len(trails)
    num_fluffs = len(fluffs)
    num_eejs = len(eejs)
    num_glitches = len(glitches)
    num_nonmeteors = len(nonmeteors)
    print("Number of trails available: ", num_trails)
    print("Number of fluff available: ", num_fluffs)
    print("Number of eej available: ", num_eejs)
    print("Number of glitches available: ", num_glitches)
    print("Number of nonmeteor solid targets available: ", num_nonmeteors)

    if num_trails_to_use > num_trails:
        num_trails_to_use = num_trails
    if num_fluffs_to_use > num_fluffs:
        num_fluffs_to_use = num_fluffs
    if num_eejs_to_use > num_eejs:
        num_eejs_to_use = num_eejs
    if num_glitches_to_use > num_glitches:
        num_glitches_to_use = num_glitches
    if num_nonmeteors_to_use > num_nonmeteors:
        num_nonmeteors_to_use = num_nonmeteors

    # Initialize random number generator
    rng = np.random.default_rng(seed=42)

    trails_avail_rng = np.arange(num_trails_to_use)
    fluffs_avail_rng = np.arange(num_fluffs_to_use)
    eejs_avail_rng = np.arange(num_eejs_to_use)
    glitches_avail_rng = np.arange(num_glitches_to_use)
    nonmeteors_avail_rng = np.arange(num_nonmeteors_to_use)

    rng.shuffle(trails_avail_rng)
    rng.shuffle(fluffs_avail_rng)
    rng.shuffle(eejs_avail_rng)
    rng.shuffle(glitches_avail_rng)
    rng.shuffle(nonmeteors_avail_rng)

    
    # Save all instances of noise as they are (i.e. no combinations)
    if num_trails_to_use > 0:
        trail_ids = trails[trails_avail_rng[:num_trails_to_use]]
        start_id = linear_comb_n_wrapper(trail_ids, examples_files, labelled_datasets, clutter_h5file, start_id, rng)
    if num_fluffs_to_use > 0:
        fluff_ids = fluffs[fluffs_avail_rng[:num_fluffs_to_use]]
        start_id = linear_comb_n_wrapper(fluff_ids, examples_files, labelled_datasets, clutter_h5file, start_id, rng)
    if num_eejs_to_use > 0:
        eej_ids = eejs[eejs_avail_rng[:num_eejs_to_use]]
        start_id = linear_comb_n_wrapper(eej_ids, examples_files, labelled_datasets, clutter_h5file, start_id, rng)
    if num_glitches_to_use > 0:
        glitch_ids = glitches[glitches_avail_rng[:num_glitches_to_use]]
        start_id = linear_comb_n_wrapper(glitch_ids, examples_files, labelled_datasets, clutter_h5file, start_id, rng, random_roll=True)
    if num_nonmeteors_to_use > 0:
        nonmeteor_ids  = nonmeteors[nonmeteors_avail_rng[:num_nonmeteors_to_use]]
        start_id = linear_comb_n_wrapper(nonmeteor_ids, examples_files, labelled_datasets, clutter_h5file, start_id, rng, random_roll=True)

    # Create some number of linear combinations of the noise
    if num_fluffs_to_use > 0 and fluff_2_mult > 0:
        nfluff_2_comb = num_fluffs_to_use*fluff_2_mult
        # Determine what linear combinations of 2 separate noise examples to make
        fluff_combs = fluff_ids[rng.integers(num_fluffs, size=(nfluff_2_comb, 2))]
        start_id = linear_comb_n_wrapper(fluff_combs, examples_files, labelled_datasets, clutter_h5file, start_id=start_id, rng=rng)

    if num_fluffs_to_use > 0 and fluff_3_mult > 0:
        nfluff_3_comb = num_fluffs_to_use*fluff_3_mult
        # Determine what linear combinations of 3 separate noise examples to make
        fluff_combs = fluff_ids[rng.integers(num_fluffs, size=(nfluff_3_comb, 3))]
        start_id = linear_comb_n_wrapper(fluff_combs, examples_files, labelled_datasets, clutter_h5file, start_id, rng=rng)

    if num_fluffs_to_use > 0 and num_trails_to_use > 0 and fluff_2_3_mult > 0:
        ntrail_add_to_fluff = (fluff_2_3_mult*(num_trails_to_use + num_fluffs_to_use))//2
        # Determine what linear combinations of 1 trail + 1 fluff to make
        trail_fluff_combs = np.stack((trail_ids[rng.integers(num_trails, size=ntrail_add_to_fluff)],
                                            fluff_ids[rng.integers(num_fluffs, size=ntrail_add_to_fluff)])).T
        start_id = linear_comb_n_wrapper(trail_fluff_combs, examples_files, labelled_datasets, clutter_h5file, start_id, rng=rng)


    if num_glitches_to_use > 0 and glitches_2_mult > 0:
        nglitches_2_comb = num_glitches_to_use*glitches_2_mult
        glitch_combs = glitch_ids[rng.integers(num_glitches, size=(nglitches_2_comb, 2))]
        start_id = linear_comb_n_wrapper(glitch_combs, examples_files, labelled_datasets, clutter_h5file, start_id=start_id, rng=rng, random_roll=True)

    if num_nonmeteors_to_use > 0 and nonmeteors_2_mult > 0:
        nnonmeteors_2_comb = num_nonmeteors_to_use*nonmeteors_2_mult
        nonmeteor_combs = nonmeteor_ids[rng.integers(num_nonmeteors, size=(nnonmeteors_2_comb, 2))]
        start_id = linear_comb_n_wrapper(nonmeteor_combs, examples_files, labelled_datasets, clutter_h5file, start_id=start_id, rng=rng, random_roll=True)







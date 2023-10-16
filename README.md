# synthetic_head_echoes_cnn
Repository for convolutional neural network used to classify rawdata from high-power radar facilities based on whether rawdata contains a meteor head echo or not. Contains routines to generate synthetic data to train the CNN.

The root directory contains Python scripts needed to train and test the convolutional neural network (CNN) used to detect head echoes. The results from these scripts are presented in the paper "Meteor Head Echo Detection at Multiple High-Power Radar Facilities via a Convolutional Neural Network Trained on Synthetic Radar Data", authored by Trevor Hedges, Nicolas Lee, and Sigrid Elschot.


These Python scripts rely on the following dependencies:

* CUDA-capable GPU
* numpy
* pandas
* matplotlib/Pyplot
* h5py
* PyTorch with CUDA/GPU capability enabled
* Pre-labelled radar data test sets and examples of noise, contained in an HDF5 file, for radar facilities of interest. This data is archived on Zenodo (doi: 10.5281/zenodo.10005351) as a tar.gz archive, and its contents must be extracted into the root directory with the Python scripts.

The following steps will run the CNN method to generate results similar to those presented in the paper. Note that each script has its own settings/parameters at the end of the script (after all classes or functions are defined) that must be modified approprately.

1. Within augment_noise_examples.py, change the facility_str variable to the facility of interest ("RISR-N", "MHO", or "JRO"), and run the script. This uses the ~2000 clutter/noise examples provided in a facility's noise_examples_small.h5 file to generate ~20000 (or more) examples of clutter/noise, by taking linear combinations of examples and translating them arbitrarily as described in the paper. This step is necessary to run the generateSyntheticRadarDataDoubles.py file which generates training examples.

2. Within generateSyntheticRadarDataDoubles.py, change the facility variable to the facility of interest and run the script to generate training examples. This script generates synthetic head echoes using randomly generated parameters, and combines them randomly with the many clutter/noise examples contained in the facility's noise_examples.h5 file.

3. Within cnn_simple.py, change the facility variable to the facility of interest. Adjust num_epochs_to_run as desired. Ensure learning_rate is set to 0.001 if it's the first training run. Ensure test_only is set to False unless you are only testing an existing CNN (not training and then testing). Then, run the script. The status of the CNN, including loss, validation F1-score, and F1-score on the real test set will be reported to the terminal as it trains. When the script finishes running, all performance metrics will be reported for each test set, including sensitivity above/below the 15 dB threshold. If you run the script again, the CNN will resume training where it left off for the number of desired epochs, unless you delete the trained model saved in the {facility_name}/synthetic_data/trained_models directory. To generate results for the paper, the CNN was run multiple times for a few epochs, with the learning rate gradually decreased to 0.00001. The performance metrics will level off after training at this learning rate for some epochs.

4. If desired, use the createCAM.py script to generate class activation maps (CAMs) like those included in the paper. Change facility_name to the desired facility, and test_set_str to the desired test set. Then, run the script. It may take some time to generate all the CAMs throughout all training steps of the model. If desired, you can set include_all_interesting to False, and then specify which specific test example IDs within the test set to generate CAMs for in ids_to_include.

Note: In the paper results, the test sets for RISR-N and MHO include all of the raw data segments contained in each facility's day1_hour0_partial.h5 AND day1_hour1_partial.h5 (note that both RISR-N and MHO have both of these files corresponding to a partial window within the first hour of the radar experiment). The test set for JRO (non-EEJ) contains raw data segments ONLY from day1_hour0_partial.h5. The test set for JRO (EEJ) contains raw data segments ONLY from day1_hour3_partial_EEJ_1.h5 (which corresponds to a partial window within the final hour of the radar experiment, when the EEJ is present).



Descriptions of each included Python script are as follows:


ablation_sim.py:

Contains helper function (the exponential trajectory function) used to generate synthetic head echoes.


augment_noise_examples.py:

Script that uses the small amount of clutter/noise examples provided in a facility's noise_examples_small.h5 file to generate a large amount of examples of clutter/noise, by taking linear combinations of examples and translating them arbitrarily as described in the paper. 


CEDAR2019_params_simple.py:

Contains experiment parameters such as pulse code and frequencies for the experiments at all three facilities from the CEDAR 2019 concurrent data collect.


cnn_simple.py:

Script that trains the CNN on synthetic head echoes and tests the CNN on real head echoes.


createCAM.py:

Script that generates class activation maps that demonstrate what parts of the image the CNN considers important in making its decision for test examples.


gather_clutter_examples.py:

File that contains helper functions for gather_noise_examples.py.


gather_noise_examples.py:

Python script that grabs pre-labelled examples of noise from a test set (labelled via a non-zero integer in the attribute "clutter" on an example) and puts them in a separate HDF5. This script is not needed to simply recreate results.


generateSyntheticRadarDataDoubles.py:

Script that generates synthetic head echoes, combines them with examples of clutter/noise from real data, and puts them into an HDF5 file as a training set. The "doubles" refers to the fact that adjacent training examples are with/without a synthetic head echo but have the same background noise/clutter.


get_false_neg_rate_vs_snr.py:

Contains helper function to generate sensitivity metrics above and below an SNR threshold.


label_data.py:

Adds "label" as an attribute to every example in an HDF5 file, and labels each example from data in a CSV. This script is not needed to simply recreate results.


load_data.py:

Helper functions to load HDF5 training and test sets for the CNN.


mfAlgs_simple.py:

Helper functions to perform matched filtering on raw data and generate plots such as RTIs.


postpro_cnn_results.py:

Helper functions to postprocess CNN results and performance metrics.


split_data.py:

Script that takes HDF5 containing large segment of rawdata and splits it into smaller segments interpreted by the CNN. This script is not needed to simply recreate results.




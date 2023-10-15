import numpy as np
from CEDAR2019_params_simple import facilParams, cLight
import mfAlgs_simple
import matplotlib.pyplot as plt
import os.path
import h5py
from load_data import load_data_hdf5_simple, get_max_id_hdf5

from ablation_simulation import ablation_sim

from scipy.stats import maxwell

# Global parameters
r0min = 75  # km
r0max = 125  # km


def generate_rxnoise_polar(dim, noise, rng=np.random.default_rng()):
    
    # Get mean value of absolute value from specified noisefloor
    mu = 10**(noise/20)
    
    # Find scale factor, a, for Maxwellian distribution
    scale = np.sqrt(np.pi/2)*mu/2
    
    # Calculate random values for theta (on the complex plane) uniformly
    theta = 2*np.pi*rng.random(dim)
    
    # Calculate random values for magnitude using maxwellian and include phase
    rawdata_synth = maxwell.ppf(rng.random(dim), scale=scale)*np.exp(1j*theta)
    
    return(rawdata_synth)


def generate_meteor(facility, r0, v0, a0, phi0, max_snr, time_observed,
                      rng=np.random.default_rng(), verbose=False):
    """Function to generate synthetic head echo.

    Generates only the meteor head echo in 2D array of its bounding box. No noise or surrounding
    data, and zeros in the corners where head echo signal not received. Output data size varies
    based on # pulses that the head echo is observed (x-dimension) and the # range gates it spans
    (y-dimension). Other routines handle placing the head echo into a rawdata segment with
    background noise etc.
    """

    # Unpack facility parameters from specified dictionary.
    ipp = facility["ipp"]  # Pulse resolution
    pulse_code = facility["CODE"]  # Complex pulse code in 1D Numpy vector
    code_len = len(pulse_code)  # code length
    deltar = facility["drng"]  # Range resolution
    f0 = facility["f"]  # Facility carrier frequency
    frx = facility["fRX"]  # Received signal sample frequency


    # Determine number of pulses the meteor is observed
    pulses_observed = int(np.ceil(time_observed / ipp))
    timerange_observed = np.linspace(0, ipp*(pulses_observed-1), pulses_observed)
    pulse_obs_start = int(np.abs(np.round(rng.normal(0,15))))
    if pulse_obs_start < 0:
        pulse_obs_start = 0

    rx_code = pulse_code
    trange = np.arange(0, pulses_observed*ipp, ipp)

    # Get full meteor trajectory
    rt, vt, at, phit = ablation_sim.simulate_trajectory_1D_exp(trange, r0,
                                                    v0, a0, phi0, f0)
    
    
    # Get bounds needed to allocate array for head echo itself.    
    # We assume the head echo is monotonic in range. Can be going "up the beam"
    #   if beam is at an angle, but velocity vector cannot be going up in altitude
    ll_p = int(np.floor(rt[0]/deltar))
    ul_p = int(np.ceil(rt[0]/deltar)) + code_len
    lr_p = int(np.floor(rt[-1]/deltar))
    ur_p = int(np.ceil(rt[-1]/deltar)) + code_len

    # Convert parallelogram to a rectangle
    if ll_p < lr_p:
        range_low = ll_p
    else:
        range_low = lr_p
    if ul_p < ur_p:
        range_high = ur_p
    else:
        range_high = ul_p
    
    # Allocate 2D array for the head echo
    num_samples_meteor = range_high-range_low+1
    dim = (pulses_observed, num_samples_meteor)
    rnggates = np.linspace(range_low, range_high, num_samples_meteor)
    rx = np.zeros(dim, dtype="complex128")

    # Get range of values at which to evaluate sinc function for SNR curve of head echo
    trange_shift_sinc = np.linspace(-1,1,pulses_observed)

    """
    # Generate SNR curve of the meteor.

    # Specify whether to use a sinc or Fourier sine series with N coeffs...
    # TODO: find a better way to implement this. Did not demonstrate improvement when tested,
    #   so not using this for 2023 paper submission.

    trange_shift_fourier_sine = np.linspace(0,1,pulses_observed)
    
    lightcurve_sinc = True

    lightcurve_rng = rng.uniform(0,1)
    if lightcurve_rng <= 0.25:
        lightcurve_sinc = True
    else:
        lightcurve_sinc = False
        if lightcurve_rng <= 0.5:
            n_fcoeffs = 2
        elif lightcurve_rng <= 0.75:
            n_fcoeffs = 3
        elif lightcurve_rng <= 0.9:
            n_fcoeffs = 4
        elif lightcurve_rng <= 0.96:
            n_fcoeffs = 5
        else:
            n_fcoeffs = 6
    

        # Get random Fourier coefficients
        c = (2*rng.dirichlet(np.ones(n_fcoeffs), size=1)[0]-1)
        c /= np.sqrt(np.abs(c))

        # Get power function with max value of 1
        f = np.sin(np.arange(n_fcoeffs)*np.pi*trange_shift_fourier_sine[:, np.newaxis]) @ c
        f /= np.sqrt(np.amax(f**2))
        f **= 2
    """

    # For each pulse in the simulated head echo observation, generate the receieved head echo data.
    for j, tp in enumerate(timerange_observed):

        if verbose:
            print("pulse ", j)

        # First, determine approximate time at which meteor first encounters radar pulse (after pulse emitted).
        tj = trange[j] + (rt[j]+vt[j]*rt[j]/cLight)/cLight

        # Get time across range of samples at which meteor observed
        tj_range = np.linspace(tj, tj+(code_len-1)/frx, code_len)

        # Determine parameters of meteor at this time based on initial parameters
        rtj, vtj, atj, phitj = ablation_sim.simulate_trajectory_1D_exp(tj_range, r0, v0, a0, phi0, f0)
        rt0 = rtj[0]
        
        # Get magnitude of received signal at this pulse
        amplitude = np.sinc(trange_shift_sinc[j])
        """
        # TODO: to implement properly along with above commented-out lines
        if lightcurve_sinc:
            amplitude = np.sinc(trange_shift_sinc[j])
        else:
            amplitude = f[j]
        """

        # Generate reflected/received pulse code
        rx_pulse = 10**((1/20)*(max_snr + facility["NOISE_RAW"])) * rx_code * \
            np.exp(1j*phitj)*amplitude
        
        # Zero-pad rx_pulse to ensure linear interpolation extrapolates to zero
        #   beyond the meteor's bounds
        rx_pulse = np.pad(rx_pulse, (1,1))
        
        rnggate0 = rt0/deltar
        rnggatesj = np.linspace(rnggate0-1, rnggate0 + code_len + 1, code_len+2)

        # Interpolate received pulse to range gates
        rx[j, :] += np.interp(rnggates, rnggatesj, rx_pulse, left=0, right=0)

    return(rx, rnggates)


def place_meteor_in_data(meteor_data, facility, size_x, size_y, edge_frac=0.75, rng=np.random.default_rng()):
    """Function to position a synthetic head echo within a raw data segment of specified size.
    Creates data array of predefined size and places meteor head echo in the array, possibly with
    only part of the head echo present in the array, since it may be hanging off the edge. The
    edge_frac argument specifies how much of each head echo side length MUST be present in the image;
    0 means the entire head echo can be off the edge whereas 1 means the entire head echo must be
    present. 0.75 means that 75% of each side length must be present in the segment array. Note that
    more of the head echo MAY be present depending on its randomly determined position.

    Addition of noise/clutter is handled in a separate routine.
    """

    # Create data square
    dim = (size_x, size_y)
    data = np.zeros(dim, dtype="complex128")

    # Determine position of center of head echo bounding box, taking into account possibility that it hangs off edge.
    num_pulses = np.shape(meteor_data)[0]
    num_samples = np.shape(meteor_data)[1]
    dxp = (2*edge_frac-1)*num_pulses/2
    dyp = (2*edge_frac-1)*num_samples/2
    xm = int(rng.uniform(low=-size_x/2+dxp, high=size_x/2-dxp))
    ym = int(rng.uniform(low=-size_y/2+dyp, high=size_y/2-dyp))

    if num_pulses % 2 == 0:
        pos_x = num_pulses//2
    else:
        pos_x = (num_pulses+1)//2
    if num_samples % 2 == 0:
        pos_y = num_samples//2
    else:
        pos_y = (num_samples+1)//2

    xm_low = xm-(pos_x-1)
    ym_low = ym-(pos_y-1)
    xm_high = xm_low+num_pulses
    ym_high = ym_low+num_samples

    # Define mapping between "center" and "lower-left" of size_x by size_y frame coordinates
    if size_x % 2 == 0:
        pos_x_f = size_x//2
    else:
        pos_x_f = (size_x+1)//2
    if size_y % 2 == 0:
        pos_y_f = size_y//2
    else:
        pos_y_f = (size_y+1)//2

    delta_x_f = pos_x_f-1
    delta_y_f = pos_y_f-1

    xm_low_f = xm_low + delta_x_f
    xm_high_f = xm_high + delta_x_f
    ym_low_f = ym_low + delta_y_f
    ym_high_f = ym_high + delta_y_f

    # Crop off edges of meteor if needed
    pulse_meteor_start = np.maximum(0, xm_low_f)
    pulse_meteor_end = np.minimum(xm_high_f, size_x)
    sample_meteor_start = np.maximum(0, ym_low_f)
    sample_meteor_end = np.minimum(ym_high_f, size_y)
    pulse_meteor_start_rel = pulse_meteor_start-xm_low_f
    pulse_meteor_end_rel = num_pulses - (xm_high_f-pulse_meteor_end)
    sample_meteor_start_rel = sample_meteor_start-ym_low_f
    sample_meteor_end_rel = num_samples - (ym_high_f-sample_meteor_end)

    # Place head echo into raw data segment
    data[pulse_meteor_start:pulse_meteor_end, sample_meteor_start:sample_meteor_end] = \
        meteor_data[pulse_meteor_start_rel:pulse_meteor_end_rel, sample_meteor_start_rel:sample_meteor_end_rel]

    # Return data segment along with bounds of the head echo (useful for other routines)
    return(data, (pulse_meteor_start, pulse_meteor_end, sample_meteor_start, sample_meteor_end))


def generate_alt_init(rng=np.random.default_rng(), dist_type="uniform"):
    """Function to determine the initial altitude of observation in meters.
    """

    if dist_type=="normal":
        r0 = rng.normal(100,10)
    else:
        r0 = rng.uniform(r0min,r0max)
    if r0 <= r0min:
        r0 = r0min
    elif r0 >= r0max:
        r0 = r0max
    
    # Convert kilometer value to meters
    r0 *= 1000

    return(r0)


def generate_meteor_init(vmin=11, vmax=73, amin=0, amax=55, tmin=0.008, tmax=0.12, snr_min=2, snr_max=25,
                         rng=np.random.default_rng()):
    """Generates the initial conditions for a meteor using random number generation
    with uniform distributions with physical limits."""

    # Generate random values...
    r0 = generate_alt_init(rng, "uniform") # m
    v0 = rng.uniform(vmin,vmax)  # km/s
    a0 = rng.uniform(amin,amax)  # km/s^2
    time_observed = rng.uniform(tmin, tmax)  # s
    max_snr = rng.uniform(snr_min,snr_max) # dB
    phi0 = rng.random()*2*np.pi  # rad

    # Impose some physical limits just in case (with uniform random values, shouldn't encounter these)
    if v0 <= 11:
        v0 = 11
    elif v0 >= 90:
        v0 = 90
    if a0 < 0.1:
        a0 = 0.1
    if time_observed < 0.008:
        time_observed = 0.008
    if time_observed > 0.15:
        time_observed = 0.15

    # Convert kilometer values to meters. Make velocity negative (going towards beam)
    #   (note that paper incorporates this sign change in equations to make v0 positive in the paper)
    v0 *= -1000
    a0 *= 1000

    # Everything gets returned in SI units (besides signal in dB)
    return(r0, v0, a0, phi0, max_snr, time_observed)


def generate_clutter(clutter, rng=np.random.default_rng(), meteor_data=None, bounds=None, clutter_coeff=1/2, gain_coeff=None):
    """Function to add a clutter or noise signal to a raw data segment that may or may not contain a head echo.
    The clutter variable is either a 2D complex array that contains the clutter (used for 2023 paper) or a string
    that points to an HDF5 containing examples of clutter. If a head echo is present, its bounding box is included
    as an input. This allows the total power of the head echo to be compared with the power of the clutter in the
    same region. If the head echo is too weak compared to the background clutter, its power is boosted to some
    minimum threshold. The clutter_coeff argument specifies this threshold as the minimum ratio of total head echo
    power to clutter power, i.e., if clutter_coeff=0.2, a head echo must have at least one-fifth the total power of
    the surrounding clutter/noise (note that with a coded long pulse, even with 1/5th the power of noise, 
    the head echo would be identifiable after a matched filter)
    """

    # Check if clutter argument is a string (path to .h5 file with examples of clutter) or an array (the clutter itself to use)
    if isinstance(clutter, str):

        # Get range of possible IDs
        id_max = get_max_id_hdf5(clutter)

        # Choose a random ID
        id = rng.integers(id_max)

        # Pull it from HDF5 file
        rawdata = load_data_hdf5_simple(clutter, id)["data"]

    else:
        # Just use the clutter input as rawdata (it should be a complex 2D numpy array or 3D real numpy array)
        rawdata = clutter

    # Choose a gain coefficient for the clutter, if not pre-specified.
    if gain_coeff is None:
        g = rng.uniform(0.4,1.2)
    else:
        g = gain_coeff

    # If rawdata is a 3D array, that means real/imag channels already separated. If it's a 2D array, it should be a complex array.
    # Make it so that clutter_data is a complex array either way.
    if rawdata.ndim == 3:
        clutter_data = g*(rawdata[0,:,:] + 1j*rawdata[1,:,:])
    else:
        clutter_data = g*rawdata

    # Boost signal power of meteor if it's too obscured by radar clutter, based on the clutter_coeff.
    if (meteor_data is not None) and (bounds is not None):

        # Calculate total power of each pulse/sample within the rawdata segment (both head echo and clutter) and sum them up.
        power_meteor_tot = np.sum(np.abs(meteor_data[bounds[0]:bounds[1], bounds[2]:bounds[3]])**2)
        power_clutter_tot = np.sum(np.abs(clutter_data[bounds[0]:bounds[1], bounds[2]:bounds[3]])**2)

        # Determine minimum total power threshold that the head echo will need
        min_power = clutter_coeff*power_clutter_tot

        # If the head echo is too weak compared to background clutter, boost its power
        if power_meteor_tot < min_power:
            c = np.sqrt(min_power/power_meteor_tot)
            meteor_data *= c

    return(clutter_data)


def generate_rxnoise_real(noise_file, rng=np.random.default_rng()):
    """Function to pull random examples of clutter/noise from a specified HDF5 file with examples collected from real data.
    """

    # Get range of possible example IDs in the HDF5
    id_max = get_max_id_hdf5(noise_file)

    # Choose a random example ID for a rawdata segment
    id = rng.integers(id_max)

    # Choose a gain coefficient for the noise
    g = rng.uniform(0.99, 1.01)

    # Get noise/clutter example from file, multiply by gain coefficient
    rawdata = load_data_hdf5_simple(noise_file, id)["data"]
    noise_data = g*(rawdata[0,:,:] + 1j*rawdata[1,:,:])

    return(noise_data)


def save_to_hdf5(data, time, rng, label, id, save_dir, data_type, facility, match=None):
    """Function to add a rawdata segment to an existing HDF5 file. Will be added with the id specified.
    """

    with h5py.File(f"{save_dir}/{data_type}.h5", "a") as h5file:
        data_str = f"{id:07d}-rawdata"
        data_grp = h5file.create_group(data_str)
        data_grp["data"] = data
        data_grp.attrs["label"] = label
        data_grp.attrs["id"] = id
        if match is not None:
            data_grp["match"] = match
        data_grp["time"] = time
        data_grp["rng"] = rng
    



def generate_set(data_type, save_dir, num_examples, pulses, samples, facility,
                 noise_file=None,
                 start_id=0, vmin=11, vmax=73, amin=0, amax=55,
                 tmin=0.008, tmax=0.12, snr_min=2, snr_max=25,
                 rng=np.random.default_rng(), plot=False, plot_verbose=False):
    """Function to generate a set of synthetic head echo + real clutter/noise raw data segments.
    Used to train the neural network at a given facility with facility parameters specified in
    the facility dictionary. The input arguments are described as follows:

    data_type: A string such as "train", "validation", or "test".
    save_dir: Directory in which to save everything from this routine, including HDF5 file with synthetic examples
    num_examples: Number of positive/negative training example pairs to generate. Note that the total number of examples will be 2x this number.
    pulses: Number of pulses on the x-axis in each raw data segment
    samples: Number of samples on the y-axis in each raw data segment
    facility: Dictionary containing facility parameters to use to generate synthetic head echoes.
    noise_file: HDF5 file containing pre-identified examples of noise/clutter at the facility. Each example should not have anything resembling a head echo.
    start_id: ID number at which to start saving examples to HDF5 (typically 0)
    vmin: Lowest head echo range rate toward facility that can be generated
    vmax: Highest head echo range rate toward facility that can be generated
    amin: Lowest head echo range deceleration toward facility that can be generated
    amax: Highest head echo range deceleration toward facility that can be generated
    tmin: Shortest length of head echo that can be generated
    tmax: Longest length of head echo that can be generated
    snr_min: Lowest peak raw data SNR of the generated head echoes
    snr_max: Highest peak raw data SNR of the generated head echoes
    rng: Random number generator object
    plot: Whether to also make RTI plots of each training example saved to HDF5 (this is slow, usually used w/ small number of examples)
    plot_verbose: Used for debugging
    """

    if plot:
        os.makedirs(f"{save_dir}/rti-match/{data_type}/", exist_ok=True)
        if plot_verbose:
            os.makedirs(f"{save_dir}/rti-match/{data_type}/meteor/", exist_ok=True)

    
    if noise_file:
        real_noise = True
    else:
        real_noise = False

    # If HDF5 enabled, make initial write to file. Error if it already exists.
    if start_id == 0:
        with h5py.File(f"{save_dir}/{data_type}.h5", "w-") as h5file:
            h5file.attrs["data_type"] = data_type
            
    
    j = 0
    for i in np.arange(start_id, start_id + num_examples):

        ## Generate positive example with a head echo

        # Generate ICs for a meteor
        r0_pos, v0, a0, phi0, max_snr, time_observed = generate_meteor_init(vmin=vmin, vmax=vmax, amin=amin, amax=amax,
                                            tmin=tmin, tmax=tmax, snr_min=snr_min, snr_max=snr_max, rng=rng)

        # Generate pulses
        rawdata_pos, samples_data = generate_meteor(facility, r0_pos, v0, a0, phi0,
                                    max_snr, time_observed, rng)
        
        start_rnggate_pos = samples_data[0]

        # Place meteor in data
        rawdata_pos, bounds = place_meteor_in_data(rawdata_pos, facility, pulses, samples, edge_frac=0.75, rng=rng)

        if plot_verbose:
            rawdata_power = mfAlgs_simple.powerLog(rawdata_pos, noisefloor=facility["NOISE_RAW"])
            fig, ax = mfAlgs_simple.plotRTI(rawdata_power, dBmin=0)
            fig.savefig(f"{save_dir}/rti-match/{data_type}/meteor/meteor-{j}.png")
            plt.close(fig)


        ## Generate negative example without a head echo (just a zero array)

        # Just generate an empty array
        rawdata_neg = np.zeros((pulses, samples), dtype="complex128")

        # Determine range gate at which this slice starts
        r0_neg = generate_alt_init(rng, "uniform")
        start_rnggate_neg = np.round(r0_neg/facility["drng"])


        ## For both positive/negative example, determine noise background
        if real_noise:  # Pull noise/clutter example from existing data
            noise_data = generate_rxnoise_real(noise_file, rng)
        else:  # Generate Gaussian noise (not used for 2023 ML paper, produced worse performance overall, maybe real noise is not quite Gaussian)
            noise_data = generate_rxnoise_polar((pulses, samples), facility["NOISE_RAW"], rng)


        # Determine gain coefficient for background noise/clutter
        g = rng.uniform(0.8, 1.8)

        ## Add clutter to positive example. May increase head echo power if it's too weak compared to the clutter being added.
        noise = generate_clutter(noise_data, rng, rawdata_pos, bounds, clutter_coeff=0.2, gain_coeff=g)
        rawdata_pos += noise

        ## Add same exact clutter to negative example.
        noise = generate_clutter(noise_data, rng, gain_coeff=g)
        rawdata_neg += noise


        # Calculate range and time spans
        ipp = facility["ipp"]
        drng = facility["drng"]
        num_pulses = np.shape(rawdata_pos)[0]
        num_rnggates = np.shape(rawdata_pos)[1]
        rng_span_pos = np.linspace(start_rnggate_pos*drng, (start_rnggate_pos+num_rnggates-1)*drng, num_rnggates)
        num_pulses = np.shape(rawdata_neg)[0]
        num_rnggates = np.shape(rawdata_neg)[1]
        rng_span_neg = np.linspace(start_rnggate_neg*drng, (start_rnggate_neg+num_rnggates-1)*drng, num_rnggates)
        time_span = np.linspace(0, (num_pulses-1)*ipp, num_pulses)

        # If plotting enabled, perform matched filter on data. This gets saved in the data file because it makes it 
        #   much easier to plot later to assess neural network performance.
        if plot:
                
            # Use filter with doppler shift based on initial meteor velocity
            match_pos = mfAlgs_simple.matchedFilter2D(rawdata_pos, facility["CODE"], params=facility, dopplerV=np.abs(v0))

            # Use filter with zero Doppler shift (since there is no meteor)
            match_neg = mfAlgs_simple.matchedFilter2D(rawdata_neg, facility["CODE"])

        else:
            match_pos = None
            match_neg = None


        # Create output arrays (need to have separate real and imaginary dimensions in a 3D array for the neural network)
        rawdata_output_pos = np.empty((2,np.shape(rawdata_pos)[0],np.shape(rawdata_pos)[1]))
        rawdata_output_pos[0,:,:] = np.real(rawdata_pos)
        rawdata_output_pos[1,:,:] = np.imag(rawdata_pos)
        rawdata_output_neg = np.empty((2,np.shape(rawdata_neg)[0],np.shape(rawdata_neg)[1]))
        rawdata_output_neg[0,:,:] = np.real(rawdata_neg)
        rawdata_output_neg[1,:,:] = np.imag(rawdata_neg)
        
        # Save positive and negative segments to HDF5.
        save_to_hdf5(rawdata_output_pos, time_span, rng_span_pos, 1, j, save_dir, data_type, facility, match=match_pos)
        j += 1
        save_to_hdf5(rawdata_output_neg, time_span, rng_span_neg, 0, j, save_dir, data_type, facility, match=match_neg)
        j += 1

        # Print progress to terminal
        if i % 100 == 0:
            print("Meteor ", i)
    
        



def main(num_meteors, facility, save_dir, data_type="train", pulses=20, samples=20,
         noise_file=None,
         vmin=11, vmax=73, amin=0, amax=55,
         tmin=0.008, tmax=0.12, snr_min=2, snr_max=25,
         seed=42, plot=False, plot_verbose=False):
    """Generates a set of synthetic data"""
    
    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)

    # Generate examples
    generate_set(data_type, save_dir, num_meteors, pulses, samples, facility,
                 noise_file=noise_file,
                 start_id=0, vmin=vmin, vmax=vmax, amin=amin, amax=amax,
                 tmin=tmin, tmax=tmax, snr_min=snr_min, snr_max=snr_max,
                 rng=rng, plot=plot, plot_verbose=plot_verbose)




if __name__ == "__main__":

    # Parameters used to run the script are set here.

    outer_dir = r"."
    
    plot=False
    if plot:
        plot_verbose=True
    else:
        plot_verbose = False

    # Specify which facility to generate data for (uncomment/comment as necessary)
    #facility = "RISR-N"
    #facility = "JRO"
    #facility = "MHO"

    # Specify physical bounds
    vmin = 11 # km/s
    vmax = 73 # km/s
    amin = 0.01 # km/s^2
    amax = 55 # km/s^2

    # At each facility, for each example, we take real examples of clutter/noise and add to synthetic echoes
    noise_enabled = True

    # Specific directory in which to generate synthetic data
    facility = "MHO"
    gen_data_dir = f"{outer_dir}/{facility}/synthetic_data/"

    # If plotting disabled, generate full-size set of training examples, otherwise generate smaller set.
    # Generate synthetic "validation" and "test" sets to be used when monitoring the training process.
    #   Note that we actually test against real data, and just keep a synthetic "test" set for comparison purposes.
    if plot==False:
        if facility == "JRO":
            num_train = 10000
            num_valid = 500
            num_test = 500
        elif facility == "RISR-N":
            num_train = 25000
            num_valid = 500
            num_test = 500
        else:
            num_train = 25000
            num_valid = 500
            num_test = 500
    else:
        num_train = 200
        num_valid = 10
        num_test = 10

    # Specify parameters that vary based on facility
    if facility=="RISR-N":
        square_size_x = 150
        square_size_y = 300
        snr_min = 0  # dB of head echo in raw data (not post-matched filter)
        snr_max = 8
        tmin = 0.0014*12
        tmax = 0.0014*60
    elif facility=="MHO":
        square_size_x = 150
        square_size_y = 150
        snr_min = 1
        snr_max = 14
        tmin = 0.002*12
        tmax = 0.002*60
    else:
        square_size_x = 150
        square_size_y = 150
        snr_min = 0
        snr_max = 20
        tmin = 0.00125*12
        tmax = 0.00125*60
    
    os.makedirs(gen_data_dir, exist_ok=True)

    # Directory of file containing clutter/noise examples
    if noise_enabled:
        noise_file = f"{outer_dir}/{facility}/noise_examples/noise_examples.h5"
    else:
        noise_file = None


    train_data_dir = "train"
    valid_data_dir = "valid"
    test_data_dir = "test"
    
    # Generate training data set. Largest set, used only to train the CNNs.
    main(num_train, facilParams[facility], data_type=train_data_dir, pulses=square_size_x,
         samples=square_size_y,
         noise_file=noise_file,
         vmin=vmin, vmax=vmax, amin=amin, amax=amax, tmin=tmin, tmax=tmax,
         snr_min=snr_min, snr_max=snr_max,
         save_dir=gen_data_dir, seed=224,
         plot=plot, plot_verbose=plot_verbose)


    # Generate validation set, used to monitor training progress.
    # Must make sure seed is different for the validation and test data!
    main(num_valid, facilParams[facility], data_type=valid_data_dir, pulses=square_size_x,
         samples=square_size_y,
         noise_file=noise_file,
         vmin=vmin, vmax=vmax, amin=amin, amax=amax, tmin=tmin, tmax=tmax,
         snr_min=snr_min, snr_max=snr_max,
         save_dir=gen_data_dir, seed=229,
         plot=plot, plot_verbose=plot_verbose)
    
    # Generate synthetic test set. Used only as a reference. Results from it are not included in 2023 paper.
    main(num_test, facilParams[facility], data_type=test_data_dir, pulses=square_size_x,
         samples=square_size_y,
         noise_file=noise_file,
         vmin=vmin, vmax=vmax, amin=amin, amax=amax, tmin=tmin, tmax=tmax,
         snr_min=snr_min, snr_max=snr_max,
         save_dir=gen_data_dir, seed=230,
         plot=plot, plot_verbose=plot_verbose)



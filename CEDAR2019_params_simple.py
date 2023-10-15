# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 08:51:51 2022

@author: thedges

Contains parameters and functions specific to the CEDAR 2019 data collect at
RISR-N, JRO, and MHO.

"""

import numpy as np


## Radar experiment stuff

# Speed of light
cLight = 299792458 # m/s

# Pulse codes at each facility with TRANSMISSION baud rate.
MSB_51 = np.array([ 1, 1, 1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1,
                   -1, -1, -1, 1, 1,  1, -1,  1,  1,  1, -1, -1,  1,
                   1, -1,  1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,
                   1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1, -1], dtype=float)
BARKER_7 = np.array([ +1, +1, +1, -1, -1, +1, -1], dtype=float)

# Specify code at each facility
CODE_RISR_TX = MSB_51
CODE_JRO_TX = MSB_51
CODE_MHO_TX = BARKER_7

# Transmit baud rates
baudTX_RISR = 1e-6 # microsec
baudTX_JRO = 1e-6
baudTX_MHO = 6e-6

# Ratio of transmission vs reception baud rates at each facility
baudRXTXratioRISR = 2
baudRXTXratioJRO = 1
baudRXTXratioMHO = 6

# Receive frequencies
fRX_RISR = 1/baudTX_RISR*baudRXTXratioRISR
fRX_JRO = 1/baudTX_JRO*baudRXTXratioJRO
fRX_MHO = 1/baudTX_MHO*baudRXTXratioMHO

# Pulse codes at each facility, with RECEIVING baud rate.
CODE_RISR = np.array([val for val in CODE_RISR_TX for _ in range(0,baudRXTXratioRISR)])
CODE_JRO = np.array([val for val in CODE_JRO_TX for _ in range(0,baudRXTXratioJRO)])
CODE_MHO = np.array([val for val in CODE_MHO_TX for _ in range(0,baudRXTXratioMHO)])

# Approximate mean noise (dB) at each facility. Empirically determined when
#   matched filter (not Doppler shifted) is applied.
NOISE_RISR = 20*np.log10(57.446) # TODO: find a better way to quantify noise
NOISE_JRO = 20*np.log10(251.19)
NOISE_MHO = 20*np.log10(236.64)

# Approximate mean noise (dB) at each facility in the raw data. Empirically
#   determined.
NOISE_RISR_RAW = 20*np.log10(5.0)
NOISE_JRO_RAW = 20*np.log10(31)
NOISE_MHO_RAW = 20*np.log10(29.681644159311652)

# Frequency at each facility (Hz)
fRISRN = 442.5e6
fJRO = 49.9e6
fMHO = 440e6

# Inter-pulse period at each facility (s)
ippRISRN = 0.0014
ippJRO = 0.00125
ippMHO = 0.002

# Altitude between range gates at each facility (m)
drngRISRN = cLight/(2*fRX_RISR)
drngJRO = cLight/(2*fRX_JRO)
drngMHO = cLight/(2*fRX_MHO)

# Code length (as received) at each facility
NRISRN = len(CODE_RISR)
NJRO = len(CODE_JRO)
NMHO = len(CODE_MHO)

# Elevation angle at each facility
angle_elev_RISRN = 86*(np.pi/180)  # RISR-N almost directly up
angle_elev_JRO = 90*(np.pi/180)  # JRO directly up
angle_elev_MHO = 45*(np.pi/180)  # MHO uses "diagonal" beam to avoid ground clutter
angle_azi_RISRN = 26*(np.pi/180)
angle_azi_JRO = 0
angle_azi_MHO = 270*(np.pi/180)  # due West

# Range of samples that describe the meteor region at each facility (approximately 70-140 km)
start_sample_RISRN = 960
end_sample_RISRN = 1894
start_sample_MHO = 760
end_sample_MHO = 1321
start_sample_JRO = 300
end_sample_JRO = 950

# Construct dict with params for each facilitiy
paramsRISRN = {"CODE_TX": CODE_RISR_TX, "BAUD_RXTX": baudRXTXratioRISR,
                    "CODE": CODE_RISR, "NOISE": NOISE_RISR, 
                    "NOISE_RAW": NOISE_RISR_RAW, "f": fRISRN,
                    "fRX": fRX_RISR, "ipp": ippRISRN, "N": NRISRN,
                    "drng": drngRISRN, "angle_elev": angle_elev_RISRN,
                    "angle_azi": angle_azi_RISRN, "fftmax": 1000,
                    "fftMax_dB_plot": 30,
                    "start_sample": start_sample_RISRN,
                    "end_sample": end_sample_RISRN,
                    "NFFT": 1024, "rngrate_low": 2000,
                    "rngrate_high": 100000,
                    "facility": "RISR-N"}
paramsJRO = {"CODE_TX": CODE_JRO_TX, "BAUD_RXTX": baudRXTXratioJRO,
                    "CODE": CODE_JRO, "NOISE": NOISE_JRO,
                    "NOISE_RAW": NOISE_JRO_RAW, "f": fJRO,
                    "fRX": fRX_JRO, "ipp": ippJRO, "N": NJRO,
                    "drng": drngJRO,
                    "angle_elev": angle_elev_JRO,
                    "angle_azi": angle_azi_JRO, "fftmax": 10000,
                    "fftMax_dB_plot": 56,
                    "start_sample": start_sample_JRO,
                    "end_sample": end_sample_JRO,
                    "NFFT": 1024, "rngrate_low": 2000,
                    "rngrate_high": 100000,
                    "facility": "JRO"}
paramsMHO = {"CODE_TX": CODE_MHO_TX, "BAUD_RXTX": baudRXTXratioMHO,
                    "CODE": CODE_MHO, "f": fMHO, "fRX": fRX_MHO,
                    "ipp": ippMHO, "N": NMHO, "drng": drngMHO,
                    "NOISE": NOISE_MHO, "NOISE_RAW": NOISE_MHO_RAW,
                    "angle_elev": angle_elev_JRO,
                    "angle_azi": angle_azi_JRO, "fftmax": 5000,
                    "fftMax_dB_plot": 30,
                    "start_sample": start_sample_MHO,
                    "end_sample": end_sample_MHO,
                    "NFFT": 1024, "rngrate_low": -30000,
                    "rngrate_high": 72000,
                    "facility": "MHO"}

facilParams = {"RISR-N": paramsRISRN, "JRO": paramsJRO, "MHO": paramsMHO}


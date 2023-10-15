# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 08:17:30 2022

@author: thedges

Contains functions that can be used for any HPLA phase-coded pulse meteor head
echo experiment.

"""

import numpy as np
import matplotlib.pyplot as plt
from CEDAR2019_params_simple import cLight


def matchedFilter2D(rawdata, code, params=None, dopplerV=0):
    """ Run matched filter on the complex 2D raw data array.
    It is assumed pulses (time) are on the first axis and samples (altitude) on
    the second axis
    
    rawdata is of shape (numPulses, numSamples) or (numPulses, numChannels, numSamples)
    if interferometric data is present.
    code is of shape (codeLen,)
    """

    if params is not None:
        # Unpack parameters of interest
        #code = params["CODE"]
        f0 = params["f"]
        fRX = params["fRX"]

    nvals = np.arange(len(code))
    if dopplerV != 0:
        if params is None:
            raise ValueError("If using a Doppler shift, must pass in params "+\
                             "dict with f0 (carrier frequency) " + \
                             "and fRX (receive frequency)")
        else:
            code = code*np.exp(dopplerV * nvals * (4 * np.pi * 1j * f0 / (fRX * cLight)))

    # If rawdata is a 3D array, interferometric data is present. We don't need this for a
    #   simple matched-filter, so just sum channels together.
    if rawdata.ndim == 3:
        rawdata_total = np.sum(rawdata, axis=1)
    else:
        rawdata_total = rawdata
    
    codeLen = np.shape(code)[0]
    numPulses = np.shape(rawdata_total)[0]
    numSamples = np.shape(rawdata_total)[1]
    
    # Zero-pad the code
    codePad = np.pad(code, pad_width=(0, numSamples-codeLen))
    
    # Take the FFT of both the transmitted code (1D) and received data (2D)
    # Conjugate on txFFT acts as time-reversal in time-domain, necessary to
    #   cross-correlate txFFT/rxFFT instead of convolve
    txFFT = np.conj(np.fft.fft(codePad)) 
    rxFFT = np.fft.fft(rawdata_total, axis=1)
    
    # Cross-correlate transmitted/received codes (i.e. multiply FFTs) and
    #   take inverse FFT to obtain matched-filter result
    match = np.fft.ifft(np.multiply(txFFT,rxFFT), axis=1)
    
    return(match)


def phase_diff(meteor_vals_1, meteor_vals_2):
    div_12 = meteor_vals_1/meteor_vals_2
    diff_12 = div_12[1:]/div_12[:-1]
    diff_angle_12 = np.angle(diff_12)
    angle_12_unwrap = np.cumsum(np.pad(diff_angle_12, (1,0))) + np.angle(div_12[0])
    return(angle_12_unwrap)



def powerLog(data, noisefloor = 0):
    """
    Get power in dB of complex data. Noisefloor is in dB.
    """
    
    power = np.abs(data)
    power[power <= 1e-10] = 1e-10 # Ensure log(nonpositive) does not occur
    power = 20*np.log10(power) - noisefloor
    
    return(power)
    
    
def plotRTI(data, alt=None, time=None, dopplerVel=None, dBmin=5, dBmax=40,
            colorbar="plasma", figsize=(6,4), title=None, aspect="auto"):
    """
    Plot range-time intensity (RTI) plot. Assumes data is a real array,
      typically in units of dB.
    """
    
    if alt is None:
        rngStart = -0.5
        rngEnd = np.shape(data)[1]-0.5
        ylabel = "Sample"
    else:
        rngStart = alt[0]-(alt[1]-alt[0])/2
        rngEnd = alt[-1]-(alt[1]-alt[0])/2
        ylabel = "Altitude (km)"
    
    numPulses = np.shape(data)[0]
    
    if time is None:
        timeStart = -0.5
        timeEnd = numPulses-0.5
        xlabel="Pulse"
    else:
        timeStart = time[0]-(time[1]-time[0])/2
        timeEnd = time[-1]-(time[1]-time[0])/2
        xlabel="Time (sec)"
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cplot = ax.imshow(np.transpose(data),
                      extent=[timeStart, timeEnd,
                              rngStart, rngEnd],
                      origin="lower", aspect=aspect, interpolation=None,
                      vmin=dBmin, vmax=dBmax, cmap=colorbar)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    #ax.set_ylim([self.startRange, self.endRange])
    if dopplerVel is not None:
        ax.set_title("v = " + str(dopplerVel))
    cbar = fig.colorbar(cplot)
    cbar.set_label("Signal Strength (dB)")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    
    return(fig, ax)


def plot2RTI(data1, data2, alt=None, time=None, dopplerVel=None, dBmin=5,
             dBmax=40, colorbar="plasma", figsize=(12,4), title=None,
             aspect="auto"):
    """Plot two range-time intensity (RTI) plots side-by-side. Assumes both
    data arrays to plot are the same size/axes.
    """
    
    if alt is None:
        rngStart = -0.5
        rngEnd = np.shape(data1)[1]-0.5
        ylabel = "Sample"
    else:
        rngStart = alt[0]-(alt[1]-alt[0])/2
        rngEnd = alt[-1]-(alt[1]-alt[0])/2
        ylabel = "Altitude (km)"
    
    numPulses = np.shape(data1)[0]
    
    if time is None:
        timeStart = -0.5
        timeEnd = numPulses-0.5
        xlabel="Pulse"
    else:
        timeStart = time[0]-(time[1]-time[0])/2
        timeEnd = time[-1]-(time[1]-time[0])/2
        xlabel="Time (sec)"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cplot1 = ax1.imshow(np.transpose(data1),
                       extent=[timeStart, timeEnd,
                               rngStart, rngEnd],
                       origin="lower", aspect=aspect, interpolation=None,
                       vmin=dBmin, vmax=dBmax, cmap=colorbar)
    cplot2 = ax2.imshow(np.transpose(data2),
                        extent=[timeStart, timeEnd,
                                rngStart, rngEnd],
                        origin="lower", aspect=aspect, interpolation=None,
                        vmin=dBmin, vmax=dBmax, cmap=colorbar)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)

    #ax.set_ylim([self.startRange, self.endRange])
    if dopplerVel is not None:
        ax1.set_title("v = " + str(dopplerVel))
        ax2.set_title("v = " + str(dopplerVel))
    cbar = fig.colorbar(cplot1)
    #cbar2 = fig.colorbar(cplot2)
    cbar.set_label("Signal Strength (dB)")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    
    return(fig, (ax1, ax2))


def plotNRTI(datas, alt=None, time=None, dopplerVel=None, dBmin=5,
             dBmax=40, colorbar="plasma", figsize=None, title=None,
             aspect="auto", colorbar_loc="left"):
    
    n_plots = len(datas)

    if figsize is None:
        figsize = (5*n_plots, 4)

    if alt is None:
        rngStart = -0.5
        rngEnd = np.shape(datas[0])[1]-0.5
        ylabel = "Sample"
    else:
        rngStart = alt[0]-(alt[1]-alt[0])/2
        rngEnd = alt[-1]-(alt[1]-alt[0])/2
        ylabel = "Altitude (km)"
    
    numPulses = np.shape(datas[0])[0]
    
    if time is None:
        timeStart = -0.5
        timeEnd = numPulses-0.5
        xlabel="Pulse"
    else:
        timeStart = time[0]-(time[1]-time[0])/2
        timeEnd = time[-1]-(time[1]-time[0])/2
        xlabel="Time (sec)"
    

    fig, ax_tuple = plt.subplots(1, n_plots, figsize=figsize)

    cplot_list = []
    for i in range(n_plots):
        cplot = ax_tuple[i].imshow(np.transpose(datas[i]),
                        extent=[timeStart, timeEnd,
                                rngStart, rngEnd],
                        origin="lower", aspect=aspect, interpolation=None,
                        vmin=dBmin, vmax=dBmax, cmap=colorbar)
        cplot_list.append(cplot)
        
        ax_tuple[i].set_xlabel(xlabel)
        ax_tuple[i].set_ylabel(ylabel)


    if colorbar_loc == "right":
        cbar = fig.colorbar(cplot_list[-1])
    else:
        cbar = fig.colorbar(cplot_list[0])

    cbar.set_label("Signal Strength (dB)")
    if title is not None:
        if dopplerVel is not None:
            suptitle = f"{title}, v = {dopplerVel}"
        else:
            suptitle = title
        fig.suptitle(suptitle)

    fig.tight_layout()
    
    return(fig, ax_tuple)



def plot3RTI(data1, data2, data3, alt=None, time=None, dopplerVel=None, dBmin=5,
             dBmax=40, colorbar="plasma", figsize=(18,4), title=None,
             aspect="auto"):
    """Routine used to plot rawdata, match, and FFT spectral peak magnitude alongside each other
    """

    if alt is None:
        rngStart = -0.5
        rngEnd = np.shape(data1)[1]-0.5
        ylabel = "Sample"
    else:
        rngStart = alt[0]-(alt[1]-alt[0])/2
        rngEnd = alt[-1]-(alt[1]-alt[0])/2
        ylabel = "Altitude (km)"
    
    numPulses = np.shape(data1)[0]
    
    if time is None:
        timeStart = -0.5
        timeEnd = numPulses-0.5
        xlabel="Pulse"
    else:
        timeStart = time[0]-(time[1]-time[0])/2
        timeEnd = time[-1]-(time[1]-time[0])/2
        xlabel="Time (sec)"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    cplot1 = ax1.imshow(np.transpose(data1),
                       extent=[timeStart, timeEnd,
                               rngStart, rngEnd],
                       origin="lower", aspect=aspect, interpolation=None,
                       vmin=dBmin, vmax=dBmax, cmap=colorbar)
    cplot2 = ax2.imshow(np.transpose(data2),
                        extent=[timeStart, timeEnd,
                                rngStart, rngEnd],
                        origin="lower", aspect=aspect, interpolation=None,
                        vmin=dBmin, vmax=dBmax, cmap=colorbar)
    cplot3 = ax3.imshow(np.transpose(data3),
                        extent=[timeStart, timeEnd,
                                rngStart, rngEnd],
                        origin="lower", aspect=aspect, interpolation=None,
                        vmin=dBmin, vmax=dBmax, cmap=colorbar)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    
    if dopplerVel is not None:
        ax1.set_title("v = " + str(dopplerVel))
    
    cbar = fig.colorbar(cplot1)
    cbar.set_label("Signal Strength (dB)")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    
    return(fig, (ax1, ax2, ax3))


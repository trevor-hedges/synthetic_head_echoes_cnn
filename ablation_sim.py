# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:47:01 2023

@author: thedges
"""

import numpy as np
from CEDAR2019_params_simple import cLight

def simulate_trajectory_1D_exp(t, r0, v0, a0, phi0, f0):
    
    # Use simple exponential deceleration equations to simulate deceleration
    #   of meteoroid. Probably oversimplified.
    
    rt = (r0 - v0**2/a0) + v0**2/a0*np.exp(a0/v0*t)
    vt = v0*np.exp(a0/v0*t)
    at = a0*np.exp(a0/v0*t)
    phit = phi0 - 4*np.pi*f0*v0**2/(cLight*a0)*(np.exp(a0/v0*t)-1)
    
    return(rt, vt, at, phit)


def simulate_trajectory_1D_odes_simple(r0, v0, a0, phi0, params):
    
    # To be implemented
    
    pass
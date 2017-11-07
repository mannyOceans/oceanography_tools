#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:05:57 2017

@author: manishdevana
"""

import numpy as np
import scipy.signal as sig
import scipy
import seawater as sw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data_load
import gsw
import cmocean
import oceans as oc
import Internal_wave_properties as iwp


if 'ladcp' not in locals(): 
    ladcp, ctd, bathy = data_load.load_data()
    rho_neutral =  np.genfromtxt('neutral_rho.csv', delimiter=',')
    strain = np.genfromtxt('strain.csv', delimiter=',')
    wl_max=350
    wl_min=100
    lambdaH, kh, omega, N2, dist, depths,\
        U, V, p_ladcp, Uspec, Vspec, etaSpec = iwp.frequencyEstimator(ctd, ladcp, bathy,\
                                            rho_neutral,strain, full_set=True)
    
    

def doppler_shifts(kh, ladcp, avg=1000):
    """
    Doppler shift the internal frequency to test for lee waves
    using the depth averaged floww
    """
    U, V, p_ladcp = oc.loadLADCP(ladcp)
    
    dz = int(np.nanmean(np.gradient(p_ladcp, axis=0)))
    window = int(np.ceil(avg/dz))
    Ubar = []
    for u, v in zip(U.T, V.T):
        mask = np.isfinite(u)
        u = u[mask]
        v = v[mask]
        u = np.nanmean(u[-window:])
        v = np.nanmean(v[-window:])
        Ubar.append(np.sqrt(u**2 + v**2))
        
    Ubar = np.vstack(Ubar)
    dshift = []
    for cast, ubar in zip(kh.T, Ubar):
        dshift.append(cast*ubar)
    
    dshift = np.vstack(dshift).T
    
    return dshift



def lee_wave_tests(kh, omega, N2, ctd, ladcp, dist, depths, plots=False):
    """
    Testing whether or not the observations can be attributed to lee waves
    """
    S, T, p_ctd, lat, lon = oc.loadCTD(ctd)
    
    dshift = doppler_shifts(kh, ladcp)
    
    # Test whether K*U is between N and f
    f = np.abs(np.nanmean(gsw.f(lat)))
    
    dshiftTest = np.full_like(kh, np.nan)
    
    for i, dump in enumerate(kh.T):
        dshiftTest[:,i] = np.logical_and(dshift[:,i] >= f, dshift[:,i]<= np.sqrt(N2[:,i]))
    
    # Test phase of velocity and isopycnal perturbations
    
    
    
    
    if plots:
        fig = plt.figure()
        plt.contourf(dist, np.squeeze(p_ladcp), U, cmap='seismic')
        plt.colorbar()
        plt.pcolormesh(dist, np.squeeze(depths), dshiftTest, cmap='binary', alpha=.2)
        plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
        plt.ylim(0, 4000)
        plt.gca().invert_yaxis()
        plt.title("u' with bins with f < Kh*U < N")
        plt.xlabel('Distance Along Transect (km)')
        plt.ylabel('Pressure (dB)')
        
       
    




def bathy_spectra(nfft=64):
    """ 
    load bathy in line with flow and do spectrum analysis to 
    see if the horizontal wavelengths fit with lamda values
    """
    
    bathy_slice = -1*np.genfromtxt('bathy_slices.csv', delimiter=',')
    lon_slices = np.genfromtxt('lon_slices.csv', delimiter=',')
    lat_slices = np.genfromtxt('lat_slices.csv', delimiter=',')
    
    stns = bathy_slice.shape[1]
    
    PS = [];
    specGrid = []
    for i, stn in enumerate(bathy_slice.T):
        
        mask = np.isfinite(stn)
        dist = gsw.distance(lon_slices[:], lat_slices[:,i], p=0)
        dist = np.cumsum(dist)/1000
        dx = np.nanmean(np.diff(dist))
        fs = 1./dx
        Nc = .5*dx
        specI  = 1/(Nc*(np.arange(0 + np.spacing(1), nfft))/(nfft/2))
        specI[0] = 0;
        specGrid.append(specI)
        dataIn = scipy.signal.detrend(stn[mask])
        PS.append(np.abs(scipy.fftpack.fft(dataIn, n=nfft)))
        
    
    fig = plt.figure()

    for grid, stn in zip(specGrid, PS):
        plt.scatter(grid[1:], stn[1:])

    
    
        
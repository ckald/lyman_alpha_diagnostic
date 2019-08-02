# -*- coding: utf-8 -*-

"""Main module."""

import pandas as pd
import numpy as np

import numpy.random
from scipy.optimize import bisect
from scipy.ndimage.filters import gaussian_filter1d
from linetools.spectra.xspectrum1d import XSpectrum1D
import astropy.units as u

from .utils import repeat


def read_los(specdata, key='h1/Flux'):
    """ Import line-of-sight (LOS) data from HDF5 file """
    vel = pd.Series(specdata['VHubble_KMpS'])
    fluxes = {}
    TOTAL = 0
    while 'Spectrum{}'.format(TOTAL) in specdata:
        h5flux = specdata['Spectrum{}'.format(TOTAL)]
        flux = pd.Series(h5flux[key])
        fluxes[TOTAL] = flux
        TOTAL += 1
    return pd.DataFrame(fluxes).set_index(vel)


def apply_fwhm(specdata, fwhm=0):
    """ Smooth the LOS using Gaussian blur to simulate
        experimental resolution """
    if not fwhm:
        return specdata

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dv = specdata.index[1] - specdata.index[0]
    sig = sigma / dv

    fluxes = {}
    for i in specdata.__iter__():
        fluxes[i] = gaussian_filter1d(specdata[i], sig)
    return pd.DataFrame(fluxes).set_index(specdata.index)


def rebin(specdata, dv=1):
    """ Re-bin the LOS using `dv` km/s bins.
        CAUTION: loses the first and the last points """
    vmin = specdata.index.min()
    vmax = specdata.index.max()
    v_rebin = np.linspace(vmin, vmax, num=(vmax - vmin)/dv - 1, endpoint=True)
    v_rebin_units = v_rebin * u.km / u.s

    rebinned = {}
    for i in specdata:
        obj = XSpectrum1D.from_tuple((specdata.index.values * u.km / u.s,
                                      specdata[i].values))
        objrebin = obj.rebin(v_rebin_units)
        rebinned[i] = np.array(objrebin.flux[1:-1])

    return pd.DataFrame(rebinned).set_index(v_rebin[1:-1])


def add_noise(specdata, SN=np.inf):
    """ Add Gaussian noise on top of LOS with standard deviation `1/SN`
        CAUTION: This is incorrect. The real data has both
                 Gaussian and Poisson noise."""
    if SN == np.inf:
        return specdata
    return specdata + numpy.random.normal(size=specdata.shape) / SN


def calculate_tau(flux):
    """ Compute effective optical depth of all given LOS """
    return -np.log(flux.mean().mean())


def rescale_fluxes(flux, tau):
    """ Iteratively linearly rescale optical depth of the data
        until the effective optical depth matches `tau`.
        Implemented by bringing the flux to some power. """

    # F = e^(-t)
    # t' = t * <t'>/<t>
    # F' = e^(-t') = e^(-t * <t'> / <t>) = F^(<t'>/<t>)
    def equation(x):
        flux_rescaled = flux.apply(lambda f: f**x, axis=0)
        return calculate_tau(flux_rescaled) - tau
    factor = bisect(equation, 1e-10, 1e10, xtol=1e-3)
    flux_rescaled = flux.apply(lambda f: f**factor, axis=0)
    return flux_rescaled


def power_spectrum(specdata):
    """ Compute power spectrum of the sampled flux.
        `specdata` is an array of LOS with shape (N_los, N_points)
        Returns power spectrum P(k) and array of wavenumbers k """
    N = len(specdata[0].index)
    dv = specdata[0].index[1] - specdata[0].index[0]
    V = dv * (len(specdata.index) - 1)

    freqs = np.fft.fftfreq(N) * (2 * np.pi / dv)
    # compute indexes to sort points by ascending wavenumber
    idx = np.argsort(freqs)
    # discard negative frequencies -- they are redundant for a real function
    idx = idx[freqs[idx] >= 0]
    freqs = freqs[idx]

    mean = specdata.mean().mean()
    overdelta = specdata.apply(lambda f: f / mean - 1.)
    fourier = np.fft.fft(overdelta, axis=0)[idx, :] * dv / V
    return np.abs(fourier)**2 * V / np.pi, freqs


def concatenate_spectra(short_spectra, concat=7, sample_count=10000):
    """ Generate randomly concatenated "long" spectra.
        Returns `sample_count` spectra `concat` times longer than original. """
    if concat == 1:
        return short_spectra

    long_spectra = {}
    for i, chunk in enumerate(repeat(
        lambda: short_spectra.columns.to_series().sample(n=concat),
        sample_count
    )):
        long_spectra[i] = [short_spectra[spec] for spec in chunk]
        long_spectra[i] = pd.concat(long_spectra[i], ignore_index=True)

    # Recompute index
    dv = short_spectra.index[1] - short_spectra.index[0]
    vel = dv * np.arange(len(long_spectra[0]))
    return pd.DataFrame(long_spectra).set_index(vel)


def generate_samples(power_spectra, sample_size=5, sample_count=10000):
    """ Generate real data-like samples of size `sample_size` """
    return repeat(
        lambda: power_spectra.T[np.random.choice(
            power_spectra.shape[1],
            size=sample_size, replace=False
        )], sample_count
    )

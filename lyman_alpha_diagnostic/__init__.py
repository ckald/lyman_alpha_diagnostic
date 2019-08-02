# -*- coding: utf-8 -*-

"""Top-level package for Lyman-alpha diagnostic."""
import numpy as np
import pandas as pd

from .data import experiment_fwhm, experiment_binning
from .spectra import (
    read_los, rescale_fluxes, calculate_tau, apply_fwhm, rebin, add_noise,
    concatenate_spectra, generate_samples,
    power_spectrum
)
from .utils import binned, build_kbins


__author__ = """Andrii Magalich"""
__email__ = 'andrew.magalich@gmail.com'
__version__ = '0.1.0'


class AveragePS(object):
    def __init__(self, h5file, tau_eff=None, SN=None):
        # Import LOS from HDF5 file instance
        self.short_spectra = read_los(h5file)
        # Rescale flux to fit `tau_eff`
        if tau_eff:
            self.short_spectra = rescale_fluxes(self.short_spectra, tau_eff)
        # Apply instrumental resolution
        self.short_spectra = apply_fwhm(self.short_spectra,
                                        fwhm=experiment_fwhm['HIRES'])
        # Rebin data
        self.short_spectra = rebin(self.short_spectra,
                                   dv=experiment_binning['HIRES'])
        # Rescale flux again in case it was affected
        if tau_eff:
            self.short_spectra = rescale_fluxes(self.short_spectra, tau_eff)
        # Apply noise
        if SN:
            self.short_spectra = add_noise(self.short_spectra, SN)

        # Cache `tau_eff`
        self.tau_eff = calculate_tau(self.short_spectra)

        # Create "long spectra"
        self.long_spectra = concatenate_spectra(self.short_spectra,
                                                concat=1, sample_count=1000)

        # Compute Flux Power Spectrum
        self.power_spectra, self.freqs = power_spectrum(self.long_spectra)
        self.kbins, self.ks = build_kbins(minlogk=-2.4, maxlogk=-1, bins=8)

        # Generate experimental-like samples
        self.samples = {}
        for i, sample in enumerate(generate_samples(self.power_spectra,
                                                    sample_size=35,
                                                    sample_count=1000)):
            self.samples[i] = binned(sample.mean(axis=0),
                                     self.freqs, self.kbins)
        self.samples = pd.DataFrame(self.samples).set_index(self.ks)

        # Compute average FPS
        self.ps = np.array(np.mean(self.samples, axis=1))

    def confidence_interval(self, width=95.):
        return (
            np.percentile(self.samples, (100-width)/2., axis=1),
            np.percentile(self.samples, 100 - (100-width)/2., axis=1)
        )

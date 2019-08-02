# -*- coding: utf-8 -*-

"""Top-level package for Lyman-alpha diagnostic."""
import numpy as np
import pandas as pd

from data import experiment_fwhm, experiment_binning
from spectra import (
    read_los, rescale_fluxes, calculate_tau, apply_fwhm, rebin, add_noise,
    concatenate_spectra, generate_samples,
    power_spectrum
)
from utils import binned, build_kbins


__author__ = """Andrii Magalich"""
__email__ = 'andrew.magalich@gmail.com'
__version__ = '0.1.0'


class AveragePS(object):
    def __init__(self, h5file, tau_eff, noise):
        self.short_spectra = read_los(h5file)
        self.short_spectra = rescale_fluxes(self.short_spectra, tau_eff)
        self.short_spectra = apply_fwhm(self.short_spectra,
                                        fwhm=experiment_fwhm['HIRES'])
        self.short_spectra = rebin(self.short_spectra,
                                   dv=experiment_binning['HIRES'])
        self.short_spectra = rescale_fluxes(self.short_spectra, tau_eff)
        self.short_spectra = add_noise(self.short_spectra, noise)

        self.tau_eff = calculate_tau(self.short_spectra)

        self.long_spectra = concatenate_spectra(self.short_spectra,
                                                concat=1, sample_count=1000)

        self.power_spectra, self.freqs = power_spectrum(self.long_spectra)
        self.kbins, self.ks = build_kbins(minlogk=-2.4, maxlogk=-1, bins=8)

        self.samples = {}
        for i, sample in enumerate(generate_samples(self.power_spectra,
                                                    sample_size=35,
                                                    sample_count=1000)):
            self.samples[i] = binned(sample.mean(axis=0),
                                     self.freqs, self.kbins)
        self.samples = pd.DataFrame(self.samples).set_index(self.ks)

        self.ps = np.array(np.mean(self.samples, axis=1))
        self.lower = np.percentile(self.samples, 2.5, axis=1)
        self.upper = np.percentile(self.samples, 97.5, axis=1)

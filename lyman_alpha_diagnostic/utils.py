import numpy as np


def build_kbins(minlogk=-2.25, maxlogk=-0.65, bins=17):
    """ Build logarithmic binning and corresponding bin centers """
    kbins = 10**np.linspace(minlogk, maxlogk, num=bins)
    kcenters = np.exp(0.5 * (np.roll(np.log(kbins), -1) + np.log(kbins)))[:-1]
    return kbins, kcenters


def binned(spectra, freqs, kbins):
    """ Average each of `spectra` in bins `kbins`. """

    # Work also with a single spectrum
    if len(spectra.shape) == 1:
        spectra = spectra.reshape(-1, *spectra.shape)

    digitized = np.digitize(freqs, kbins)
    return np.array([
        spectra[:, digitized == i].mean() for i in range(1, len(kbins))
    ])


def repeat(fun, n):
    """ Generator that calls a function `fun` repeatedly `n` times """
    for i in range(n):
        yield fun()

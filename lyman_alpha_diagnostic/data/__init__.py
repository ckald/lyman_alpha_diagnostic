from pkg_resources import resource_filename

import pandas as pd
import numpy as np


experiment_fwhm = {  # km/s
    'MIKE': 13.6,
    'HIRES': 6.7
}
experiment_binning = {  # km/s
    'MIKE': 5.0,
    'HIRES': 2.1
}


def import_viel_data():
    """ Import power spectrum data by Viel et al. 2013 """
    filename = resource_filename(__name__, 'powerspectrahiresmike_final.dat')
    data = pd.read_csv(
        filename,
        sep='\s+',
        header=None,
        names=[
            'z', 'log10 k',
            'ESI val', 'ESI err',
            'MIKE val', 'MIKE err',
            'HIRES val', 'HIRES err'
        ],
        skiprows=0
    )

    return data


def import_boera_data():
    """ Import power spectrum data by Boera et al. 2018 """
    filename = resource_filename(__name__, 'boera.dat')
    data = pd.read_csv(
        filename,
        sep='\s+',
        header=None,
        names=[
            'z', 'log10 k',
            'HIRES uncorr', 'HIRES val', 'HIRES err'
        ],
        skiprows=2
    )
    data['HIRES val'] = data['HIRES uncorr'] * 10**data['log10 k'] / np.pi
    data['HIRES err'] = data['HIRES err'] * 10**data['log10 k'] / np.pi
    return data


def load_viel_icovmat():
    """ Import inverse covariance matrix by Viel et al. 2013 """

    filename = resource_filename(__name__, 'icovmat.txt')
    lines = open(filename, 'r').readlines()
    invcovmatrix = []
    for line in lines[8:]:
        invcovmatrix.append([float(item) for item in line.split()])

    invcovmatrix = np.array(invcovmatrix)

    column_map = [
        {'z': 4.2, 'num': 7, 'label': 'MIKE'},
        {'z': 4.2, 'num': 7, 'label': 'HIRES'},
        {'z': 4.6, 'num': 7, 'label': 'MIKE'},
        {'z': 4.6, 'num': 7, 'label': 'HIRES'},
        {'z': 5.0, 'num': 7, 'label': 'MIKE'},
        {'z': 5.0, 'num': 7, 'label': 'HIRES'},
        {'z': 5.4, 'num': 7, 'label': 'HIRES'},
    ]
    ds = []
    col = 0
    for i, map in enumerate(column_map):
        ds.append(dict(
            matrix=invcovmatrix[col:col+map['num'], col:col+map['num']],
            **map
        ))
        col += map['num']

    return pd.DataFrame(ds).set_index(['z', 'label'])


def plot_viel_data(ax, data, redshift):
    """ Plot data points with errors by Viel et al. 2013
        `dataz` is an output of `import_viel_data` filtered by redshfit. """

    # Remove large scale biased points [Viel et al. 2013]
    dataz = dict(list(data[data['log10 k'] > -2.4].groupby('z')))
    dataz = dataz[redshift]

    if 'MIKE val' in dataz:
        ax.errorbar(10**dataz['log10 k'], dataz['MIKE val'],
                    yerr=dataz['MIKE err'],
                    capthick=1, elinewidth=1, ls='',
                    capsize=5, marker='o', label=r'MIKE', color='r')
    if 'HIRES val' in dataz:
        ax.errorbar(10**dataz['log10 k'], dataz['HIRES val'],
                    yerr=dataz['HIRES err'],
                    capthick=1, elinewidth=1, ls='',
                    capsize=5, marker='o', label=r'HIRES', color='b')

    return ax

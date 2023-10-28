import numpy as np
import os
import glob
from pathlib import Path
from matplotlib import pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table,Column
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from scipy.interpolate import UnivariateSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import peak_local_max
from astropy.stats import sigma_clip, gaussian_fwhm_to_sigma
from numpy.polynomial.chebyshev import chebfit, chebval
from astropy.modeling.models import Gaussian1D, Chebyshev2D
from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib import gridspec, rcParams, rc
from IPython.display import Image

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 15
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# Define your working directory
HOME = Path.home()
WD = HOME/'C:'/'SNU'/'2022_2'/'AO2'
SUBPATH = WD/'data'

Biaslist = sorted(list(Path.glob(SUBPATH, 'Bias*.fit')))
Darklist= sorted(list(Path.glob(SUBPATH, 'Cali*d*.fit')))
Flatlist = sorted(list(Path.glob(SUBPATH, 'Flat*.fit')))
Complist = sorted(list(Path.glob(SUBPATH, 'Comp*.fit')))

Objectlist1 = sorted(list(Path.glob(SUBPATH, 'jupiter_h*.fit')))
Objectlist2 = sorted(list(Path.glob(SUBPATH, 'jupiter_v*.fit')))
Objectlist3 = sorted(list(Path.glob(SUBPATH, 'saturn_h*.fit')))
Objectlist4 = sorted(list(Path.glob(SUBPATH, 'saturn_v*.fit')))

Standlist = sorted(list(Path.glob(SUBPATH, 'HR9087*.fit')))

# Checking the paths
print(Table([Biaslist], names=['Bias']), 2*'\n')
print(Table([Darklist], names=['Darklist']), 2*'\n')
print(Table([Flatlist], names=['Flatlist']), 2*'\n')
print(Table([Complist], names=['Complist']), 2*'\n')
print(Table([Objectlist1], names=['Objectlist1']), 2*'\n')
print(Table([Objectlist2], names=['Objectlist2']), 2*'\n')
print(Table([Objectlist3], names=['Objectlist3']), 2*'\n')
print(Table([Objectlist4], names=['Objectlist4']), 2*'\n')
print(Table([Standlist], names=['Standlist']), 2*'\n')
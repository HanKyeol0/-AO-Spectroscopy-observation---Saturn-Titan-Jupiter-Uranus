%config InlineBackend.figure_format = 'retina'
# %matplotlib widget
# %matplotlib notebook

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
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# Define your working directory
HOME = Path.home()
WD = HOME/'Desktop'/'AO2'
SUBPATH = WD/'data'
RAWPATH = WD/'2022-11-04'
SPECPATH = WD/'spec'

DATAPATH = Path("C:/SNU/2022_2/AO2/data")

Biaslist = sorted(list(Path.glob(DATAPATH, 'cali*bias.fit')))

Darklist= sorted(list(Path.glob(DATAPATH, 'Cali*d*.fit')))

Flatlist = sorted(list(Path.glob(DATAPATH, 'Flat*.fit')))

Complist = sorted(list(Path.glob(DATAPATH, 'Arc*af.fit')))

NGC7331list = sorted(list(Path.glob(DATAPATH, 'NGC7331*.fit')))
Jupiterlist_h = sorted(list(Path.glob(DATAPATH, 'jupiter_h*.fit')))
Jupiterlist_v = sorted(list(Path.glob(DATAPATH, 'jupiter_v*.fit')))
Saturnlist_h = sorted(list(Path.glob(DATAPATH, 'saturn_h*.fit')))
Saturnlist_v = sorted(list(Path.glob(DATAPATH, 'saturn_v*.fit')))
Titanlist = sorted(list(Path.glob(DATAPATH, 'titan*.fit')))
Uranuslist = sorted(list(Path.glob(DATAPATH, 'uranus*.fit')))
Neptunelist = sorted(list(Path.glob(DATAPATH, 'neptune*.fit')))
Objectlist = NGC7331list+Jupiterlist_h+Jupiterlist_v+Saturnlist_h+Saturnlist_v+Titanlist+Uranuslist+Neptunelist

Stand8634list = sorted(list(Path.glob(DATAPATH, 'HR8634*.fit'))) #NGC7331
Stand9087list = sorted(list(Path.glob(DATAPATH, 'HR9087*.fit'))) #Jupiter, Saturn, Titan
Stand718list = sorted(list(Path.glob(DATAPATH, 'HR718*.fit'))) #Uranus
Stand7596list = sorted(list(Path.glob(DATAPATH, 'HR7596*.fit'))) #Neptune
Standlist = Stand8634list+Stand9087list+Stand718list+Stand7596list
Pointlist = Titanlist+Uranuslist+Neptunelist+Standlist

# - Bring Gain 
gain = 1.5 #e/ADU
Name = []
RN = []
for i in range(len(Biaslist)-1):
    hdul1 = fits.open(Biaslist[i])
    bias1 = hdul1[0].data
    bias1 = np.array(bias1).astype('float64')
    hdul2 = fits.open(Biaslist[i+1])
    bias2 = hdul2[0].data
    bias2 = np.array(bias2).astype('float64')
    dbias = bias2 - bias1
    RN.append(np.std(dbias)*gain / np.sqrt(2))
print('\nmean RN:', np.mean(RN))    
RN = np.mean(RN)

# Bring the sample image

OBJECTNAME = DATAPATH/'pjupiter_h-0005.fits'
hdul = fits.open(OBJECTNAME)[0]
obj = hdul.data
header = hdul.header
EXPTIME = header['EXPTIME']
fig,ax = plt.subplots(1,1,figsize=(10,15))
ax.imshow(obj,vmin=0,vmax=6000)
ax.set_title('EXPTIME = {0}s'.format(EXPTIME))
ax.set_xlabel('Dispersion axis [pixel]')
ax.set_ylabel('Spatial axis \n [pixel]')

# Let's find the peak along the spatial axis

# Plot the spectrum along the spatial direction
lower_cut = 700
upper_cut = 750
apall_1 = np.sum(obj[:,lower_cut:upper_cut],axis=1)

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(apall_1)
ax.set_xlabel('Pixel number (Spatial axis)')
ax.set_ylabel('Pixel value')
ax.grid(ls=':')

# Find the peak

peak_pix = peak_local_max(apall_1, num_peaks=1,
                          min_distance=30,
                          threshold_abs=np.median(apall_1))
peak_pix = np.array(peak_pix[:,0])
print(peak_pix)

fig,ax = plt.subplots(1,1,figsize=(6,6))
x = np.arange(0, len(apall_1))
ax.plot(x, apall_1)
ax.set_xlabel('Spatial axis')

peak_value = []
for i in peak_pix:
    ax.plot((i, i), 
            (apall_1[i]+0.01*max(apall_1),apall_1[i]+0.1*max(apall_1)),
            color='r', ls='-', lw=1)
    ax.annotate(i, (i, apall_1[i]+0.1*max(apall_1)),
            fontsize='small', rotation=45)
    peak_value.append(apall_1[i])
    print(i)

peak_pix = np.sort(peak_pix)
    
ax.grid(ls=':')
ax.set_xlabel('Pixel number (Spatial axis)')
ax.set_ylabel('Pixel value')

plt.show()

print(f'Pixel coordinatate in spatial direction = {peak_pix}')

# Select the sky area
peak = peak_pix[0] # center
print('Peak pixel is {0} pix'.format(peak_pix[0]))

SKYLIMIT = 35 # pixel limit around the peak
RLIMIT = 30 # pixel limit from the rightmost area
LLIMIT = 20 # pixel limit from the leftmost area

mask_sky = np.full_like(x, True, dtype=bool)
for p in peak_pix:
    mask_sky[(x > p-SKYLIMIT) & (x < p+SKYLIMIT)] = False    
mask_sky[:LLIMIT] = False
mask_sky[-RLIMIT:] = False
    
x_sky = x[mask_sky]
sky_val = apall_1[mask_sky]

# Plot the sky area
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(x, apall_1, lw=1)
ax.fill_between(x, 0, 1, where=mask_sky, alpha=0.1, transform=ax.get_xaxis_transform())

ax.set_xlabel('Pixel number')
ax.set_ylabel('Pixel value sum')

# Sigma clipping
Sigma = 3
clip_mask = sigma_clip(sky_val,
                       sigma=Sigma,
                       maxiters= 5).mask

# Fit the sky
ORDER_APSKY = 3
coeff_apsky, fitfull = chebfit(x_sky[~clip_mask], 
                               sky_val[~clip_mask],
                               deg=ORDER_APSKY,
                               full=True)
sky_fit = chebval(x, coeff_apsky) 

# Calculate the RMS of Fit
residual = fitfull[0][0] 
fitRMS = np.sqrt(residual/len(x_sky[~clip_mask]))


n_sky = len(x_sky)
n_rej = np.count_nonzero(clip_mask)

# Plot the sky area & fitted sky
fig,ax = plt.subplots(1,1,figsize=(8, 5))

ax.plot(x, apall_1, lw=1)
ax.plot(x, sky_fit, ls='--',
        label='Sky Fit ({:d}/{:d} used)'.format(n_sky - n_rej, n_sky))
ax.plot(x_sky[clip_mask], sky_val[clip_mask], marker='x', ls='', ms=10)
ax.fill_between(x, 0, 1, where=mask_sky, alpha=0.1, transform=ax.get_xaxis_transform())


title_str = r'Skyfit: {:s} order {:d} ({:.1f}-sigma {:d}-iters)'
ax.set_xlabel('Pixel number')
ax.set_ylabel('Pixel value sum')
ax.legend()
ax.set_title(title_str.format('Chebyshev', ORDER_APSKY,
                              Sigma, 5))

# Finding the peak center by fitting the gaussian 1D function again

sub_apall_1 = apall_1 - sky_fit # profile - sky

# OUTPIX = 50 # number of pixels to rule out outermost area in spacial direction
# xx = x[OUTPIX:-OUTPIX]
# yy = sub_apall_1[OUTPIX:-OUTPIX]

xx = x[peak-SKYLIMIT:peak+SKYLIMIT]
yy = sub_apall_1[peak-SKYLIMIT:peak+SKYLIMIT]

# gaussian modeling
g_init = Gaussian1D(amplitude=sub_apall_1[peak], 
                    mean=peak,
                    stddev=15*gaussian_fwhm_to_sigma)

fitter = LevMarLSQFitter()
fitted = fitter(g_init, xx, yy)

params = [fitted.amplitude.value, fitted.mean.value, fitted.stddev.value]

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(0, len(apall_1))
ax.plot(x, sub_apall_1, label='Spatial profile')

ax.fill_between(x, 0, 1, where=mask_sky, alpha=0.1, transform=ax.get_xaxis_transform())

g = Gaussian1D(*params)

ax.plot(x, g(x), label='Fitted 1D Gaussian profile')
ax.grid(ls=':')
ax.set_xlabel('Pixel number (Spatial axis)')
ax.set_ylabel('Pixel value')
ax.set_ylim(0, sub_apall_1[peak])
ax.legend(loc='upper right',fontsize=10)
plt.show()

center_pix = params[1] 
print('center pixel:', center_pix)

# Trace the aperture (peak) along the wavelength.
# Repeat the above process for all wavelength bands.
# This process is called "aperture tracing".
aptrace = []
aptrace_fwhm = []
STEP_AP = 10  
N_AP = len(obj[0])//STEP_AP
FWHM_AP = 10
peak = center_pix
for i in range(N_AP - 1):
    lower_cut, upper_cut = i*STEP_AP, (i+1)*STEP_AP
    apall_i = np.sum(obj[:, lower_cut:upper_cut], axis=1)
    sky_val = apall_i[mask_sky]
    clip_mask = sigma_clip(sky_val,
                           sigma=3,
                           maxiters=5).mask
    coeff, fitfull = chebfit(x_sky[~clip_mask], 
                             sky_val[~clip_mask],
                             deg=ORDER_APSKY,
                             full=True)
    apall_i -= chebval(x,coeff)  # Object profile - the fitted sky
    
    search_min = int(peak - 3*FWHM_AP)
    search_max = int(peak + 3*FWHM_AP)
    cropped = apall_i[search_min:search_max]
    x_cropped = np.arange(len(cropped))
    
    peak_pix_trace = peak_local_max(cropped,
                              min_distance=FWHM_AP,
                              num_peaks=1)
    peak_pix_trace = np.array(peak_pix_trace[:,0])
 
    if len(peak_pix_trace) == 0: # return NaN (Not a Number) if there is no peak found. 
        aptrace.append(np.nan)
        aptrace_fwhm.append(0)
    else:
        peak_pix_trace = np.sort(peak_pix_trace)
        peak_trace = peak_pix_trace[0]   
        g_init = Gaussian1D(amplitude=cropped[peak_trace], # Gaussian fitting to find centers
                            mean=peak_trace,
                            stddev=FWHM_AP*gaussian_fwhm_to_sigma,
                            bounds={'amplitude':(0, 2*cropped[peak_trace]) ,
                                    'mean':(peak_trace-3*FWHM_AP, peak_trace+3*FWHM_AP),
                                    'stddev':(0.00001, FWHM_AP*gaussian_fwhm_to_sigma)})
        fitted = fitter(g_init, x_cropped, cropped)
        center_pix_new = fitted.mean.value + search_min
        aptrace_fwhm.append(fitted.fwhm)
        aptrace.append(center_pix_new)  
        
aptrace = np.array(aptrace)
aptrace_fwhm = np.array(aptrace_fwhm)     

# Plot the center of profile peak
fig,ax = plt.subplots(2,1,figsize=(10,5))
ax[0].imshow(obj,vmin=0,vmax=300)
ax[0].set_xlabel('Dispersion axis',fontsize=15)
ax[0].set_ylabel('Spatial axis',fontsize=15)
xlim = ax[0].get_xlim()

ax[1].plot(np.arange(len(aptrace))*10, aptrace,ls='', marker='+', ms=10,color='lightskyblue')
ax[1].set_xlim(xlim)
ax[1].set_ylim(200,0) # you can see the jiggly-wiggly shape when you zoom in
ax[1].set_xlabel('Dispersion axis',fontsize=15)
ax[1].set_ylabel('Spatial axis',fontsize=15)

# Fitting the peak with Chebyshev function

ORDER_APTRACE = 9
SIGMA_APTRACE = 3
ITERS_APTRACE = 5 # when sigma clipping

# Fitting the line
x_aptrace = np.arange(N_AP-1) * STEP_AP
coeff_aptrace = chebfit(x_aptrace, aptrace, deg=ORDER_APTRACE)

# Sigma clipping
resid_mask = sigma_clip(aptrace - chebval(x_aptrace, coeff_aptrace), 
                        sigma=SIGMA_APTRACE, maxiters=ITERS_APTRACE).mask

# Fitting the peak again after sigma clipping
x_aptrace_fin = x_aptrace[~resid_mask]
aptrace_fin = aptrace[~resid_mask]
coeff_aptrace_fin = chebfit(x_aptrace_fin, aptrace_fin, deg=ORDER_APTRACE)   

fit_aptrace_fin   = chebval(x_aptrace_fin, coeff_aptrace_fin)
resid_aptrace_fin = aptrace_fin - fit_aptrace_fin
del_aptrace = ~np.in1d(x_aptrace, x_aptrace_fin) # deleted points #x_aptrace에서 x_aptrace_fin이 없으면 True

# Plot the Fitted line & residual
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(3, 1)
ax1 = plt.subplot(gs[0:2])
ax2 = plt.subplot(gs[2])

ax1.plot(x_aptrace, aptrace, ls='', marker='+', ms=10,color='lightskyblue')
ax1.plot(x_aptrace_fin, fit_aptrace_fin, ls='--',color='crimson',zorder=10,lw=2,
         label="Aperture Trace ({:d}/{:d} used)".format(len(aptrace_fin), N_AP-1))
ax1.plot(x_aptrace[del_aptrace], aptrace[del_aptrace], ls='', marker='x',color='salmon', ms=10)
ax1.set_ylabel('Found object position')
ax1.grid(ls=':')
ax1.legend()


ax2.plot(x_aptrace_fin, resid_aptrace_fin, ls='', marker='+')
ax2.axhline(+np.std(resid_aptrace_fin, ddof=1), ls=':', color='k')
ax2.axhline(-np.std(resid_aptrace_fin, ddof=1), ls=':', color='k', 
            label='residual std')


ax2.set_ylabel('Residual (pixel)')
ax2.set_xlabel('Dispersion axis (pixel)')

ax2.grid(ls=':')
ax2.set_ylim(-3, 3)
ax2.legend()

#Set plot title
title_str = ('Aperture Trace Fit ({:s} order {:d})\n'
            + 'Residuials {:.1f}-sigma, {:d}-iters clipped')
plt.suptitle(title_str.format('Chebshev', ORDER_APTRACE,
                              SIGMA_APTRACE, ITERS_APTRACE))
plt.show()

# Aperture sum

apsum_sigma_lower = 3 # [Sigma]
apsum_sigma_upper = 3
ap_fwhm = np.median(aptrace_fwhm[~resid_mask]) # [pix]
ap_sigma = ap_fwhm * gaussian_fwhm_to_sigma # [pixel/sigma]
x_ap = np.arange(len(obj[0])) # pixel along the dispersion axis
y_ap = chebval(x_ap, coeff_aptrace_fin) # center of peak for each line

# Extract the spectrum along the dispersion axis
ap_summed  = []
ap_sig = []

for i in range(len(obj[0])):
    cut_i = obj[:,i] # Cut spatial direction
    peak_i = y_ap[i]
    offset_i = peak_i - peak
    
    mask_sky_i = np.full_like(x, True, dtype=bool)
    for p in peak_pix:
        mask_sky_i[(x > p+offset_i-SKYLIMIT) & (x < p+offset_i+SKYLIMIT)] = False    
    mask_sky_i[:LLIMIT] = False
    mask_sky_i[-RLIMIT:] = False
 
    # aperture size = apsum_sigma_lower * ap_sigma
    x_obj_lower = int(np.around(peak_i - apsum_sigma_lower * ap_sigma)) 
    x_obj_upper = int(np.around(peak_i + apsum_sigma_upper * ap_sigma))         
    x_obj = np.arange(x_obj_lower, x_obj_upper)
    obj_i = cut_i[x_obj_lower:x_obj_upper]

    # Fitting Sky value
    x_sky = x[mask_sky_i]
    sky_val = cut_i[mask_sky_i]
    clip_mask = sigma_clip(sky_val, sigma=Sigma,
                           maxiters=5).mask
    coeff = chebfit(x_sky[~clip_mask],
                    sky_val[~clip_mask],
                    deg=ORDER_APSKY)

    # Subtract the sky
    sub_obj_i = obj_i - chebval(x_obj, coeff) # obj - lsky  subtraction

    
    # Calculate error
    sig_i = RN **2 + sub_obj_i + chebval(x_obj,coeff)
    # RN**2 + flux_i + sky value 
    
    ap_summed.append(np.sum(sub_obj_i)) 
    ap_sig.append(np.sqrt(np.sum(sig_i)))
    
ap_summed = np.array(ap_summed) / EXPTIME    
ap_std = np.array(ap_sig) / EXPTIME    

# Plot the spectrum 

x_pix = np.arange(len(obj[0]))

fig,ax = plt.subplots(1,1,figsize=(15,10))
ax.plot(x_pix,ap_summed,color='k',alpha=1)


FILENAME = OBJECTNAME.name
ax.set_title(FILENAME,fontsize=20)
ax.set_ylabel('Instrument Intensity\n(apsum/EXPTIME)',fontsize=20)
ax.set_xlabel(r'Dispersion axis [Pixel]',fontsize=20) 
ax.tick_params(labelsize=15)
plt.show()

spec_before_wfcali = Table([x_pix, ap_summed, ap_std],
                          names=['x_pix', 'ap_summed', 'ap_std'])


spec_before_wfcali.write(DATAPATH/(OBJECTNAME.stem+'_inst_spec.csv'),
                        overwrite=True, format='csv')

# Bring the sample image

OBJECTNAME = DATAPATH/'pHR9087-0005.fits'
hdul = fits.open(OBJECTNAME)[0]
obj = hdul.data
header = hdul.header
EXPTIME = header['EXPTIME']
fig,ax = plt.subplots(1,1,figsize=(10,15))
ax.imshow(obj,vmin=0,vmax=300)
ax.set_title('EXPTIME = {0}s'.format(EXPTIME))
ax.set_xlabel('Dispersion axis [pixel]')
ax.set_ylabel('Spatial axis \n [pixel]')

# Let's find the peak along the spatial axis

# Plot the spectrum along the spatial direction
lower_cut = 700
upper_cut = 750
apall_1 = np.sum(obj[:,lower_cut:upper_cut],axis=1)

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(apall_1)
ax.set_xlabel('Pixel number (Spatial axis)')
ax.set_ylabel('Pixel value')
ax.grid(ls=':')

# Find the peak

peak_pix = peak_local_max(apall_1, num_peaks=10,
                          min_distance=30,
                          threshold_abs=np.median(apall_1))
peak_pix = np.array(peak_pix[:,0])
print(peak_pix)

fig,ax = plt.subplots(1,1,figsize=(6,6))
x = np.arange(0, len(apall_1))
ax.plot(x, apall_1)
ax.set_xlabel('Spatial axis')

peak_value = []
for i in peak_pix:
    ax.plot((i, i), 
            (apall_1[i]+0.01*max(apall_1),apall_1[i]+0.1*max(apall_1)),
            color='r', ls='-', lw=1)
    ax.annotate(i, (i, apall_1[i]+0.1*max(apall_1)),
            fontsize='small', rotation=45)
    peak_value.append(apall_1[i])
    print(i)

peak_pix = np.sort(peak_pix)
    
ax.grid(ls=':')
ax.set_xlabel('Pixel number (Spatial axis)')
ax.set_ylabel('Pixel value')

plt.show()

print(f'Pixel coordinatate in spatial direction = {peak_pix}')

# Select the sky area
peak = peak_pix[0] # center
print('Peak pixel is {0} pix'.format(peak_pix[0]))

SKYLIMIT = 25 # pixel limit around the peak
RLIMIT = 30 # pixel limit from the rightmost area
LLIMIT = 20 # pixel limit from the leftmost area

mask_sky = np.full_like(x, True, dtype=bool)
for p in peak_pix:
    mask_sky[(x > p-SKYLIMIT) & (x < p+SKYLIMIT)] = False    
mask_sky[:LLIMIT] = False
mask_sky[-RLIMIT:] = False
    
x_sky = x[mask_sky]
sky_val = apall_1[mask_sky]

# Plot the sky area
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(x, apall_1, lw=1)
ax.fill_between(x, 0, 1, where=mask_sky, alpha=0.1, transform=ax.get_xaxis_transform())

ax.set_xlabel('Pixel number')
ax.set_ylabel('Pixel value sum')

# Sigma clipping
Sigma = 3
clip_mask = sigma_clip(sky_val,
                       sigma=Sigma,
                       maxiters= 5).mask

# Fit the sky
ORDER_APSKY = 3
coeff_apsky, fitfull = chebfit(x_sky[~clip_mask], 
                               sky_val[~clip_mask],
                               deg=ORDER_APSKY,
                               full=True)
sky_fit = chebval(x, coeff_apsky) 

# Calculate the RMS of Fit
residual = fitfull[0][0] 
fitRMS = np.sqrt(residual/len(x_sky[~clip_mask]))


n_sky = len(x_sky)
n_rej = np.count_nonzero(clip_mask)

# Plot the sky area & fitted sky
fig,ax = plt.subplots(1,1,figsize=(8, 5))

ax.plot(x, apall_1, lw=1)
ax.plot(x, sky_fit, ls='--',
        label='Sky Fit ({:d}/{:d} used)'.format(n_sky - n_rej, n_sky))
ax.plot(x_sky[clip_mask], sky_val[clip_mask], marker='x', ls='', ms=10)
ax.fill_between(x, 0, 1, where=mask_sky, alpha=0.1, transform=ax.get_xaxis_transform())


title_str = r'Skyfit: {:s} order {:d} ({:.1f}-sigma {:d}-iters)'
ax.set_xlabel('Pixel number')
ax.set_ylabel('Pixel value sum')
ax.legend()
ax.set_title(title_str.format('Chebyshev', ORDER_APSKY,
                              Sigma, 5))

# Finding the peak center by fitting the gaussian 1D function again

sub_apall_1 = apall_1 - sky_fit # profile - sky

# OUTPIX = 50 # number of pixels to rule out outermost area in spacial direction
# xx = x[OUTPIX:-OUTPIX]
# yy = sub_apall_1[OUTPIX:-OUTPIX]

xx = x[peak-SKYLIMIT:peak+SKYLIMIT]
yy = sub_apall_1[peak-SKYLIMIT:peak+SKYLIMIT]

# gaussian modeling
g_init = Gaussian1D(amplitude=sub_apall_1[peak], 
                    mean=peak,
                    stddev=15*gaussian_fwhm_to_sigma)

fitter = LevMarLSQFitter()
fitted = fitter(g_init, xx, yy)

params = [fitted.amplitude.value, fitted.mean.value, fitted.stddev.value]

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(0, len(apall_1))
ax.plot(x, sub_apall_1, label='Spatial profile')

g = Gaussian1D(*params)

ax.plot(x, g(x), label='Fitted 1D Gaussian profile')
ax.grid(ls=':')
ax.set_xlabel('Pixel number (Spatial axis)')
ax.set_ylabel('Pixel value')
ax.set_ylim(0, sub_apall_1[peak])
ax.legend(loc=2,fontsize=10)
plt.show()

center_pix = params[1] 
print('center pixel:', center_pix)

# Trace the aperture (peak) along the wavelength.
# Repeat the above process for all wavelength bands.
# This process is called "aperture tracing".
aptrace = []
aptrace_fwhm = []
STEP_AP = 10  
N_AP = len(obj[0])//STEP_AP
FWHM_AP = 10
peak = center_pix
for i in range(N_AP - 1):
    lower_cut, upper_cut = i*STEP_AP, (i+1)*STEP_AP
    apall_i = np.sum(obj[:, lower_cut:upper_cut], axis=1)
    sky_val = apall_i[mask_sky]
    clip_mask = sigma_clip(sky_val,
                           sigma=3,
                           maxiters=5).mask
    coeff, fitfull = chebfit(x_sky[~clip_mask], 
                             sky_val[~clip_mask],
                             deg=ORDER_APSKY,
                             full=True)
    apall_i -= chebval(x,coeff)  # Object profile - the fitted sky
    
    search_min = int(peak - 3*FWHM_AP)
    search_max = int(peak + 3*FWHM_AP)
    cropped = apall_i[search_min:search_max]
    x_cropped = np.arange(len(cropped))
    
    peak_pix_trace = peak_local_max(cropped,
                              min_distance=FWHM_AP,
                              num_peaks=1)
    peak_pix_trace = np.array(peak_pix_trace[:,0])
 
    if len(peak_pix_trace) == 0: # return NaN (Not a Number) if there is no peak found. 
        aptrace.append(np.nan)
        aptrace_fwhm.append(0)
    else:
        peak_pix_trace = np.sort(peak_pix_trace)
        peak_trace = peak_pix_trace[0]   
        g_init = Gaussian1D(amplitude=cropped[peak_trace], # Gaussian fitting to find centers
                            mean=peak_trace,
                            stddev=FWHM_AP*gaussian_fwhm_to_sigma,
                            bounds={'amplitude':(0, 2*cropped[peak_trace]) ,
                                    'mean':(peak_trace-3*FWHM_AP, peak_trace+3*FWHM_AP),
                                    'stddev':(0.00001, FWHM_AP*gaussian_fwhm_to_sigma)})
        fitted = fitter(g_init, x_cropped, cropped)
        center_pix_new = fitted.mean.value + search_min
        aptrace_fwhm.append(fitted.fwhm)
        aptrace.append(center_pix_new)  
        
aptrace = np.array(aptrace)
aptrace_fwhm = np.array(aptrace_fwhm)   

# Plot the center of profile peak
fig,ax = plt.subplots(2,1,figsize=(10,5))
ax[0].imshow(obj,vmin=0,vmax=300)
ax[0].set_xlabel('Dispersion axis',fontsize=15)
ax[0].set_ylabel('Spatial axis',fontsize=15)
xlim = ax[0].get_xlim()

ax[1].plot(np.arange(len(aptrace))*10, aptrace,ls='', marker='+', ms=10,color='lightskyblue')
ax[1].set_xlim(xlim)
ax[1].set_ylim(200,0) # you can see the jiggly-wiggly shape when you zoom in
ax[1].set_xlabel('Dispersion axis',fontsize=15)
ax[1].set_ylabel('Spatial axis',fontsize=15)

# Fitting the peak with Chebyshev function

ORDER_APTRACE = 9
SIGMA_APTRACE = 3
ITERS_APTRACE = 5 # when sigma clipping

# Fitting the line
x_aptrace = np.arange(N_AP-1) * STEP_AP
coeff_aptrace = chebfit(x_aptrace, aptrace, deg=ORDER_APTRACE)

# Sigma clipping
resid_mask = sigma_clip(aptrace - chebval(x_aptrace, coeff_aptrace), 
                        sigma=SIGMA_APTRACE, maxiters=ITERS_APTRACE).mask

# Fitting the peak again after sigma clipping
x_aptrace_fin = x_aptrace[~resid_mask]
aptrace_fin = aptrace[~resid_mask]
coeff_aptrace_fin = chebfit(x_aptrace_fin, aptrace_fin, deg=ORDER_APTRACE)   

fit_aptrace_fin   = chebval(x_aptrace_fin, coeff_aptrace_fin)
resid_aptrace_fin = aptrace_fin - fit_aptrace_fin
del_aptrace = ~np.in1d(x_aptrace, x_aptrace_fin) # deleted points #x_aptrace에서 x_aptrace_fin이 없으면 True
'''
test = np.array([0, 1, 2, 5, 0])
states = [0, 2]
mask = np.in1d(test, states)
mask
array([ True, False,  True, False,  True])
'''


# Plot the Fitted line & residual
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(3, 1)
ax1 = plt.subplot(gs[0:2])
ax2 = plt.subplot(gs[2])

ax1.plot(x_aptrace, aptrace, ls='', marker='+', ms=10,color='lightskyblue')
ax1.plot(x_aptrace_fin, fit_aptrace_fin, ls='--',color='crimson',zorder=10,lw=2,
         label="Aperture Trace ({:d}/{:d} used)".format(len(aptrace_fin), N_AP-1))
ax1.plot(x_aptrace[del_aptrace], aptrace[del_aptrace], ls='', marker='x',color='salmon', ms=10)
ax1.set_ylabel('Found object position')
ax1.grid(ls=':')
ax1.legend()


ax2.plot(x_aptrace_fin, resid_aptrace_fin, ls='', marker='+')
ax2.axhline(+np.std(resid_aptrace_fin, ddof=1), ls=':', color='k')
ax2.axhline(-np.std(resid_aptrace_fin, ddof=1), ls=':', color='k', 
            label='residual std')


ax2.set_ylabel('Residual (pixel)')
ax2.set_xlabel('Dispersion axis (pixel)')

ax2.grid(ls=':')
ax2.set_ylim(-3, 3)
ax2.legend()

#Set plot title
title_str = ('Aperture Trace Fit ({:s} order {:d})\n'
            + 'Residuials {:.1f}-sigma, {:d}-iters clipped')
plt.suptitle(title_str.format('Chebshev', ORDER_APTRACE,
                              SIGMA_APTRACE, ITERS_APTRACE))
plt.show()

# Aperture sum

apsum_sigma_lower = 3 # [Sigma]
apsum_sigma_upper = 3
ap_fwhm = np.median(aptrace_fwhm[~resid_mask]) # [pix]
ap_sigma = ap_fwhm * gaussian_fwhm_to_sigma # [pixel/sigma]
x_ap = np.arange(len(obj[0])) # pixel along the dispersion axis
y_ap = chebval(x_ap, coeff_aptrace_fin) # center of peak for each line

# Extract the spectrum along the dispersion axis
ap_summed  = []
ap_sig = []

for i in range(len(obj[0])):
    cut_i = obj[:,i] # Cut spatial direction
    peak_i = y_ap[i]
    offset_i = peak_i - peak
    
    mask_sky_i = np.full_like(x, True, dtype=bool)
    for p in peak_pix:
        mask_sky_i[(x > p+offset_i-SKYLIMIT) & (x < p+offset_i+SKYLIMIT)] = False    
    mask_sky_i[:LLIMIT] = False
    mask_sky_i[-RLIMIT:] = False
 
    # aperture size = apsum_sigma_lower * ap_sigma
    x_obj_lower = int(np.around(peak_i - apsum_sigma_lower * ap_sigma)) 
    x_obj_upper = int(np.around(peak_i + apsum_sigma_upper * ap_sigma))         
    x_obj = np.arange(x_obj_lower, x_obj_upper)
    obj_i = cut_i[x_obj_lower:x_obj_upper]

    # Fitting Sky value
    x_sky = x[mask_sky_i]
    sky_val = cut_i[mask_sky_i]
    clip_mask = sigma_clip(sky_val, sigma=Sigma,
                           maxiters=5).mask
    coeff = chebfit(x_sky[~clip_mask],
                    sky_val[~clip_mask],
                    deg=ORDER_APSKY)

    # Subtract the sky
    sub_obj_i = obj_i - chebval(x_obj, coeff) # obj - lsky  subtraction

    
    # Calculate error
    sig_i = RN **2 + sub_obj_i + chebval(x_obj,coeff)
    # RN**2 + flux_i + sky value 
    
    ap_summed.append(np.sum(sub_obj_i)) 
    ap_sig.append(np.sqrt(np.sum(sig_i)))
    
ap_summed = np.array(ap_summed) / EXPTIME    
ap_std = np.array(ap_sig) / EXPTIME    

# Plot the spectrum 

x_pix = np.arange(len(obj[0]))

fig,ax = plt.subplots(1,1,figsize=(15,10))
ax.plot(x_pix,ap_summed,color='k',alpha=1)


FILENAME = OBJECTNAME.name
ax.set_title(FILENAME,fontsize=20)
ax.set_ylabel('Instrument Intensity\n(apsum/EXPTIME)',fontsize=20)
ax.set_xlabel(r'Dispersion axis [Pixel]',fontsize=20) 
ax.tick_params(labelsize=15)
plt.show()

spec_before_wfcali = Table([x_pix, ap_summed, ap_std],
                          names=['x_pix', 'ap_summed', 'ap_std'])


spec_before_wfcali.write(DATAPATH/(OBJECTNAME.stem+'_inst_spec.csv'),
                        overwrite=True, format='csv')

# Bring the Master Comparison image
Master_comparison = DATAPATH/'pMaster_Comp.fits'
compimage = fits.open(Master_comparison)[0].data
identify = np.median(compimage[90:110,:],
                         axis=0)
fig,ax = fig,ax = plt.subplots(1,1,figsize=(15,5))
ax.imshow(compimage, vmin=0, vmax=1000)

fig,ax = plt.subplots(1,1,figsize=(15,5))
x_identify = np.arange(0,len(identify))
ax.plot(x_identify,identify,lw=1)
ax.set_xlabel('Piexl number')
ax.set_ylabel('Pixel value sum')
ax.set_xlim(0,len(identify))
plt.show()
plt.tight_layout()

# Find the local peak

peak_pix = peak_local_max(identify,
                          num_peaks=max(identify),
                          min_distance=4,
                          threshold_abs=max(identify)*0.001)

fig,ax = plt.subplots(1, 1, figsize=(10, 5))
x_identify = np.arange(0, len(identify))
ax.plot(x_identify, identify, lw=1)
for i in peak_pix:
    ax.plot([i, i],
            [identify[i]+0.01*max(identify), identify[i]+0.06*max(identify)],
            color='r',lw=1)
    ax.annotate(i[0], (i, identify[i]+0.06*max(identify)),
                fontsize=8,
                rotation=80)

ax.set_xlabel('Piexl number')
ax.set_ylabel('Pixel value sum')
ax.set_xlim(min(peak_pix)[0] - 50,max(peak_pix)[0] + 50)
ax.set_ylim(-2000, max(identify) + max(identify)*0.2)

axins = ax.inset_axes([0.07, 0.35, 0.7, 0.6])
axins.plot(x_identify, identify, lw=1)
axins.set_xlim(100, 800)
axins.set_ylim(-100, 4000)
for i in peak_pix:
    axins.plot([i, i],
            [identify[i]+0.003*max(identify), identify[i]+0.01*max(identify)],
            color='r', lw=1)
    axins.annotate(i[0],(i,identify[i]+0.01*max(identify)),
                fontsize=8,
                rotation=80)

plt.show()
plt.tight_layout()

# # Find the matching pair of wavelength and pixel
# pixel_init, wavelength = np.array([
#                                     [171,  4200.674],
#                                     [276,  4657.901],
#                                     [301,  4764.865],
#                                     # [739,  6677.282],
#                                     [550,  5852.488], # Ne
#                                     [571,  5944.834], # Ne
#                                     [592,  6029.997], # Ne
#                                     [617,  6143.063], # Ne
#                                     [645,  6266.495], # Ne
#                                     [676,  6402.246], # Ne
#                                     [700,  6506.528], # Ne
#                                     [739,  6678.276], # Ne
#                                     [756,  6752.834],
#                                     [804,  6965.431],
#                                     [827,  7067.218],
#                                     [874,  7272.936],
#                                     [898,  7383.980],
#                                     [926,  7503.869],
#                                     [955,  7635.106],
#                                     [975,  7724.207],
#                                     [1025, 7948.176],
#                                     [1039, 8006.157],
#                                     [1062, 8103.693],
#                                    ]).T

# ID_init = dict(pixel_init=pixel_init.astype('int'), wavelength=wavelength)

ID_init = dict(pixel_init = [160,215,334,814, 838, 884, 909, 936, 966, 986, 1036, 1051, 1073],
               wavelength = [4102,4341,4861,6965.4, 7067.2, 7273, 7384, 7503.8, 7635.1, 7724, 7948, 8006.1, 8103.6])

ID_init = Table(ID_init)
plt.plot(ID_init['pixel_init'],ID_init['wavelength'],marker='x',ls='')

def linear(x,a,b):
    return a*x + b
popt,pcov = curve_fit(linear,ID_init['pixel_init'],ID_init['wavelength'])
plt.plot(ID_init['pixel_init'],linear(ID_init['pixel_init'],*popt))
plt.xlabel('pixel')
plt.ylabel('wavelength')
print(linear(550,*popt))

# Fit the each peak with Gaussian 1D function

peak_gauss = []
fitter = LevMarLSQFitter()
LINE_FITTER = LevMarLSQFitter()
FWHM_ID = 3
x_identify = np.arange(0,len(identify))

# Gaussian fitting for each peak (pixel)
for peak_pix in ID_init['pixel_init']:
    g_init = Gaussian1D(amplitude=identify[peak_pix],
                        mean = peak_pix,
                        stddev = FWHM_ID*gaussian_fwhm_to_sigma,
                        bounds={'amplitude':(0,2*identify[peak_pix]),
                                'mean':(peak_pix - FWHM_ID,peak_pix + FWHM_ID),
                                'stddev':(0,FWHM_ID)})
    fitted = LINE_FITTER(g_init,x_identify,identify) #Model, x, y
    peak_gauss.append(fitted.mean.value)
    print(peak_pix,'->',fitted.mean.value)
    
    
peak_gauss = Column(data=peak_gauss,
                        name='pixel_gauss',
                        dtype=float)    
peak_shift = Column(data=peak_gauss - ID_init['pixel_init'],
                    name='piexl_shift',
                    dtype=float) 
ID_init['pixel_gauss'] = peak_gauss
ID_init['pixel_shift'] = peak_gauss - ID_init['pixel_init']
ID_init.sort('wavelength')
ID_init.pprint()

#Derive dispersion solution

ORDER_ID = 3 #Order of fitting function #보통 3 이하로 함. 기기의 사용파장대가 넓으면 4를 사용할때도 있음. 
coeff_ID, fitfull = chebfit(ID_init['pixel_gauss'],
                           ID_init['wavelength'],
                           deg=ORDER_ID,
                           full=True)  #Derive the dispersion solution

fitRMS = np.sqrt(fitfull[0][0]/len(ID_init))
rough_error = ((max(ID_init['wavelength'])-min(ID_init['wavelength']))
               /(max(ID_init['pixel_gauss'])-min(ID_init['pixel_gauss'])))/2
residual = (ID_init['wavelength'] #wavelength from reference
            -chebval(ID_init['pixel_gauss'],coeff_ID)) #wavelength derived from fitting
res_range = np.max(np.abs(residual))

fig,ax = plt.subplots(2,1,figsize=(10,5))
ax[0].plot(ID_init['pixel_gauss'],
         ID_init['wavelength'],
         ls = ':',marker='x')
ax[1].plot(ID_init['pixel_gauss'],
          residual,
          ls='',marker='+',
          color='k')
max_ID_init = max(ID_init['pixel_gauss'])
min_ID_init = min(ID_init['pixel_gauss'])
fig_xspan = max_ID_init - min_ID_init
fig_xlim = np.array([min_ID_init, max_ID_init]) + np.array([-1,1])*fig_xspan*0.1
ax[1].set_xlim(fig_xlim)
ax[0].set_xlim(fig_xlim)
ax[0].set_ylabel(r'Wavelength ($\AA$)')
ax[1].set_ylabel(r'Residual ($\AA$)')
ax[1].set_xlabel('Pixel along dispersion axis')
ax[0].set_title('First Identify (Chebyshev order {:d})\n'.format(ORDER_ID) 
              + r'RMSE = {:.2f} $\AA$'.format(fitRMS))
ax[1].grid()
plt.show()

fig,ax = fig,ax = plt.subplots(1,1,figsize=(15,5))
ax.imshow(compimage,vmin=0,vmax=1000)

# REIDENTIFY 

STEP_AP = 5 #Step size in pixel (dispersion direction)
STEP_REID = 10 #Step size in pixel (spatial direction)
N_SPATIAL,N_WAVELEN = np.shape(compimage) #(220, 2048)
N_REID = N_SPATIAL//STEP_REID #Spatial direction 
N_AP = N_WAVELEN//STEP_AP #Dispersion direction
TOL_REID = 5 # tolerence to lose a line in pixels

ORDER_WAVELEN_REID = 3 
ORDER_SPATIAL_REID = 3

line_REID = np.zeros((N_REID-1,len(ID_init))) #Make the empty array (height, width)
spatialcoord = np.arange(0,(N_REID-1)*STEP_REID,STEP_REID) + STEP_REID/2
# spatialcoord = array([  5.,  15.,  25.,  35.,  45.,  55.,  65.,  75.,  85.,  95., 105.,
#       115., 125., 135., 145., 155., 165., 175., 185., 195., 205.])




#Repeat we did above along the spatial direction
for i in range(0,N_REID-1): 
    lower_cut = i*STEP_REID
    upper_cut = (i+1)*STEP_REID
    reidentify_i = np.sum(compimage[lower_cut:upper_cut,:],axis=0)

    peak_gauss_REID = [] 

    for peak_pix_init in ID_init['pixel_gauss'][:3]:
        peak_gauss_REID.append(peak_pix_init)
    
    for peak_pix_init in ID_init['pixel_gauss'][3:]:
        search_min = int(np.around(peak_pix_init - TOL_REID))
        search_max = int(np.around(peak_pix_init + TOL_REID))
        cropped = reidentify_i[search_min:search_max]
        x_cropped = np.arange(len(cropped)) + search_min

        #Fitting the initial gauss peak by usijng Gausian1D
        Amplitude_init = np.max(cropped)
        mean_init = peak_pix_init
        stddev_init = 5*gaussian_fwhm_to_sigma
        g_init = Gaussian1D(amplitude = Amplitude_init,
                           mean = mean_init,
                           stddev = stddev_init,
                           bounds={'amplitude':(0, 2*np.max(cropped)) ,
                                 'stddev':(0, TOL_REID)})
        g_fit = fitter(g_init,x_cropped,cropped)
        fit_center = g_fit.mean.value    
        if abs(fit_center - peak_pix_init) > TOL_REID: #스펙트럼 끝에서는 잘 안 잡힐수있으니까
            peak_gauss_REID.append(np.nan)
            continue
        else:
            peak_gauss_REID.append(fit_center)
            
    peak_gauss_REID = np.array(peak_gauss_REID)  
    nonan_REID = np.isfinite(peak_gauss_REID)
    line_REID[i,:] = peak_gauss_REID  
    peak_gauss_REID_nonan = peak_gauss_REID[nonan_REID] 
    n_tot = len(peak_gauss_REID)
    n_found = np.count_nonzero(nonan_REID)
    coeff_REID1D, fitfull = chebfit(peak_gauss_REID_nonan,
                                    ID_init['wavelength'][nonan_REID], 
                                    deg=ORDER_WAVELEN_REID,
                                    full=True)
    fitRMS = np.sqrt(fitfull[0][0]/n_found)
    
points = np.vstack((line_REID.flatten(),
                    np.tile(spatialcoord, len(ID_init['pixel_init']))))
#np.tile(A,reps):Construct an array by repeating A the number of times given by reps.
# a = np.array([1, 2, 3])
# b = np.array([2, 3, 4])
# np.vstack((a,b)) = array([[1, 2, 3],[2, 3, 4]])
points = points.T # list of ()  
                   

values = np.tile(ID_init['wavelength'], N_REID - 1) #Wavelength corresponding to each point
values = np.array(values.tolist())  #
# errors = np.ones_like(values)


# #Fitting the wavelength along spatial direction and dispertion direction 
coeff_init = Chebyshev2D(x_degree=ORDER_WAVELEN_REID, y_degree=ORDER_SPATIAL_REID)
fit2D_REID = fitter(coeff_init, points[:, 0], points[:, 1], values) 
#Dispersion solution (both spatial & dispersion) #fitter(order,x,y,f(x,y))

#Plot 2D wavelength callibration map and #Points to used re-identify

fig,ax = plt.subplots(1,1,figsize=(10,4))
ww, ss = np.mgrid[:N_WAVELEN, :N_SPATIAL]
im = ax.imshow(fit2D_REID(ww, ss).T, origin='lower',vmin=4264.4,vmax=8108.99)
ax.plot(points[:, 0], points[:, 1], ls='', marker='+', color='r',
             alpha=0.8, ms=10)
fig.colorbar(im, ax=ax,orientation = 'horizontal')

ax.set_ylabel('Spatial \n direction')
ax.set_xlabel('Dispersion direction')
title_str = ('Reidentify and Wavelength Map\n'
+ 'func=Chebyshev, order (wavelength, dispersion) = ({:d}, {:d})')


plt.suptitle(title_str.format(ORDER_WAVELEN_REID, ORDER_SPATIAL_REID))

# Check how the dispersion solution change along the spatial axis
# Divide spectrum into 4 equal parts in the spatial direction

fig,ax = plt.subplots(2,1,figsize=(10,7))
ax[0].imshow(fit2D_REID(ww, ss).T, origin='lower')
ax[0].plot(points[:, 0], points[:, 1], ls='', marker='+', color='r',
             alpha=0.8, ms=10)
# ax[0].set_xlim(0,2000)
ax[0].set_ylabel('Spatial \n direction')

title_str = ('Reidentify and Wavelength Map\n'
+ 'func=Chebyshev, order (wavelength, dispersion) = ({:d}, {:d}) \n')
plt.suptitle(title_str.format(ORDER_WAVELEN_REID, ORDER_SPATIAL_REID))


for i in (1, 2, 3):
    vcut = N_WAVELEN * i/4
    hcut = N_SPATIAL * i/4
    vcutax  = np.arange(0, N_SPATIAL, STEP_REID) + STEP_REID/2 #Spatial dir coordinate
    hcutax  = np.arange(0, N_WAVELEN, 1) # pixel along dispersion axis
    
    vcutrep = np.repeat(vcut, len(vcutax)) #i/4에 해당하는 dispersion pixel * len(spatial)
    hcutrep = np.repeat(hcut, len(hcutax)) #i/4에 해당하는 spatial pixel * len(dispersion)

    ax[0].axvline(x=vcut, ls=':', color='k')   
    ax[0].axhline(y=hcut, ls=':', color='k')

    ax[1].plot(hcutax, fit2D_REID(hcutax, hcutrep), lw=1, 
             label="horizon cut {:d} pix".format(int(hcut)))
# ax[1].set_xlim(0,2000)

ax[1].grid(ls=':')
ax[1].legend()
ax[1].set_xlabel('Dispersion direction')
ax[1].set_ylabel('Wavelength \n(horizontal cut)')
ax[1].set_ylim(min(ID_init['wavelength'])*0.5,max(ID_init['wavelength'])*1.5)


plt.tight_layout()

fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[:1, :2])
ax2 = plt.subplot(gs[1:3, :2])
ax3 = plt.subplot(gs[:3, 2])

# title
title_str = ('Reidentify and Wavelength Map\n'
             + 'func=Chebyshev, order (wavelength, dispersion) = ({:d}, {:d})')
plt.suptitle(title_str.format(ORDER_WAVELEN_REID, ORDER_SPATIAL_REID))



interp_min = line_REID[~np.isnan(line_REID)].min()
interp_max = line_REID[~np.isnan(line_REID)].max()

ax1.imshow(fit2D_REID(ww, ss).T, origin='lower')
ax1.axvline(interp_max, color='r', lw=1)
ax1.axvline(interp_min, color='r', lw=1)
ax1.plot(points[:, 0], points[:, 1], ls='', marker='+', color='r',
             alpha=0.8, ms=10)
ax1.set_xlim(0,2000)
ax1.set_ylabel('Spatial \n direction')



for i in (1, 2, 3):
    vcut = N_WAVELEN * i/4
    hcut = N_SPATIAL * i/4
    vcutax  = np.arange(0, N_SPATIAL, STEP_REID) + STEP_REID/2
    hcutax  = np.arange(0, N_WAVELEN, 1)
    vcutrep = np.repeat(vcut, len(vcutax))
    hcutrep = np.repeat(hcut, len(hcutax))

    ax1.axvline(x=vcut, ls=':', color='k')   
    ax1.axhline(y=hcut, ls=':', color='k')

    ax2.plot(hcutax, fit2D_REID(hcutax, hcutrep), lw=1, 
             label="hcut {:d}".format(int(hcut)))

    
    vcut_profile = fit2D_REID(vcutrep, vcutax)
    vcut_normalize = vcut_profile / np.median(vcut_profile)
    
    ax3.plot(vcut_normalize, vcutax, lw=1,
             label="vcut {:d}".format(int(vcut)))



ax2.axvline(interp_max, color='r', lw=1,ls='--')
ax2.axvline(interp_min, color='r', lw=1,ls='--')    
    
ax1.set_ylabel('Spatial direction')
ax2.grid(ls=':')
ax2.legend(fontsize=15)
ax2.set_xlabel('Dispersion direction [pix]')
ax2.set_ylabel('Wavelength change\n(horizontal cut)')

ax3.axvline(1, ls=':', color='k')
ax3.grid(ls=':', which='both')
ax3.set_xlabel('Fractional change \n (vertical cut)')
ax3.legend(fontsize=10)

ax1.set_ylim(0, N_SPATIAL)
ax1.set_xlim(0, N_WAVELEN)
# ax2.set_xlim(300, 1700)
ax2.set_xlim(0, N_WAVELEN)
ax2.set_ylim(4000,9000)
ax3.set_ylim(0, N_SPATIAL)
plt.show()    

#Plot the spectrum respect to wavelength

Wavelength = chebval(np.arange(len(compimage[0])),coeff_ID)

x_pix = np.arange(len(obj[0]))


ObjStdList = Jupiterlist_h + Stand9087list
for i, path in enumerate(ObjStdList):
    fig, ax = plt.subplots(1,1,figsize=(10,3))
    OBJECTNAME = path.name
    FILENAME = path.stem
    obj = ascii.read(DATAPATH/('p'+FILENAME+'_inst_spec.csv'))
    ap_summed = obj['ap_summed']
    ap_std = obj['ap_std']
    ax.plot(Wavelength, ap_summed, color='k',alpha=1)
    ax.set_title(FILENAME, fontsize=20)
    ax.set_ylabel('Instrument Intensity\n(apsum/EXPTIME)', fontsize=20)
    ax.set_xlabel(r'Dispersion axis $[\AA]$', fontsize=20) 
    ax.tick_params(labelsize=15)
    # ax.set_xlim(4500, 8000)

    SAVE_FILENAME = DATAPATH/('p'+FILENAME+'_w_spec.csv')

    Data = [Wavelength, ap_summed, ap_std]
    data = Table(Data, names=['wave','inten','std'])
    data['wave'].format = "%.3f" 
    data['inten'].format = "%.3f" 
    data['std'].format = "%.3f" 

    ascii.write(data, SAVE_FILENAME, overwrite=True, format='csv')
    
plt.tight_layout()
'''
import combine_spec

Objectname = 'saturn_v'

obj1 = ascii.read(DATAPATH/('p'+Objectname+'-0001_w_spec.csv'))
obj2 = ascii.read(DATAPATH/('p'+Objectname+'-0002_w_spec.csv'))
obj3 = ascii.read(DATAPATH/('p'+Objectname+'-0003_w_spec.csv'))
obj4 = ascii.read(DATAPATH/('p'+Objectname+'-0004_w_spec.csv'))
obj5 = ascii.read(DATAPATH/('p'+Objectname+'-0005_w_spec.csv'))

obj_wave = np.stack((obj1['wave'],obj2['wave'],obj3['wave'],obj4['wave'],obj5['wave']))
obj_flux = np.stack((obj1['inten'],obj2['inten'],obj3['inten'],obj4['inten'],obj5['inten']))
obj_err = np.stack((obj1['std'],obj2['std'],obj3['std'],obj4['std'],obj5['std']))

combined_wave, combined_flux, combined_err = combine_spec.err_weighted_combine(obj_wave,obj_flux,obj_err)

spectrum = Table([combined_wave, combined_flux, combined_err],
                    names=['wave', 'inten', 'std'])
spectrum['wave'].format = "%.3f" 
spectrum['inten'].format = "%.3e" 
spectrum['std'].format = "%.3e"

fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.plot(combined_wave,combined_flux,c='k')
# ax.set_xlim(4000,8200)
# ax.set_ylim(0,5*10**(-11))

SPEC_SAVEPATH = DATAPATH/('c'+Objectname+'_w_spec.csv') 
spectrum.write(SPEC_SAVEPATH, overwrite=True, format='csv')
'''
###############################################################################

stdfile = DATAPATH/'fhr9087.txt'
stddata = ascii.read(stdfile)
std_wave, std_flux, std_wth  = stddata['col1'], stddata['col2']*1e-16, stddata['col4']

obj = ascii.read(DATAPATH/'cHR9087_w_spec.csv')
obj_wave = obj['wave']
obj_flux = obj['inten']

fig,ax = plt.subplots(1,2,figsize=(14,4))
ax[0].plot(obj_wave,obj_flux)
ax[0].set_xlim(4000,8000)
ax[0].set_title('Extracted spectrum')
ax[0].set_ylabel(r'counts')
ax[0].set_xlabel(r'Dispersion axis $[\AA]$',fontsize=20) 
ax[1].plot(std_wave,std_flux)
ax[1].set_xlim(4000,8000)
ax[1].set_ylim(0, 1e-10)
ax[1].set_title('Reference')
ax[1].set_ylabel(r'erg $s^{-1}cm^{-2}\AA^{-1}$ ')
ax[1].set_xlabel(r'Dispersion axis $[\AA]$',fontsize=20) 

balmer = np.array([6563, 4861, 4341, 4100, 6867, 7593.7], dtype='float')
for i in balmer:
    ax[0].axvline(i,color='coral',ls=':')
    ax[1].axvline(i,color='coral',ls=':')
 
 ##############################################################################   
 
obj_flux_ds = []
obj_wave_ds = []
std_flux_ds = []

for i in range(len(std_wave)):
    rng = np.where((obj_wave >= std_wave[i] - std_wth[i] / 2.0) &
                           (obj_wave < std_wave[i] + std_wth[i] / 2.0)) #STD-wave 범위 안에 들어가는 obj_wave

    IsH = np.where((balmer >= std_wave[i] - 15*std_wth[i] / 2.0) &
                           (balmer < std_wave[i] + 15*std_wth[i] / 2.0))
    
    if (len(rng[0]) > 1) and (len(IsH[0]) == 0): 
        # does this bin contain observed spectra, and no Balmer line?
        # obj_flux_ds.append(np.sum(obj_flux[rng]) / std_wth[i])
        obj_flux_ds.append( np.nanmean(obj_flux[rng]) )
        obj_wave_ds.append(std_wave[i])
        std_flux_ds.append(std_flux[i])
        
ratio = np.abs(np.array(std_flux_ds, dtype='float') /
                       np.array(obj_flux_ds, dtype='float'))
LogSensfunc = np.log10(ratio)


fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(obj_wave_ds,LogSensfunc,marker='x',ls='')
ax.set_xlim(4500,8000)
# ax.set_ylim(-16,-13)
ax.set_xlabel(r'Dispersion axis $[\AA]$',fontsize=20) 
ax.set_ylabel(r'$\log_{10} \left( \frac{reference}{Observed} \right)$',fontsize=20) 

###############################################################################

# interpolate back on to observed wavelength grid
spl = UnivariateSpline(obj_wave_ds, LogSensfunc, ext=0, k=2 ,s=0.0025)
sensfunc2 = spl(obj_wave)
# SF_CHEB_DEG = 5
# sf_coeff = chebfit(obj_wave_ds, LogSensfunc, deg=SF_CHEB_DEG)
# sensfunc2 = chebval(obj_wave, sf_coeff)

fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(obj_wave_ds,LogSensfunc,marker='x',ls='')
ax.plot(obj_wave,sensfunc2)
ax.set_xlim(4500,8000)
ax.set_xlabel(r'Dispersion axis $[\AA]$',fontsize=20) 
ax.set_ylabel(r'$\log_{10} \left( \frac{reference}{Observed} \right)$',fontsize=20) 

###############################################################################

sensfunc = 10**sensfunc2
obj_cal = obj_flux*sensfunc #flux after flux calibration

fig,ax = plt.subplots(1,2,figsize=(14,4))
ax[0].plot(obj_wave,obj_flux*sensfunc)
ax[0].set_xlim(4000,8000)
ax[0].set_title('After flux calibration')
ax[0].set_ylabel(r'counts')
ax[0].set_xlabel(r'Dispersion axis $[\AA]$',fontsize=20) 
ax[0].set_ylim(0,1.5e-10)
ax[1].plot(std_wave,std_flux)
ax[1].set_xlim(4000,8000)
ax[1].set_title('Reference')
ax[1].set_ylabel(r'erg $s^{-1}cm^{-2}\AA^{-1}$ ')
ax[1].set_xlabel(r'Dispersion axis $[\AA]$',fontsize=20) 
ax[1].set_ylim(0,1.5e-10)

###############################################################################

path = Jupiterlist_h[0]

tar = ascii.read(DATAPATH/('p'+path.stem+'_w_spec.csv'))
tar_wave = tar['wave']
tar_flux = tar['inten']
tar_cal = tar_flux*sensfunc #flux after flux calibration
tar_std = tar['std']*sensfunc

fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.plot(tar_wave,tar_cal,c='k')
# ax.set_xlim(4000,8000)
ax.set_title('After flux calibration')
ax.set_ylabel(r'erg $s^{-1}cm^{-2}\AA^{-1}$ ')
ax.set_xlabel(r'Dispersion axis $[\AA]$',fontsize=20) 
ax.set_ylim(0,1e-8)

spectrum = Table([tar_wave, tar_cal, tar_std],
                names=['wave', 'flux', 'error'])
spectrum['wave'].format = "%.3f" 
spectrum['flux'].format = "%.3e" 
spectrum['error'].format = "%.3e"

SPEC_SAVEPATH = DATAPATH/('p'+path.stem+'_wf_spec.csv') 
spectrum.write(SPEC_SAVEPATH, overwrite=True, format='csv')



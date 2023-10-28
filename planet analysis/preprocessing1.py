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
WD = HOME/'C:'/'SNU'/'2022_2'/'AO2'/'data'
SUBPATH = WD/'data'

DATAPATH = Path("C:/SNU/2022_2/AO2/data")
DATAPATH1 = Path("C:/SNU/2022_2/AO2/dataa")

Biaslist = sorted(list(Path.glob(DATAPATH1, 'Cali*bias.fit')))
Darklist= sorted(list(Path.glob(DATAPATH1, 'Cali*d*.fit')))
Flatlist = sorted(list(Path.glob(DATAPATH1, 'Flat*.fit')))
Complist = sorted(list(Path.glob(DATAPATH, 'Arc*af.fit')))

Objectlist1 = sorted(list(Path.glob(DATAPATH, 'jupiter_h*.fit')))
Objectlist2 = sorted(list(Path.glob(DATAPATH, 'jupiter_v*.fit')))
Objectlist3 = sorted(list(Path.glob(DATAPATH, 'saturn_h*.fit')))
Objectlist4 = sorted(list(Path.glob(DATAPATH, 'saturn_v*.fit')))

Standlist = sorted(list(Path.glob(DATAPATH1, 'HR9087*.fit')))

##Biaslist = sorted(list(Path.glob(SUBPATH, 'Cali*bias.fit')))
##Darklist= sorted(list(Path.glob(SUBPATH, 'Cali*d*.fit')))
##Flatlist = sorted(list(Path.glob(SUBPATH, 'Flat*.fit')))
##Complist = sorted(list(Path.glob(SUBPATH, 'Arc*af.fit')))

##Objectlist1 = sorted(list(Path.glob(SUBPATH, 'jupiter_h*.fit')))
##Objectlist2 = sorted(list(Path.glob(SUBPATH, 'jupiter_v*.fit')))
##Objectlist3 = sorted(list(Path.glob(SUBPATH, 'saturn_h*.fit')))
##Objectlist4 = sorted(list(Path.glob(SUBPATH, 'saturn_v*.fit')))

##Standlist = sorted(list(Path.glob(SUBPATH, 'HR9087*.fit')))

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

###############################################################################

# Checking the bias image
Name = []
Mean = []
Min = []
Max = []
Std = []

for i in range(len(Biaslist)):
    hdul = fits.open(Biaslist[i])
    data = hdul[0].data
    data = np.array(data).astype('float64')  # Change datatype from uint16 to float64
    mean = np.mean(data)
    minimum = np.min(data)
    maximum = np.max(data)
    stdev = np.std(data)
    print(Biaslist[i],mean,minimum,maximum)
    
    Name.append(os.path.basename(Biaslist[i]))
    Mean.append("%.2f" % mean )
    Min.append("%.2f" % minimum)
    Max.append("{0:.2f}".format(maximum))
    Std.append(f'{stdev:.2f}')
    
table = Table([Name, Mean, Min, Max, Std],
              names=['Filename','Mean','Min','Max','Std'])    
print(table)    

###############################################################################

# Plot the Sample image of bias image
File = Biaslist[1]  # pick one of the bias image
sample_hdul = fits.open(File)
sample_data = sample_hdul[0].data

fig,ax = plt.subplots(1,1,figsize=(10,5))
vmin = np.mean(sample_data) - 40
vmax = np.mean(sample_data) + 40
im = ax.imshow(sample_data,
               cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title(File, fontsize=12)

###############################################################################

# Median combine Bias image    
    
Master_bias = []
for i in range(0,len(Biaslist)):
    hdul = fits.open(Biaslist[i])
    bias_data = hdul[0].data
    bias_data = np.array(bias_data).astype('float64')
    Master_bias.append(bias_data)
MASTER_Bias = np.median(Master_bias,axis=0)


# Let's make Master bias image fits file

# Making header part
bias_header = hdul[0].header  # fetch header from one of bias images
bias_header['OBJECT'] = 'Bias'
# - add information about the combine process to header comment
bias_header['COMMENT'] = f'{Biaslist} bias images are median combined on ' \
                          + datetime.now().strftime('%Y-%m-%d %H:%M:%S (KST)')


SAVE_bias = DATAPATH/'Master_Bias.fits'  # path to save master bias
fits.writeto(SAVE_bias, MASTER_Bias, header=bias_header, overwrite=True)


# Plot Master bias 
fig,ax = plt.subplots(2, 1, figsize=(10,5))
im = ax[0].imshow(sample_data, cmap='gray',
                  vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
ax[0].set_title('Sample Bias image')
ax[0].tick_params(labelbottom=False, labelleft=False,
                  bottom=False, left=False)
plt.colorbar(im,cax=cax)

im1 = ax[1].imshow(MASTER_Bias, cmap='gray',
                   vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size='5%', pad=0.05)
ax[1].set_title('Master Bias image')
ax[1].tick_params(labelbottom=False, labelleft=False,
                  bottom=False, left=False)
plt.colorbar(im1, cax=cax)

###############################################################################

# Let's derive RN

# The following is just to give you an idea about the calculation
# - Bring Bias1 data
hdul1 = fits.open(Biaslist[0]) 
bias1 = hdul1[0].data            
bias1 = np.array(bias1).astype('float64')

# - Bring Bias2 data
hdul2 = fits.open(Biaslist[1])
bias2 = hdul2[0].data
bias2 = np.array(bias2).astype('float64')

# - Derive the differential image
dbias = bias2 - bias1

# - Bring Gain 
gain = 1.5 #e/ADU

# - Calculate RN
RN = np.std(dbias)*gain / np.sqrt(2)
print(f'Readout Noise is {RN:.2f}')



# Let's do it for all bias data
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

    print(i,'st',np.std(dbias)*gain / np.sqrt(2))
    RN.append(np.std(dbias)*gain / np.sqrt(2))
print('\nmean RN:', np.mean(RN))    
RN = np.mean(RN)

###############################################################################

# Checking what exposure time is taken
exptime = []

for i in range(len(Darklist)):
    hdul = fits.open(Darklist[i])[0]
    header = hdul.header
    exp = header['EXPTIME']  # get exposure time from its header 
    exptime.append(exp)
exptime = set(exptime)  # only unique elements will be remained
exptime = sorted(exptime)
print(exptime)

###############################################################################

# Bring master bias
Mbias = fits.open(SAVE_bias)[0].data
Mbias = np.array(Mbias).astype('float64')

# Making master dark image for each exposure time
for exp_i in exptime:
    Master_dark = []
    for i in range(len(Darklist)):
        hdul = fits.open(Darklist[i])[0]
        header = hdul.header
        exp = header['EXPTIME']

        if exp == exp_i:
            data = hdul.data
            data = np.array(data).astype('float64')
            bdata = data - Mbias  # bias subtracting
            Master_dark.append(bdata)
            
    MASTER_dark = np.median(Master_dark, axis=0)  # median combine bias-subtracted dark frames
    
    header['COMMENT'] = 'Bias_subtraction is done' + datetime.now().strftime('%Y-%m-%d %H:%M:%S (KST)')
    header["COMMENT"] = f'{len(Master_dark)} dark images are median combined on '\
                         + datetime.now().strftime('%Y-%m-%d %H:%M:%S (KST)')
    
    SAVE_dark = DATAPATH/('Master_Dark_'+str(exp_i)+'s.fits')
    fits.writeto(SAVE_dark, MASTER_dark, header=header, overwrite=True)
    print(exp_i ,'s is done!', SAVE_dark,' is made.')
    
###############################################################################

# Plot the flat image
fig, ax = plt.subplots(2, 1, figsize=(20, 10))
hdul = fits.open(Flatlist[4])[0]
data = hdul.data
im = ax[0].imshow(data, vmin=0, vmax=20000)
ax[0].set_title('Flat image')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right" ,size='5%', pad=0.05)
plt.colorbar(im, cax=cax)

# Standard star image
hdul = fits.open(Standlist[0])[0]
data = hdul.data
im = ax[1].imshow(data, vmin=0, vmax=20000)
ax[1].set_title('Standard star image')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right" ,size='5%', pad=0.05)
plt.colorbar(im, cax=cax)

###############################################################################

# Checking intensity profile through the dispersion axis

fig,ax = plt.subplots(2,1,figsize=(10,5))

for i in range(len(Flatlist)):
    hdul = fits.open(Flatlist[i])[0]
    data = hdul.data.astype('float64')
    flat = np.mean(data[115:135,:], axis=0)
    
    ax[0].plot(flat, label=Flatlist[i].name, lw=1)
    ax[0].legend(fontsize=8)
    ax[0].set_ylabel('Instrument intensity')
    ax[0].set_xlabel('Dispersion axis [pixel]')

hdul_std = fits.open(Standlist[0])[0]
data_std = hdul_std.data.astype('float64')
std = np.sum(data_std[115:135,:], axis=0)
ax[1].plot(std, label=Standlist[0].name, lw=1, c='k')
ax[1].set_ylabel('Instrument intensity')
ax[1].set_xlabel('Dispersion axis [pixel]')

###############################################################################

fig,ax = plt.subplots(4,1,figsize=(10,10), sharex=True)

def smooth(y, width):    # Moving box averaging
    box = np.ones(width)/width
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth   

smoothing_size = 100
smoother = median_filter

Coor_shift = []
for i in range(len(Flatlist)):
    hdul = fits.open(Flatlist[i])[0]
    data = hdul.data.astype('float64')
    flat = np.mean(data[115:135,:], axis=0)
    
    ax[0].plot(flat,label=Flatlist[i].name, lw=1, zorder=20)
    ax[0].legend(fontsize=8)
    
    sflat = smoother(flat, smoothing_size)
    nor_flat = flat/sflat  # flat field normalized by smoothed curve
    ax[0].plot(sflat, color='r', ls='--')
    
    ax[2].plot((flat-sflat)/sflat, lw=1, zorder=15, label=Flatlist[i].name)
    # ax[2].legend(fontsize=8)
    ax[2].axhline(0, color='k', lw=1)

# Let's see if a supposedly smooth source (in this case, a standard star)
# has same "wiggles" and "bumps" on its spectrum. If this is the case, we should
# model the flat lamp with "high order" and normalize the flat image with the model.
# If not, we 
sstd = smoother(std, smoothing_size)

ax[1].plot(std,label='Standard star', c='k', lw=1, zorder=20)
ax[1].plot(sstd, color='r', ls='--')
ax[1].legend(fontsize=8)
    
ax[3].plot((std-sstd)/sstd, lw=1, zorder=15)
ax[3].axhline(0, color='k', lw=1)

ax[2].annotate('Fringe pattern due to the grating', xy=(0.8, 0.8), xytext=(0.8, 0.9), xycoords='axes fraction', 
               ha='center', va='bottom',
               arrowprops=dict(arrowstyle='-[, widthB=10.0, lengthB=1.5', lw=2.0))
ax[2].annotate('Bump not originated \n from the grating', xy=(530, 0.03),  xycoords='data',
               xytext=(0.4, 0.7), textcoords='axes fraction',
               arrowprops=dict(facecolor='black', shrink=0.05),
               ha='right', va='bottom')

ax[0].set_title('Flat profile')
ax[1].set_title('Std profile')
ax[2].set_title('Normalized flat profile')
ax[3].set_title('Normalized std profile')

###############################################################################

# Bring Master bias & Dark

biasfile = DATAPATH/'Master_Bias.fits'
Mbias = fits.open(biasfile)[0].data
Mbias = np.array(Mbias).astype('float64')

darkfile = DATAPATH/'Master_Dark_30.0s.fits' # check the exposure time for your flat frames
Mdark = fits.open(darkfile)[0].data 
Mdark = np.array(Mdark).astype('float64')


# Make master Flat image
Master_flat = []
for i in range(len(Flatlist)):
    hdul = fits.open(Flatlist[i])[0]
    header = hdul.header
    
    data = hdul.data
    data = np.array(data).astype('float64')
    
    bdata = data - Mbias    # bias subtraction
    dbdata = bdata - Mdark  # dark subtraction
    Master_flat.append(dbdata)
    
MASTER_flat = np.median(Master_flat,axis=0)
header['COMMENT'] = 'Bias_subtraction is done' + datetime.now().strftime('%Y-%m-%d %H:%M:%S (KST)')
header["COMMENT"] = f'{len(Master_dark)} dark images are median combined on '\
                    + datetime.now().strftime('%Y-%m-%d %H:%M:%S (KST)')+'with '+' Master_Dark_'+str(exp)+'s.fits'




flat = np.median(MASTER_flat[:,900:1100],axis=1)
fig,ax=plt.subplots(2,1,figsize=(10,7))
ax[0].plot(flat,label='Master flat',lw=1,zorder=20)
ax[0].set_xlim(200,1500)
ax[0].legend(fontsize=10)


sflat = median_filter(flat, 100) # median filtering scale should be defined interactively
nor_flat = flat/sflat
ax[0].plot(sflat,color='r',ls='--')

ax[1].plot(nor_flat,lw=1,zorder=15)
ax[1].set_xlim(200,1500)
ax[1].axhline(1,color='r',lw=1)    

nor_flat2d = []
for i in range(len(MASTER_flat[0])):
    flat = MASTER_flat[:,i]
#     sflat = median_filter(flat, 100) # this might work better for cutout frames
    nor_flat2d.append(flat / sflat)
    
nor_flat2d = np.array(nor_flat2d).T    
header['COMMENT'] = 'Normalized' + datetime.now().strftime('%Y-%m-%d %H:%M:%S (KST)')

SAVE_flat = DATAPATH/'Master_Flat.fits'
fits.writeto(SAVE_flat, nor_flat2d, header=header, overwrite=True)



fig,ax = plt.subplots(1,1,figsize=(5,5))
im=ax.imshow(nor_flat2d)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",size='5%',pad=0.05)
plt.colorbar(im,cax=cax)

###############################################################################

# Bring Master bias & Dark

biasfile = DATAPATH/'Master_Bias.fits'
Mbias = fits.open(biasfile)[0].data
Mbias = np.array(Mbias).astype('float64')

darkfile = DATAPATH/'Master_Dark_15.0s.fits' # check the exposure time for your flat frames
Mdark = fits.open(darkfile)[0].data 
Mdark = np.array(Mdark).astype('float64')


# Make master Flat image
Master_flat = []
for i in range(len(Flatlist)):
    hdul = fits.open(Flatlist[i])[0]
    header = hdul.header
    
    data = hdul.data
    data = np.array(data).astype('float64')
    
    bdata = data - Mbias    # bias subtraction
    dbdata = bdata - Mdark  # dark subtraction
    Master_flat.append(dbdata)
    
MASTER_flat = np.median(Master_flat,axis=0)
header['COMMENT'] = 'Bias_subtraction is done' + datetime.now().strftime('%Y-%m-%d %H:%M:%S (KST)')
header["COMMENT"] = f'{len(Master_dark)} dark images are median combined on '\
                    + datetime.now().strftime('%Y-%m-%d %H:%M:%S (KST)')+'with '+' Master_Dark_'+str(exp)+'s.fits'

# normalizing master flat image
flat = np.median(MASTER_flat[115:135,:],axis=0)
fig,ax=plt.subplots(2,1,figsize=(10,7))
ax[0].plot(flat,label='Master flat',lw=1,zorder=20)
# ax[0].set_xlim(200,1500)
ax[0].legend(fontsize=10)

HIGH_WINDOW_LIM = [500, 620] # range limit for low- and high-order median filtering
flat_low1 = flat[:HIGH_WINDOW_LIM[0]]
flat_low2 = flat[HIGH_WINDOW_LIM[1]:]
flat_high = flat[HIGH_WINDOW_LIM[0]:HIGH_WINDOW_LIM[1]]

sflat_low1 = median_filter(flat_low1, 100, mode='nearest') # low-order filtering (left)
sflat_low2 = median_filter(flat_low2, 100, mode='nearest') # low-order filtering (right)
sflat_high = median_filter(flat_high, 10, mode='nearest') # high-order filtering (middle)
sflat = np.hstack([sflat_low1, sflat_high, sflat_low2])

nor_flat = flat/sflat
ax[0].plot(sflat,color='r',ls='--')

ax[1].plot(nor_flat,lw=1,zorder=15)
ax[1].set_ylim(0.5,1.5)
ax[1].axhline(1,color='r',lw=1)

high_window = np.full_like(flat, False)
high_window[HIGH_WINDOW_LIM[0]:HIGH_WINDOW_LIM[1]] = True
for a in ax:
    a.fill_between(np.arange(len(flat)),0, 1, where=high_window,
                    alpha=0.2, transform=a.get_xaxis_transform())
    a.text(0.1, 0.9, 'low-order-filtered range', transform=a.transAxes)
    a.text(0.51, 0.1, 'high-order-\nfiltered\nrange', transform=a.transAxes,
           c='#0296f0', ha='center')
    a.text(0.9, 0.1, 'low-order-filtered range', transform=a.transAxes,
           ha='right')

nor_flat2d = []
for i in range(MASTER_flat.shape[0]):
    flat = MASTER_flat[i,:]
    
    flat_low1 = flat[:HIGH_WINDOW_LIM[0]]
    flat_low2 = flat[HIGH_WINDOW_LIM[1]:]
    flat_high = flat[HIGH_WINDOW_LIM[0]:HIGH_WINDOW_LIM[1]]

    sflat_low1 = median_filter(flat_low1, 100, mode='nearest') # low-order filtering (left)
    sflat_low2 = median_filter(flat_low2, 100, mode='nearest') # low-order filtering (right)
    sflat_high = median_filter(flat_high, 10, mode='nearest') # high-order filtering (middle)
    
    sflat = np.hstack([sflat_low1, sflat_high, sflat_low2])
    nor_flat2d.append(flat / sflat)
    
nor_flat2d = np.array(nor_flat2d)  
header['COMMENT'] = 'Normalized' + datetime.now().strftime('%Y-%m-%d %H:%M:%S (KST)')

SAVE_flat = DATAPATH/'Master_Flat.fits'
fits.writeto(SAVE_flat, nor_flat2d, header=header, overwrite=True)



fig,ax = plt.subplots(1,1,figsize=(10,5))
im=ax.imshow(nor_flat2d)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",size='5%',pad=0.05)
plt.colorbar(im,cax=cax)
ax.set_title('Normalized master flat')

###############################################################################

# Find the local peak for each image

fig,ax = plt.subplots(2,1,figsize=(15,15))
Coor_shift = []
for i in range(len(Complist)):
    hdul = fits.open(Complist[i])[0]
    data = hdul.data
    data = np.array(data).astype('float64')
    neon = np.median(data[:,980:1020],axis=1)
    
    ax[0].plot(neon,label=Complist[i].name,lw=1)
    ax[0].set_xlim(200,1500)
    ax[0].legend(fontsize=10)
    
    
    x = np.arange(0,len(neon))
    # peak finding
    coordinates = peak_local_max(neon, min_distance=10, 
                                 threshold_abs=max(neon)*0.01)
    
    Coor_shift.append(coordinates) # to compare peak locations frame by frame
    
    

ax[0].set_title('Total {0} peakes'.format(int(len(coordinates))))

for i in coordinates:
    ax[0].annotate(i[0],(i,max(neon)+ max(neon)*0.03),
                  fontsize=10,rotation=80)
    ax[0].plot([i,i],
            [neon[i] + max(neon)*0.01, neon[i] + max(neon)*0.03],
            color='r',lw=1)
    
    
reference_coor = sorted(Coor_shift[0])    
     
for i in range(len(Coor_shift)-1):
    coor_i = Coor_shift[i+1]
    x_shift = []
    y_shift = []
    for k in range(len(reference_coor)):
        for t in range(len(coor_i)):
            if abs(reference_coor[k]-coor_i[t]) <5 :
                y_shift.append(reference_coor[k]-coor_i[t])
                x_shift.append(reference_coor[k])
    ax[1].plot(x_shift,y_shift,ls='',marker=f'${int(i)}$',
               markersize=15,color='r',alpha=1-i*0.1, label=Complist[i])    
ax[1].set_xlim(200,1500)    
ax[1].set_ylim(-3,5)
ax[1].set_ylabel('Degree of the shifted pixel \n of the each peak')
ax[1].axhline(0,color='gray')
ax[1].grid()
ax[1].legend()

plt.tight_layout()


###############################################################################


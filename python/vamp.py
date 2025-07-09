"""
Codes from https://bitbucket.org/dsorini/pygadds

Fit mock absorption spectra with Voigt profiles using the VAMP Bayesian VP fitter.

Romeel Dave, April 2019

History: VAMP was initially assembled by Chris Lovell in Fall 2016, with help
from Kate Storey-Fisher, to fit Gaussians to a spectrum via MCMC.  In Spring 2018, 
Sarah Appleby made some improvements to compute physical quantities from the
best-fit Gaussians.  In Fall 2018/Spring 2019, Jacob Christiansen added various other
useful features, extensively debugged the code, and made it into a usable package.
In Spring 2019, Romeel Dave tweaked the algorithm to make it more workable.
and inserted the VAMP package into pygad.

"""
import matplotlib,os
#matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from utils import *

# gaussian smoothing modules
from skimage.feature.peak import peak_local_max

#TODO: change the way that "Models are instantiated" (need to understand that first)
import warnings
warnings.filterwarnings("ignore", message="Instantiating a Model object directly is deprecated. We recommend passing variables directly to the Model subclass.")

def GaussFunction(x, amplitude, centroid, sigma):
    """
    Gaussian.

    Args:
    x (numpy array): wavelength array
    amplitude (float)
    centroid (float): must be between the limits of wavelength_array
    sigma (float)
    """
    return amplitude * np.exp(-0.5 * ((x - centroid) / sigma) ** 2)

def find_regions(wavelengths, fluxes, noise, continuum=1., min_region_width=2, N_sigma=4.0, extend=False, peak_dist=5, max_pixwidth=10, plot=False, tosave=None, verbose=True):
    """
    Finds detection regions above some detection threshold and minimum width.

    Args:
    wavelengths (numpy array)
    fluxes (numpy array): flux values at each wavelength
    noise (numpy array): noise value at each wavelength 
    min_region_width (int): minimum width of a detection region (pixels)
    N_sigma (float): detection threshold (std deviations)
    extend (boolean): default is False. Option to extend detected regions untill tau
                      returns to continuum.
    peak_dist: minimum distance between consecutive line peaks to be found

    Returns:
    regions_l (numpy array): contains subarrays with start and end wavelengths
    regions_i (numpy array): contains subarrays with start and end indices
    """

    if tosave and os.path.exists(tosave):
        regions_l, regions_i, regions_ipk = pkloadgzip(tosave, verbose=verbose)
    else:
        num_pixels = len(wavelengths)
        pixels = range(num_pixels)
        min_pix = 1
        max_pix = num_pixels - 1 # used in range(min_pix, max_pix)
        
        flux_ews = [0.] * num_pixels
        noise_ews = [0.] * num_pixels
        det_ratio = np.array([-float('inf')] * num_pixels) # max SNR among several gauss-smoothed spec
        
        # flux_ews has units of wavelength since flux is normalised. so we can use it for optical depth space
        for i in range(min_pix, max_pix):
            flux_dec = continuum - fluxes[i]
            if flux_dec < noise[i]:
                flux_dec = 0.0
            flux_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
            noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]
        
        # dev: no need to set end values = 0. since loop does not set end values
        flux_ews[0] = 0.
        noise_ews[0] = 0.
        
        # Range of standard deviations for Gaussian convolution in pixels
        std_min = 2
        std_max = int(np.round(max_pixwidth/(2.*np.sqrt(2.*np.log(2))))) # fwhm to sigma
        
        # Convolve varying-width Gaussians with equivalent width of flux and noise
        xarr = np.array([p - (num_pixels-1)/2.0 for p in range(num_pixels)]) # zero at middle pixel
        
        # this part can remain the same, since it uses EW in wavelength units, not flux
        for std in range(std_min, std_max+1):
        
            gaussian = GaussFunction(xarr, 1.0, 0.0, std)
        
            flux_func = np.convolve(flux_ews, gaussian, 'same')
            noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), 'same')
        
            # Select highest detection ratio of the Gaussians
            for i in range(min_pix, max_pix):
                noise_func[i] = 1.0 / np.sqrt(noise_func[i])
                if flux_func[i] * noise_func[i] > det_ratio[i]:
                    det_ratio[i] = flux_func[i] * noise_func[i]
        
        # Select regions based on detection ratio at each point, combining nearby regions
        # SNR > N_sigma and flux < 1 and width > min_region_width
        start = 0
        region_endpoints = []
        for i in range(num_pixels):
            if start == 0 and det_ratio[i] > N_sigma and fluxes[i] < continuum:
                start = i
            elif start != 0 and (det_ratio[i] < N_sigma or fluxes[i] > continuum):
                if (i - start) > min_region_width:
                    end = i
                    region_endpoints.append([start, end])
                start = 0
        
        # made extend a kwarg option
        # lines may not go down to 0 again before next line starts
        
        if extend:
            # Expand edges of region until flux goes above 1
            regions_expanded = []
            for reg in region_endpoints:
                start = reg[0]
                i = start
                while i > 0 and fluxes[i] < continuum:
                    i -= 1
                start_new = i
                end = reg[1]
                j = end
                while j < (len(fluxes)-1) and fluxes[j] < continuum:
                    j += 1
                end_new = j
                regions_expanded.append([start_new, end_new])
        
        else: regions_expanded = region_endpoints
        
        # Change to return the region indices
        # Combine overlapping regions, check for detection based on noise value
        # and extend each region again by a buffer
        regions_l = []
        regions_i = []
        buffer = 3 # to extend region by +/- buffer pixels
        for i in range(len(regions_expanded)):
            start = regions_expanded[i][0]
            end = regions_expanded[i][1]
            #TODO: this part seems to merge regions if they overlap - try printing this out to see if it can be modified to not merge regions?
            if i<(len(regions_expanded)-1) and end > regions_expanded[i+1][0]:
                end = regions_expanded[i+1][1]
            for j in range(start, end):
                flux_dec = continuum - fluxes[j]
                if flux_dec > abs(noise[j]) * N_sigma:
                    if start >= buffer:
                        start -= buffer
                    if end < len(wavelengths) - buffer:
                        end += buffer
                    regions_l.append([wavelengths[start], wavelengths[end]])
                    regions_i.append([start, end])
                    break
        
        # find line peaks in each region
        regions_ipk = []
        for start, end in regions_i:
            ipeaks = peak_local_max(det_ratio[start:end], peak_dist, threshold_abs=N_sigma)
            ipeaks = ipeaks.flatten()
            regions_ipk.append(ipeaks)
        print('VAMP : Found {} detection regions.'.format(len(regions_l)))
        print('VAMP : Found {} lines.'.format(sum([len(ipk) for ipk in regions_ipk])))
        regions_l = np.array(regions_l, dtype=object)
        regions_i = np.array(regions_i, dtype=object)
        regions_ipk = np.array(regions_ipk, dtype=object)
        if tosave: pkdumpgzip([regions_l, regions_i, regions_ipk], tosave)
        
        # plot spectrum, det_ratio and regions
        if plot:
            N_sigma /= max(det_ratio)
            det_ratio /= max(det_ratio)
            plt.plot(wavelengths, fluxes)
            plt.plot(wavelengths, noise, 'r', lw=0.5)
            plt.plot(wavelengths, det_ratio)
            for left,right in regions_i:
                left = wavelengths[left]
                right = wavelengths[right]
                plt.fill_betweenx([0,continuum],left,right,alpha=0.3,color='y')
                plt.plot([left,right], [0,continuum], 'k')
                plt.axvline(left, color='k')
                plt.axvline(right, color='k')
            for ireg,[start,end] in enumerate(regions_i): plt.plot(wavelengths[start:end][regions_ipk[ireg]], det_ratio[start:end][regions_ipk[ireg]],'^r')
            for ireg,[start,end] in enumerate(regions_i): plt.plot(wavelengths[start:end][regions_ipk[ireg]], fluxes[start:end][regions_ipk[ireg]],'vb')
            plt.axhline(N_sigma,color='k')
            plt.axhline(continuum,color='k')
            plt.axhline(0, color='k')
            plt.show()

    return regions_l, regions_i, regions_ipk

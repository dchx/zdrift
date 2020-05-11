from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.rc('font',size=15) # global font size
import time,datetime,copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import os,glob,pickle,gzip,sys,itertools
from skimage.feature.peak import peak_local_max
import astropy.units as u
import astropy.constants as c
from astropy.modeling.models import Voigt1D
from astropy.modeling.models import Gaussian1D
from astropy import cosmology

#path = '/astro/homes/dcx/dcxroot/zdrift/spec_sim/'
#path = '/Users/dong/Documents/Research/zdrift/spec_sim/'
#path = '/Volumes/SeagateBlack/temple_20200313/zdrift/spec_sim/'
path = os.path.dirname(os.path.realpath(os.path.dirname(__file__))) + '/' # python/.. -> spec_sim/
lya_wave = 1215.67 # Angstrom
def nu2aa(nu): return c.c.value / nu * 1e10
def aa2nu(aa): return c.c.value / aa * 1e10
lya_freq = aa2nu(lya_wave) # s-1

def rest_frame(lam_obs, z):
	# input observed wavelengths lam_obs, provide redshift z
	# output rest frame wavelengths
	lam_emit = lam_obs/(1.+z)
	return lam_emit 

def obs_frame(lam_emit, z):
	# input rest frame wavelengths lam_emit, provide redshift z
	# output observed wavelengths
	lam_obs=lam_emit*(1.+z)
	return lam_obs

def lam2z(lam):
	'''
	convert a series of wavelengths (lam) in a Lya forest to redshifts of clouds on the path
	assuming lam in observed frame
	'''
	zs = lam / lya_wave - 1.
	return zs

def rest_fram_pars(redsft,plot_rest_frame=True):
	if plot_rest_frame: 
		z_plot_rest_frame=redsft #redshift if plot in rest frame, else zero
		lya_toplot=lya_wave # lya 1215.67 Angstrom
	else: 
		z_plot_rest_frame=0.
		lya_toplot=obs_frame(lya_wave,redsft)
	return z_plot_rest_frame,lya_toplot

def expand_plt_range(range0,factor=0.1):
	# expand the axis range by factor on each side
	# input initial range0: [mini,maxi], output expanded range
	mini,maxi=range0
	dist=maxi-mini
	return mini-factor*dist, maxi+factor*dist

def ra_hms2deg(rastr):
	# convert RA 'hh:mm:ss.ss' to degrees
	hh=int(rastr.split(':')[0])
	mm=int(rastr.split(':')[1])
	ss=float(rastr.split(':')[2])
	deg=(hh+mm/60.+ss/3600.)*15.
	return deg

def dec_dms2deg(decstr):
	# convert Dec 'dd:mm:ss.ss' to degrees
	dd=int(decstr.split(':')[0])
	mm=int(decstr.split(':')[1])
	ss=float(decstr.split(':')[2])
	deg=dd+mm/60.+ss/3600.
	return deg

def csv2recarr(fcsv):
	# input csv file path string, output recarray
	lines=open(fcsv).readlines()
	lines=[line.replace(',,',',0,').rstrip('\n') for line in lines] # replace empty element to '0', return list of lines
	array=[tuple(line.split(',')) for line in lines]
	names=list(array[0]);names[0]=names[0].lstrip('\xef\xbb\xbf').lstrip('\ufeff')
	arraytype=[(i,int) if (i=='No' or i=='KOAjobID') else (i,'S10') if (i=='flag') else (i,float) for i in names]
	table1=np.array(array[1:],dtype=arraytype)
	return table1

def recarr2csv(recarr):
	# input recarray, output string to be write to csv file
	outstr=','.join(recarr.dtype.names)
	for line in recarr: outstr+='\r'+','.join([str(col) for col in line])
	return outstr

matched=csv2recarr(path+'Table1_matched.csv') # table1 keck csv file
def saveid_func(i): return '%02d_%03d'%(i,matched['No'][i])

def z_from_koajobid(koajobid):
	# return redshift from koajobid
	return matched['z'][matched['KOAjobID']==koajobid][0]

def koajobid2num(koajobid):
	# int int
	return matched['No'][matched['KOAjobID']==koajobid][0]

def num2koajobid(num):
	# int int
	return matched['KOAjobID'][matched['No']==num][0]

def cut_lya(spec,lya_toplot,searchrange=50.,adjust_ind=-10,searchlya=False):
	# spec: (lam,flux,flux_err(,disp,exptime))
	# searchrange: search lya peak at (lya-searchrange,lya+searchrange), in Angstrom
	# return only lya forest spectrum
	# adjust_ind: how many more values to the right of lya peak
	lam=spec[0]
	flux=spec[1]

	# search for lya peak
	if searchlya:
		therange=[lya_toplot-searchrange,lya_toplot+searchrange]
		searchindex=np.where((lam>therange[0])*(lam<therange[1]))[0]
		if len(searchindex)==0: raise ValueError("No Lya in the spectrum.")
		lyaindex=searchindex[np.argmax(flux[searchindex])]
		foundlya=lam[lyaindex]
	else:
		cuttedind=np.where(lam<lya_toplot)[0]
		if len(cuttedind)==0: lyaindex=0
		else: lyaindex=cuttedind[-1]

	if (lyaindex+1)==len(lam): right_edge=len(lam)
	else: right_edge=np.min([len(lam),(lyaindex+adjust_ind)])# right wave index to cut
	newspec=[]
	for ispec in range(len(spec)): newspec.append(spec[ispec][:right_edge]) # do cut
	newspec=tuple(newspec)
	if searchlya: return newspec, foundlya
	else: return newspec

def fit_poly(spec, local_dist=5, poly_deg=4):
	'''
	spec: lam, flux, flux_err(, disp, exptime)
	local_dist: min_distance for peak_local_max
	'''
	lam=spec[0]
	flux=spec[1]

	# use only local max
	ipeak=peak_local_max(flux,min_distance=local_dist).flatten()
	if len(ipeak)==0: ipeak=list(range(len(lam))) # can't find local max: use whole spec
	lam_topoly=lam[ipeak]
	flux_topoly=flux[ipeak]

	ppoly=np.polyfit(lam_topoly,flux_topoly,poly_deg)
	return ppoly

def cut_spec(spec,lamlim):
	'''
	cut a spec [lam, flux, ...] with lamlim [min, max]
	'''
	spec=np.array(spec)
	initial_dim=len(spec.shape)
	spec=np.atleast_2d(spec)
	specut=spec.T[(spec[0]>=lamlim[0])*(spec[0]<=lamlim[1])].T
	if specut.shape[0]==1 and initial_dim==1: specut=np.squeeze(specut,axis=0) # if spec has only lam
	return specut

def connect_chunks(specs):
	'''
	connect chunked spectra to one
	------
	Input specs: [(lam, flux, flux_err, ...), (lam, flux, flux_err, ...), ...]
	Output connected_spec: [lam, flux, flux_err, ...] connected
	'''
	connected_spec = []
	for idim in range(len(specs[0])): # loop through lam, flux, ...
		connected_dim = []
		for chunk in specs: connected_dim.append(chunk[idim]) # loop through chunks
		connected_spec.append(np.hstack(connected_dim))
	indsortlam = np.argsort(connected_spec[0])
	for idim in range(len(connected_spec)): connected_spec[idim] = connected_spec[idim][indsortlam] # sort by lam
	return connected_spec

def add_shot_noise(flux, nphot, sky=1e-12, return_error=False):
	'''
	flux should be normalized to [0, 1]
	sky - value in [0, 1], squeezes the spectrum to [sky, 1]
	'''
	if nphot==np.inf: # no error
		flux_werr = flux
		error = np.zeros(flux.shape)
	else:
		flux_nphot = (flux + sky) / (1. + sky) * nphot
		err_nphot = flux_nphot**0.5
		#print('flux percentage error after adding noise:', np.mean(np.abs(err_nphot*np.random.normal(0.0,1.0,len(flux_nphot)))/flux_nphot))
		#flux_nphot=flux_nphot+err_nphot*np.random.normal(0.0,1.0,len(flux_nphot)) # use normal with sigma=sqrt(flux_nphot)
		flux_nphot_werr = np.random.poisson(flux_nphot) # use poisson
		flux_werr = flux_nphot_werr / float(nphot) * (1. + sky) - sky
		error = err_nphot / float(nphot)
	if return_error: return flux_werr, error
	else: return flux_werr

def pkdump(data, pfile, verbose=True):
	with open(pfile,'wb') as f:
		if sys.version_info.major == 2: pickle.dump(data, f)
		if sys.version_info.major == 3: pickle.dump(data, f, protocol=2)
	if verbose: print('Saved:%s'%pfile)

def pkload(pfile, verbose=True):
	if sys.version_info.major == 2: 
		with open(pfile, 'r') as f: results = pickle.load(f)
	if sys.version_info.major == 3:
		with open(pfile, 'rb') as f: results = pickle.load(f, encoding='latin1')
	if verbose: print('Loaded:%s'%pfile)
	return results

def pkdumpgzip(data, pfile, verbose=True):
	with gzip.open(pfile,'wb') as f:
		if sys.version_info.major == 2: pickle.dump(data, f)
		if sys.version_info.major == 3: pickle.dump(data, f, protocol=2)
	if verbose: print('Saved:%s'%pfile)
	
def pkloadgzip(pfile, verbose=True):
	with gzip.open(pfile,'r') as f:
		if sys.version_info.major == 2: results = pickle.load(f)
		if sys.version_info.major == 3: results = pickle.load(f, encoding='latin1')
	if verbose: print('Loaded:%s'%pfile)
	return results

numtest=[23,54,68,105,126,155,161,162,164,186,203,205]
#numtest=[102,185] # with high resolution
itest=np.array([np.where(numt==matched['No']) for numt in numtest]).flatten() # select numtest

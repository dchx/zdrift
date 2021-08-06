from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.rc('font',size=12, family='serif') # global font size
matplotlib.rc(('xtick', 'ytick'), direction='in') # axis tick direction
matplotlib.rc('xtick', top=True)
matplotlib.rc('ytick', right=True)
matplotlib.rc(('xtick.major', 'ytick.major'), size=6)
matplotlib.rc(('xtick.minor', 'ytick.minor'), size=3)
import os, glob, pickle, gzip, sys, itertools, warnings, time, datetime, copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.tick_params(which='major', length=6)
#plt.tick_params(which='minor', length=3)
from astropy.io import fits
import astropy.io.ascii as asc
import astropy.table as tb
if sys.version_info.major == 3: from importlib import reload
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
keck_catalog = 'elqs' # '.', 'elqs' or 'brightest'

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

df_sdss = pd.read_csv(path+'Table1_Keck_addKOAjobID.csv')
df_elqs = pd.read_csv(path+'data/elqs_full_sortM1450_addmore.csv')
#df_elqs_ps = pd.read_csv(path+'data/elqs_panstarrs_table7_concise.dat',sep='\s+') # pan-starrs elqs catalog
df_elqs_ps = pd.read_csv(path+'data/elqs_panstarrs_table7_concise.csv') # pan-starrs elqs catalog
df_all = pd.read_csv(path + 'elqs_and_sdss_allwithKOAjobID.csv') # all available good data (both elqs and sdss), considering duplicates
df_all = df_all.set_index(df_all.KOAjobID)
#southern = pd.read_csv(path + 'data/Boutsia2020_table3_apjsabafc1t3_ascii.txt', sep='\s+', comment='#')
southern = pd.read_csv(path + 'data/Boutsia2020_table3_apjsabafc1t3_ascii.csv')
def get_matched(keck_catalog):
	if keck_catalog == '.' or keck_catalog == 'sdss':
		d = df_sdss
		#matched=csv2recarr(path+'Table1_matched.csv') # table1 keck csv file
	elif keck_catalog == 'elqs':
		d = df_elqs
	elif keck_catalog == 'ps-elqs':
		d = df_elqs_ps
	elif keck_catalog == 'Boutsia20':
		d = southern
	else: raise Exception("keck_catalog not recognized")
	matched = d.to_records(index=False)
	return d, matched
d, matched = get_matched(keck_catalog)

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

def pkdump(data, pfile, verbose=True):
	'''
	for .pickle files
	'''
	with open(pfile,'wb') as f:
		if sys.version_info.major == 2: pickle.dump(data, f)
		if sys.version_info.major == 3: pickle.dump(data, f, protocol=2)
	if verbose: print('Saved:%s'%pfile)

def pkload(pfile, verbose=True):
	'''
	for .pickle files
	'''
	if sys.version_info.major == 2: 
		with open(pfile, 'r') as f: results = pickle.load(f)
	if sys.version_info.major == 3:
		with open(pfile, 'rb') as f:
			try: results = pickle.load(f, encoding='latin1')
			except Exception as e:
				print('Error in %s'%pfile)
				raise e
	if verbose:
		print('Loaded:%s'%pfile)
	return results

def pkdumpgzip(data, pfile, verbose=True):
	'''
	for .pzip files
	'''
	with gzip.open(pfile,'wb') as f:
		if sys.version_info.major == 2: pickle.dump(data, f)
		if sys.version_info.major == 3: pickle.dump(data, f, protocol=2)
	if verbose: print('Saved:%s'%pfile)
	
def pkloadgzip(pfile, verbose=True):
	'''
	for .pzip files
	'''
	with gzip.open(pfile,'r') as f:
		if sys.version_info.major == 2: results = pickle.load(f)
		if sys.version_info.major == 3: results = pickle.load(f, encoding='latin1')
	if verbose: print('Loaded:%s'%pfile)
	return results

def subplot_shape(naxes):
	nrow = np.ceil(np.sqrt(naxes))
	ncol = np.ceil(naxes/nrow)
	return int(nrow), int(ncol)

numtest=[23,54,68,105,126,155,161,162,164,186,203,205]
#numtest=[102,185] # with high resolution
itest=np.array([np.where(numt==matched['No']) for numt in numtest]).flatten() # select numtest

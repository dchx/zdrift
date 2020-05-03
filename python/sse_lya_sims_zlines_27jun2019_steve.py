# coding: utf-8

# this version divides the spectrum into line regions for correlation.
from utils import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import scipy.ndimage
import numpy as np
import gzip, pickle, glob
from astropy.modeling.models import Voigt1D
from read_koa import smoothwidth,get_res
import generate_spec as gs
import voigtforest as vf
import sse
#from norm_koa import koa_normed_spec
from continuum_fit import get_keck_spec
from cosmology import dz2dv
from liske_sigma import sigma_liske_empir
fwhm_smooth = smoothwidth/2.*2.355 * 0.01 #AA
ind = 3; # lamrange=[1100.,1300.]; R=35800
ind = 7; # lamrange=[900.,1200.]; R=35800
good_inds = 7, 30, 33, 35, 39, 41, 59 # good distribution
soso_inds = 19, 31, 37, 48, 53, 54 # have peak at 0 but shorter than good peak
bigpeak_inds = 24, 27, 28, 42, 44, 49, 50 # have spike like peak
ind = 7
#koa_spec = koa_normed_spec(ind)
koa_spec = get_keck_spec(ind, normalize=False)
lamrange = [np.min(koa_spec[0]), np.max(koa_spec[0])] 
saveid = saveid_func(ind)
z = matched['z'][ind]
koajobid = matched['KOAjobID'][ind]
R_keck = get_res(koajobid)
#R_keck = 46200.

#### generate_spec arguments
use_genspec = 0 # use generated spectra from literature line parameter distributinos (generate_spec.py)
restframe_genspec = 0 # whether to generate in rest frame
ispec = None # if to generate using different sets of parameters
#### Keck spectra arguments
smooth = True # whether to use smoothed Keck spectra
koa_rest2obs = 1 # whether to convert restframe spec to observed frame for keck
#### voigtforest fit arguments
use_voigtforest = 1 # use voigt fit result by voigtforest.py
fitaddline = 0 # whether to try add lines when fitting voigt
CSL_cut = 1.5 # default 1.5
if use_genspec:
	smooth = False
	koa_rest2obs = False
	use_voigtforest = False
#### applying-to-all arguments
fixres = 'resele' # res or resele, whether to fix R or resolution element size (vs lam)
dlam_fixresele = 0.0125 # AA per pixel, useful only if fixres=='resele'

# texts
smoothtxt = '_smooth%dpix'%smoothwidth if smooth else '_nosmooth'
fitaddlinetxt = '_fitaddline' if fitaddline else ''
csltxt = '_CSLcut%.1f'%CSL_cut
fitcont_mode = 'poly' # poly, linear or cubic
if ind==7: fitcont_dist = 100; fitcont_deg = 10
else: fitcont_dist = 100; fitcont_deg = 10
fitconttxt = '_dist%d'%fitcont_dist + ('_deg%d'%fitcont_deg if fitcont_mode=='poly' else '_cont%s'%fitcont_mode)

toglob = path+'paras/bestparams_%s_*_connected'%saveid+smoothtxt+'.pzip'
def voigtforest_pfile_fromind(ind): 
	saveid = saveid_func(ind)
	#voigtforest_pfile = path+'paras/voigtforest_bestp_'+saveid+smoothtxt+fitconttxt+fitaddlinetxt+csltxt+'_applyminlamwidth.pzip'
	#voigtforest_pfile = path+'paras/voigtforest_bestp_'+saveid+smoothtxt+fitconttxt+fitaddlinetxt+csltxt+'_typicallw20kms.pzip'
	voigtforest_pfile = path+'paras/voigtforest_bestp_'+saveid+smoothtxt+fitconttxt+fitaddlinetxt+csltxt+'.pzip'
	return voigtforest_pfile
voigtforest_pfile = voigtforest_pfile_fromind(int(saveid[:2]))
if use_voigtforest: param_files = voigtforest_pfile
else: param_files = glob.glob(toglob)

if use_voigtforest:
	def get_params_dist(param_files, res=2e4, divide=8., adjlw=True):
		parray = vf.pfile2parray(param_files) # lam0, AL, fL, fG
		parray[2] /= divide
		parray[3] /= divide
		if adjlw: parray[2], parray[3] = adjust_fwhm_byR(parray[2], parray[3], parray[0], res)
		return parray
else:
	def get_params_dist(param_files, res=2e4, divide=8., adjlw=True):
		'''
		adjlw: whether to adjust line width according to res and divide
		'''
		lam0_list=list()
		AL_list=list()
		fL_list=list()
		fG_list=list()
		if not adjlw: divide=1.

		for pfile in param_files:
			bestp = pkloadgzip(pfile)
			npara=len([i for i in list(bestp.keys()) if 'lam0abs' in i])
			for i in range(npara):
				lam0 = bestp['lam0abs'+str(i)].value
				fL = bestp['fLabs'+str(i)].value / divide
				fG = bestp['fGabs'+str(i)].value / divide
				if adjlw: fL, fG = adjust_fwhm_byR(fL, fG, lam0, res)
				lam0_list.append(lam0)
				AL_list.append(bestp['ALabs'+str(i)].value)
				fL_list.append(fL)
				fG_list.append(fG)
		lam0=np.array(lam0_list)
		AL=np.array(AL_list)
		fL=np.array(fL_list)
		fG=np.array(fG_list)
		return lam0,AL,fL,fG

def mk_qso_spec3_dz(zqso,dz,refparams,res,addline=250,divide=8.,mode='dz'):
	# this version adds an additional 250 small lines, able to vary res
	# additional line paras randomly drop from paras from the param_files
	# res = spectral resolution
	frametxt = '' if koa_rest2obs else '_restframe'
	if fixres == 'res': spec_file = path + 'data/keck_gen_spec/keck_spec_%s%s%s_R%.1e_%s%.3e.pickle'%(saveid,frametxt,fitconttxt,res,mode,dz)
	elif fixres == 'resele': spec_file = path + 'data/keck_gen_spec/keck_spec_%s%s%s_dlam%.3e_%s%.3e.pickle'%(saveid,frametxt,fitconttxt,dlam_fixresele,mode,dz)
	else: raise Exception("argument 'fixres' should be either 'res' or 'resele'")
	if os.path.exists(spec_file): # load previously saved spectra, all not added noise
		lam, flux1 = pkload(spec_file, verbose=False)
	else:
		lam, res_ele = gs.lam_grid(lamrange[0], lamrange[1], fixres, res=res, dlam_fixresele=dlam_fixresele)
		flux1 = multivoigt_dz(param_files,lam,zqso,dz,res,divide=divide,mode=mode)
		# add 250 small lines
		if addline:
			for i in range(addline):
				flux1 += singlevoigt_dz(refparams[0,i],refparams[1,i],refparams[2,i],refparams[3,i],lam, zqso, dz, mode=mode)
		# ---- adjust flux
		#flux1=flux1-np.min(flux1) # make it above zero
		#flux1=2.41*flux1/np.sum(flux1) # normalize
		if koa_rest2obs: lam = obs_frame(lam, zqso) # convert to observed frame, assuming already in rest frame
		pkdump((lam, flux1), spec_file)
	return flux1, lam

def intrinsic_fwhm(fl_obs, fg_obs, lam0, smooth, R_keck=R_keck):
	'''
	return intrinsic line width adjusting for fitting smooth and R_keck
	'''
	fl_obs, fg_obs, lam0 = np.atleast_1d(fl_obs),  np.atleast_1d(fg_obs),  np.atleast_1d(lam0)
	fwhm_keck = lam0 / R_keck
	fv_obs = vf.fv(fl_obs, fg_obs)
	if smooth: fwhm_toadj = np.sqrt(fwhm_smooth**2. + fwhm_keck**2.)
	else: fwhm_toadj = fwhm_keck
	least_fv_intrin = 0./3e5*lam0 # make intrinsic fwhm to be at least 5 km/s
	fv_intrin_square = fv_obs**2. - fwhm_toadj**2.
	fv_intrin = np.array([least_fv_intrin[i] if fv_intrin_square[i]<least_fv_intrin[i]**2.\
	                      else np.sqrt(fv_intrin_square[i]) for i in range(len(fv_obs))])
	return np.squeeze(fv_intrin), np.squeeze(fv_obs)

R_int = 1.
n_int = gs.lam_grid(lamrange[0], lamrange[1], fixres, res=R_int, dlam_fixresele=dlam_fixresele, return_nlam=True)
def adjust_fwhm_byR(fl_para, fg_para, lam0, res):
	fv_intrin, fv_para = intrinsic_fwhm(fl_para, fg_para, lam0, smooth)
	if fixres == 'res': fwhm_ins = lam0 / res # my instrument fwhm
	elif fixres == 'resele': fwhm_ins =  4. * np.sqrt(2*np.log(2.)) * dlam_fixresele # 2sigma == 4 * dlam_fixresele
	fwhm_tot = np.sqrt(fv_para**2. + fwhm_ins**2.) # for noadjustKecklw (adding my instrument R)
	fwhm_tot = np.sqrt(fv_intrin**2. + fwhm_ins**2.) # for adjust all (subtracting keck R, adding my instrument R)
	factor = fwhm_tot / fv_para
	fl_adjust = fl_para * factor
	fg_adjust = fg_para * factor
	#fl_adjust, fg_adjust = fl_para, fg_para # for noadjustlw
	return fl_adjust, fg_adjust

def inte_spec(lam_dense, flux_dense, lam_lowres):
	'''
	for low resolution spectra, integrate denser pixel scale spectra to generate
	'''
	bin_width_half = (lam_lowres[1] - lam_lowres[0])/2.
	lowres_grid = np.r_[lam_lowres[0]-bin_width_half, lam_lowres + bin_width_half]
	flux_lowres = np.zeros(len(lam_lowres))
	for i in range(len(lam_lowres)): # loop through lowres bins
		inds = (lowres_grid[i] <= lam_dense) * (lam_dense < lowres_grid[i+1])
		flux_lowres[i] = np.mean(flux_dense[inds])
	return flux_lowres

def multivoigt_dz(param_files, lam, zqso, dz, res, divide=8., mode='dz'):
	'''
	return spectrum as either tau or exp(-tau), setted in voigtforest
	mode - 'dz' if dz is real dz
	       'dl' if dz is d_lambda
	'''
	if res<R_int: lam_tovoigt = np.linspace(lamrange[0], lamrange[1], n_int)
	else: lam_tovoigt = lam
	parray = get_params_dist(param_files, res=res, divide=divide) # lam0, AL, fL, fG
	multivoigt = gs.parray2flux_dz(parray, lam_tovoigt, mode, dz, zqso, voigt_is_tau=vf.voigt_is_tau, v1d=vf.v1d)
	if res<R_int: multivoigt = inte_spec(lam_tovoigt, multivoigt, lam)
	return multivoigt

def singlevoigt_dz(lam0, AL, fL, fG, lam, zqso, dz, mode='dz'):
	nlam = len(lam)
	if nlam<n_int: lam_tovoigt = np.linspace(lamrange[0], lamrange[1], n_int) # corresponding to R=2e4
	else: lam_tovoigt = lam
	parray = np.array([[lam0], [AL], [fL], [fG]])
	multivoigt = gs.parray2flux_dz(parray, lam_tovoigt, mode, dz, zqso, voigt_is_tau=vf.voigt_is_tau, v1d=vf.v1d)
	if nlam<n_int: multivoigt = inte_spec(lam_tovoigt, multivoigt, lam)
	return multivoigt

def find_zshift_lines(res, flux_obs, refparams, zmin, zmax, delta_dz, chk_boundinds, nphot, zqso, addline=250, divide=8., mode='dz', fix='res', dlam=0.0125, ispec=None):
	'''
	res - spectral resolution
	chk_boundinds - shape:(nchunk, 2), left and right index bounds of chunks
	'''
	ztest=np.arange(zmin,zmax,delta_dz)
	zfit=np.zeros(len(chk_boundinds))
	rcorr=np.zeros([len(chk_boundinds),len(ztest)])
	
	for i in range(len(ztest)): # loop through all test dz
		if use_genspec:
			lam, ftest = gs.generate_spec(zqso, res=res, dz=ztest[i], rest_frame=restframe_genspec, shiftmode=mode, fix=fix, dlam=dlam, ispec=ispec)
		else:
			ftest, lam = mk_qso_spec3_dz(zqso,ztest[i],refparams,res,addline=addline,divide=divide,mode=mode)
		# ftest = add_shot_noise(ftest,nphot) # for bothNoise
		for ichk in range(len(chk_boundinds)): # loop through chunks
			i1=chk_boundinds[ichk,0]
			i2=chk_boundinds[ichk,1]
			rcorr[ichk,i] = np.corrcoef(ftest[i1:i2],flux_obs[i1:i2])[0,1]

	# find best dz
	for ichk in range(len(chk_boundinds)): # loop through chunks
		zfit[ichk] = ztest[np.min(np.where(rcorr[ichk,:] == np.max(rcorr[ichk,:])))]
	# zfit = ztest[np.argmax(rcorr,axis=1)]
	
	return zfit # the best dz found

def do_lya_sims_zlines(zqso,nphot,zmin,zmax,delta_dz,minlinesize,prefix,res,real_dz=0.,ntrial=1,addline=250,divide=8.,mode='dz',load=False,i_refpar=0,fix='res',dlam=0.0125,ispec=None):
	'''
	use same spectra for each trial
	Returns
	zfits - nchunk * 100, best dz found for each chunk each trial
	load - whether to load saved parameters or run the analysis
	'''
	if use_genspec: tosave = path+'paras/zlines_cut'+prefix
	else: tosave = path+'paras/zlines_cut'+saveid+prefix
	print('tosave: ',tosave)
	if load:
		zfits = pkload(tosave+'_zfits.pickle')
		#print('Loaded:', tosave+'_zfits.pickle')
		return zfits

	if use_genspec:
		lam, flux1 = gs.generate_spec(zqso, res=res, rest_frame=restframe_genspec, fix=fix, dlam=dlam, ispec=ispec)
		refparams = None
	else:
		# create basic parameters for reference spectrum
		# generate 250 lines by randomly choose AL, fL, fG from the param_files with random lam0 in (1120,1205)
		refparams=np.zeros([4,addline])
		if addline:
			ireftxt = '' if i_refpar==0 else '_%d'%i_refpar
			if use_voigtforest: refparams_file = path+'paras/voigtforest_refparams_'+saveid+'_addline%d'%addline+smoothtxt+ireftxt+'.pickle'
			else: refparams_file = path+'paras/refparams_'+saveid+'_addline%d'%addline+smoothtxt+ireftxt+'.pickle'
			lams, AL,fL,fG = get_params_dist(param_files, res=res, divide=divide) # get the distribution of parameters for widths
			if os.path.exists(refparams_file):
				i_fwhms, i_amps, lam0s = pkload(refparams_file)
			else:
				i_fwhms=[]; i_amps=[]; lam0s = [];
				for i in range(addline):
					j=np.int(round(len(fL)*np.random.uniform(0,1))) # pick one set of fL and fG at random
					k=np.int(round(len(AL)*np.random.uniform(0,1))) # pick one value of AL at random
					lam0=np.random.uniform(min(lams),max(lams)) # random wavelenth in this range
					i_fwhms.append(j); i_amps.append(k); lam0s.append(lam0)
				pkdump((i_fwhms, i_amps, lam0s), refparams_file)
			for i in range(addline):
				refparams[:,i]=np.array([lam0s[i],AL[i_amps[i]],fL[i_fwhms[i]],fG[i_fwhms[i]]])
		# read in params, add 250 lines
		flux1, lam = mk_qso_spec3_dz(zqso,real_dz,refparams,res,addline=addline,divide=divide,mode=mode) # spec not shifted

	# divide spectral template into chunks
	chk_boundinds = sse.chunk_spec(lam, flux1, minlinesize=minlinesize, chunk_flux_threshold=None) # chunk_flux_threshold: None (gradchk), 0.8, 0.975

	zfits=np.zeros([len(chk_boundinds),ntrial]) # the best fit redshift for each of 100 trials for each lambda bin
	for i in range(ntrial): # try 100 times
		t1=time.time()
		flux3=add_shot_noise(flux1,nphot) # add noise

		# find and store the best dz
		zf=find_zshift_lines(res, flux3, refparams, zmin, zmax, delta_dz, chk_boundinds, nphot, zqso, addline=addline, divide=divide, mode=mode, fix=fix, dlam=dlam, ispec=ispec) # size: nchunk
		zfits[:,i]=zf
		dz_est = np.mean(zf)
		print(('Monte #', i, 'cost %.2f min'%((time.time()-t1)/60.), dz_est, np.abs((dz_est-real_dz)/(zmin-real_dz))))

	pkdump([nphot,zmin,zmax,delta_dz], tosave+'_params.pickle')
	pkdump(zfits, tosave+'_zfits.pickle')
	print(('Saved:', tosave+'_params _zfits.pickle'))
	return zfits

def do_lya_sims_diffaddline(nphot,zmin,zmax,delta_dz,minlinesize,prefix,res,real_dz=0.,ntrial=1,addline=250,divide=8.,mode='dz',load=False):
	'''
	add different sets of lines for each trial
	'''
	tosave = path+'paras/zlines_diffaddline_'+saveid+'_'+prefix
	if load:
		zfits = pkload(tosave+'_zfits.pickle')
		#print('Loaded:', tosave+'_zfits.pickle')
		return zfits

	zfits=[] # list with ntrial arrays each nchk size; the best fit redshift for each of 100 trials for each chunk
	for i in range(ntrial): # try 100 times
		t1=time.time()
		refparams=np.zeros([4,addline])
		for i in range(addline):
			j=np.int(round(len(fL)*np.random.uniform(0,1))) # pick one set of fL and fG at random
			k=np.int(round(len(AL)*np.random.uniform(0,1))) # pick one value of AL at random
			lam0=np.random.uniform(1120,1205) # random wavelength in this range
			refparams[:,i]=np.array([lam0s,AL[k],fL[j],fG[j]])
		# read in params, add 250 lines, add noise
		flux1, lam = mk_qso_spec3_dz(zqso,real_dz,refparams,res,addline=addline,divide=divide,mode=mode) # spec not shifted
		flux3 = add_shot_noise(flux1,nphot) # add noise
		# divide spec into chuncks
		chk_boundinds = sse.chunk_spec(lam, flux1, minlinesize=minlinesize, chunk_flux_threshold=0.975) # chunk_flux_threshold: None (gradchk), 0.8, 0.975
		# find and store the best dz
		zf = find_zshift_lines(res, flux3, refparams, zmin, zmax, delta_dz, chk_boundinds, nphot, addline=addline, divide=divide, mode=mode) # size: nchunk
		zfits.append(zf)
		dz_est = np.mean(zf)
		print(('Monte #', i, 'cost %.2f min'%((time.time()-t1)/60.), dz_est, np.abs((dz_est-real_dz)/(zmin-real_dz))))

	pkdump(zfits, tosave+'_zfits.pickle')
	print(('Saved:', tosave+'_zfits.pickle'))
	return zfits

def sigma_weightmean(arr, sigma=0., axis=None):
	'''
	sigma weighted average considering sigma==0
	arr and sigma same size
	'''
	if 0. in sigma:
		arr = np.ma.masked_where(sigma!=0., arr)
		mean = np.mean(arr, axis=axis)
		err = 0.
	else:
		w = 1./np.mean(sigma, axis=np.delete(list(range(len(arr.shape)))))
		mean = np.average(arr, axis=axis, weights=w)
		err = 1./np.sum(w)
	return mean, err

def unbiased_std(arr, axis=None):
	arr = np.array(arr)
	v = np.var(arr, axis=axis)
	if axis==None: n = np.prod(arr.shape)
	else: n = arr.shape[axis]
	return np.sqrt(v*n/(n-1.))
	#return np.std(arr,axis=axis)

if __name__ == '__main__':
	'''
	# test dz in [-1.4e-8, 1.4e-8] with 2e-10 step, lambda R=2e4
	# return dz_best, chunk bounds, line indexes
	zfs = do_lya_sims_zlines(0,5e12,-1.4e-8,1.4e-8,2e-10,3,'R20Knph5e12',2e4,real_dz=0., addline=250, divide=8., mode='dz')
	'''

	load = 1 # whether to load saved parameters or run the analysis
	ntrial = 100
	mls = 3 # 3 min line size
	nphot = 1.69e8 #1.69e8 5e12 #5e7 to give realtive flux error ~0.01
	res = 2e4 # 5e4 to give pixel scale ~0.01AA, not useful if fixres=='resele'
	z_genspec = 3. # quassar redshift for generate_spec
	test_range_factor = 50. # default 10, determines the test range [real_dz +/- test_range_factor * delta_dz]
	tradl = False # (trial add line) whether to add different sets of lines for each trial

	addnoisetxt = '_addnoise' if (ntrial == 1) else '_addnoise%dmean'%ntrial
	testrangetxt = '_testrange%dx'%test_range_factor if (test_range_factor != 10.) else ''

	# for different modes
	modes = ['dz','dl']
	mode_units = [' (cm/s)', ' (AA)']
	delta_dzs_modes = [np.logspace(-5,-10,6), np.logspace(-2,-7,6)] # for [dz, dl]
	delta_dzs_modes = [np.logspace(-9,-11,3), np.logspace(-6,-8,3)] # for [dz, dl]
	delta_dzs_modes = [np.r_[np.arange(7e-11,1.1e-10,1e-11),np.arange(2e-10,1.1e-9,1e-10),3e-9], np.logspace(-6,-8,3)] # for [dz, dl]
	delta_dzs_modes = [[7e-11,1e-10,2e-10,4e-10,7e-10,1e-9,3e-9], np.logspace(-6,-8,3)] # for [dz, dl]
	#delta_dzs_modes[0] = delta_dzs_modes[0][::-1][:5]
	delta_dzs_assign = np.array([[7e-11],[8e-11,6e-10],[9e-11,4e-10],[1e-10,3e-10,1e-9],[2e-10,5e-10,7e-10,8e-10,9e-10]])
	delta_dzs_modes = [[2e-10], np.logspace(-6,-8,3)] # for [dz, dl]

	# for dz looping regime
	dzlooptxts = ['_fixrealdz', '_looprealdz']
	dzmarkers = ['r', 'b']
	#dzloop_xlabel_preffixes = [' real', ' test scale']
	#dzloop_ylabels = ['relative error', 'best/testrange']

	# for addline looping regime
	addlines = [250, 0]
	#addlines = [0, 50, 100, 150, 200, 250, 300]
	addlinetxts = ['' if addline==0 else '_addline' if addline==250 else '_addline%d'%addline for addline in addlines]
	addlinemarkers = ['-', '--']
	# for divide lopping regime
	divides = [8., 1.]
	#divides = [1., 3., 5., 8., 10., 12.]
	dividetxts = ['' if divide==1. else '_divide%d'%int(round(divide)) for divide in divides]
	dividemarkers = ['x', 'o']
	dividemss = [20, 8]

	# line styles for legend
	lines = []
	lines.append(Line2D([0],[0], ls=addlinemarkers[0], c='k')) # addline
	lines.append(Line2D([0],[0], ls=addlinemarkers[1], c='k')) # not addline
	lines.append(Line2D([0],[0], ls='-', c='k', marker=dividemarkers[0], ms=dividemss[0])) # divide8
	lines.append(Line2D([0],[0], ls='-', c='k', marker=dividemarkers[1], ms=dividemss[1])) # not divide8
	lines.append(Line2D([0],[0], ls='-', c=dzmarkers[0])) # fixrealdz
	lines.append(Line2D([0],[0], ls='-', c=dzmarkers[1])) # looprealdz
	legends = np.hstack([addlinetxts, dividetxts, dzlooptxts])
	legends = [legend.replace('_', '') for legend in legends] # delete '_'s
	legends = ['not '+legends[i-1] if legends[i]=='' else legends[i] for i in range(len(legends))] # add 'not'

	xlabel_prefix = ' test scale'
	ylabel = 'relative error'

	#fig, ax = plt.subplots(1,2,figsize=(12,8)) # plot rela_err vs delta_dz
	#ax = ax.flatten()
	#fig1, ax1 = plt.subplots(figsize=(8,8)) # plot sqrt(1/sum(1/v)) vs R (spec res) / vs delta_dz
	step_counter = 1
	t1 = time.time()
	imod = 0; ial = 1; idv = 1; idz = 0; # default values
	# different resolutions
	reses = np.r_[1e1,5e1,1e2,3e2,5e2,1e3,3e3,5e3,7e3,np.arange(1e4,5.1e4,1e4)]
	reses = np.r_[1e3,3e3,5e3,7e3,1e4,2e4,3e4,4e4,5e4] # for intRkeck
	reses2 = np.arange(1.1e4,2e4,1e3)
	reses2 = np.arange(1.1e4,2e4,2e3)
	#reses = np.r_[reses,reses2]
	# different sets of refparas (for addline)
	i_refpar = 0
	i_refpars = list(range(1,10))[::-1]
	i_refpars = list(range(10))
	# different number of photons
	nphots = np.r_[5e7, 2e8, 5e8, 5e9, 5e10, 5e12]
	sqrtvinvsuminvs = []; sqrtvinvsuminvs_err = []
	# different zqsos for generate spec
	z_genspecs = np.arange(2.,4.1,0.2)
	z_genspecs = np.r_[2.,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,4.]
	z_genspecs = np.r_[2.,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6]
	#z_genspecs = z_genspecs[::-1]
	#for i_refpar in i_refpars: # different sets of refparas (for addline)
	#for res in reses:
	#for nphot in nphots:
	#for z_genspec in z_genspecs:
	for imod in [0]:#range(len(modes)): # dz or dl
		# zqso
		zqso = z_genspec if use_genspec else z
		# spectypetxt
		if use_voigtforest: spectypetxt = '_voigtforest' + csltxt + fitconttxt
		elif use_genspec:
			restframetxt = '_restframe' if restframe_genspec else ''
			spectypetxt = '_genspec_zqso%.1f'%z_genspec + restframetxt
			if gs.LiskeDist: spectypetxt = '_genspec_zqso%.1f_LiskeDist'%z_genspec + restframetxt
		else: spectypetxt = ''
		spectypetxt = spectypetxt #+ '_bothNoise'
		# nohpttxt
		nphottxt = '_Nphot%.0e_samerefp_trueadj_gradchk'%nphot # true adjustment by factor, adjusting by both instrumental and Keck resolution, should not divide8
		nphottxt = '_Nphot%.0e_samerefp_noadjustlw'%nphot # not adjusting any linewidth, should divide8
		nphottxt = '_Nphot%.0e_samerefp_trueadj_noadjustKecklw'%nphot # true adjustment by factor, adjusting only by instrumental resolution, should divide8
		nphottxt = '_Nphot%.0e_samerefp_trueadj_fitmoreline'%nphot # true adjustment by factor, adjusting by both instrumental and Keck resolution, should not divide8
		nphottxt = '_Nphot%.0e_samerefp_trueadj_adjsmooth'%nphot # true adjustment by factor, adjusting by both instrumental and Keck resolution, should not divide8
		nphottxt = '_Nphot%.0e_samerefp_trueadj'%nphot # true adjustment by factor, adjusting by both instrumental and Keck resolution, should not divide8
		nphottxt = '_Nphot%.2e'%nphot if nphot==1.69e8 else '_Nphot%.0e'%nphot
		nphottxt = nphottxt + spectypetxt + '_gradchk' # + '_chk0.8flux' # 
		# restxt
		#restxt = '_R%.1eintRkeck'%res if res<R_keck else '_R%.1e'%res
		if fixres == 'resele': restxt = '_dlam%.3e'%(dlam_fixresele)
		else: restxt = '_R%.1eintR%.0e'%(res,R_int) if res<R_int else '_R%.1e'%res
		restxt = restxt.replace('.0e','e')

		xlabel =  modes[imod] + xlabel_prefix + mode_units[imod]
		delta_dzs = delta_dzs_modes[imod]
		for ial in [1]:#range(len(addlines)):
			addlinetxt = addlinetxts[ial] if i_refpar==0 else addlinetxts[ial]+'_lineset%d'%i_refpar
			for idv in [1]:#range(len(divides)):
				for idz in [0]:#range(len(dzlooptxts)):
					if not load: print('-------- mode:'+modes[imod]+' addline:'+str(addlines[ial])+' divide:'+str(divides[idv])+' '+dzlooptxts[idz]+' res:'+str(res)+' --------')
					#marker = dzmarkers[idz]+addlinemarkers[ial]+dividemarkers[idv]
					t2 = time.time()
					relaerrs = []; relaerr_errs = [];
					#for delta_dz in delta_dzs_assign[4]:
					for delta_dz in delta_dzs:
						if not load: print(('delta '+modes[imod]+':', delta_dz))
						delta_dztxt = '_'+modes[imod]+'%.0e'%delta_dz

						if idz == 0: real_dz = 0. # fix real_dz at 0, loop through dz resolutions
						elif idz == 1: real_dz = delta_dz * test_range_factor # loop through real_dz
						# -------- varying test range by delta_dz scale
						zmin = real_dz - delta_dz * test_range_factor
						zmax = real_dz + delta_dz * (test_range_factor + 0.5)
						# -------- fixed test range
						zmin = -5e-8
						zmax = 5e-8 + delta_dz
						testrangetxt = '_testrange%.0e'%(np.abs(zmin-real_dz))
						# --------
						prefix = smoothtxt + fitaddlinetxt + delta_dztxt + addlinetxt + dividetxts[idv] + nphottxt + restxt + dzlooptxts[idz] + addnoisetxt + testrangetxt
						# ---------------- add different sets of lines for each trial
						if tradl: zfs_list = do_lya_sims_diffaddline(nphot,zmin,zmax,delta_dz,mls,prefix,res,real_dz,ntrial,addlines[ial],divides[idv],modes[imod],load)
						# ---------------- use same spectra for each trial
						else: zfs = do_lya_sims_zlines(zqso,nphot,zmin,zmax,delta_dz,mls,prefix,res,real_dz,ntrial,\
						      addlines[ial],divides[idv],modes[imod],load,i_refpar,fixres,dlam_fixresele,ispec=ispec)
						# ----------------

						# mask dz result data by relaerr_thrshld
						relaerr_thrshld = 0.75 # relative error threshold
						if tradl: 
							relaerr_all_list = [np.abs((real_dz-z)/(real_dz-zmin)) for z in zfs_list]
							zfs_masked_list = [np.ma.masked_where(relaerr_all > relaerr_thrshld, z) for z in zfs_list]
							relaerr_all_masked_list = [np.ma.masked_where(relaerr_all > relaerr_thrshld, relaerr) for relaerr in relaerr_all_list]
						else:
							relaerr_all = np.abs((real_dz-zfs)/(real_dz-zmin))
							zfs_masked = np.ma.masked_where(relaerr_all > relaerr_thrshld, zfs)
							relaerr_all_masked = np.ma.masked_where(relaerr_all > relaerr_thrshld, relaerr_all)
						# compute std and dropout high std
						std_perchk = unbiased_std(zfs_masked, axis=1) # size:nchk, std over ntrials
						ichk_tokeep = std_perchk<7.5e-9
						#ichk_tokeep = std_perchk<1.
						zfs_masked = zfs_masked[ichk_tokeep]
						std_perchk = std_perchk[ichk_tokeep]
						# other measures
						sigma_dz = unbiased_std(zfs_masked)
						dz_perchk = np.mean(zfs_masked, axis=1) # size:nchk, mean over ntrials
						mask_count_perchk = zfs_masked.count(axis=1) # count over ntrials
						#for isub in range(10): # divide into subgroup
						if ntrial==ntrial: # bootstrapping
							#subinds = range(isub*100,(isub+1)*100) # divide into subgroup
							#zfs_sub = zfs_masked[:,subinds] # divide into subgroup
							bs_time = 1000 # number of resamples
							sqrtvinvsuminv_pertimes = []
							for i in range(bs_time):
								inds = np.random.randint(0,ntrial,ntrial)
								#inds = np.random.randint(0,10,10) # use only first 10
								#inds = np.random.randint(0,100,100) # divide into subgroup
								std_perchk_pertime = unbiased_std(zfs_masked[:,inds], axis=1) # std over ntrials
								std_perchk_pertime = std_perchk_pertime[std_perchk_pertime>0] # filter not to divide by zero
								#std_perchk_pertime = unbiased_std(zfs_sub[:,inds], axis=1) # divide into subgroup
								sqrtvinvsuminv_pertimes.append(np.sqrt(1./np.sum(1./(std_perchk_pertime**2.))))
							sig_bs = np.mean(sqrtvinvsuminv_pertimes) # for bootstrapping
							sig_bs_err = unbiased_std(sqrtvinvsuminv_pertimes)
							sqrtvinvsuminvs.append(sig_bs)
							sqrtvinvsuminvs_err.append(sig_bs_err)
							print('sigma: %.2f +/- %.2f cm/s'%(dz2dv(sig_bs,zqso),dz2dv(sig_bs_err,zqso)))
						continue
						if np.sum(mask_count_perchk) == 0.: # all relaerr > relaerr_thrshld
							relaerr = 1.
							relaerr_err = 0.
						else:
							# ---- compute weighted average using non-masked count weights
							dz_pertrial = np.average(zfs_masked, axis=0, weights = mask_count_perchk)
							dz_err = np.sqrt(np.sum(mask_count_perchk**2. * std_perchk**2))/np.sum(mask_count_perchk)
							# ---- compute weighted average using sigma weights
							#dz_pertrial, dz_err = sigma_weightmean(zfs_masked, axis=0, sigmas=np.repeat(np.atleast_2d(std_perchk).T,zfs_masked.shape[1],axis=1))
							#relaerr_err = dz_err/np.abs(real_dz-zmin)
							relaerr_pertrial = np.abs((real_dz-dz_pertrial)/(real_dz-zmin))
							relaerr = np.mean(relaerr_pertrial)
							relaerr_err = unbiased_std(relaerr_pertrial)
						# test if median has error > relaerr_thrshld
						relaerr_all_median = np.median(relaerr_all)
						if relaerr_all_median > relaerr_thrshld: # use this relaerr_all_median as relaerr
							relaerr = relaerr_all_median
						# appending
						relaerrs.append(relaerr)
						relaerr_errs.append(relaerr_err)
						# ------------- compute sqrt(1/sum(1/v))
						sqrtvinvsuminv = np.sqrt(1./np.sum(1./(std_perchk**2.))) # steve's measure
						sqrtvinvsuminvs.append(sqrtvinvsuminv)
						print('dz:%.0e'%delta_dz,'sigma_dz:%.2e'%sigma_dz,'sigma_dz/dz:%.2f'%(sigma_dz/delta_dz),'sigma_dv:%.2f cm/s'%(sigma_dz/zqso),'vinv:%.2f cm/s'%(dz2dv(sqrtvinvsuminv,zqso)))
						#plt.hist(zfs.flatten(),bins=50);plt.show()
						#plt.hist(zfs.flatten(),bins=50,histtype='step',color='r',normed=1);plt.hist(zfs[:,0:10].flatten(),bins=50,histtype='step',color='b',normed=1); plt.show()
						#plt.hist(std_perchk,bins=20,histtype='step',color='r',normed=1);plt.hist(std_perchk_pertime,bins=20,histtype='step',color='b',normed=1);plt.show()
					#ax[imod].errorbar(dz2dv(np.array(delta_dzs),zqso), relaerrs, yerr=relaerr_errs, fmt=marker, ms=dividemss[idv])
					if not load: print('%d/16 one mode took: %.2f min'%(step_counter, (time.time()-t2)/60.))
					step_counter += 1
		#plt.errorbar(zqso, dz2dv(sig_bs,zqso), yerr=dz2dv(sig_bs_err,zqso), fmt='or') # vs zqso
		# -------- relaerr vs delta_dz
		'''
		ax[imod].set_xscale('log')
		ax[imod].set_xlabel(xlabel)
		ax[imod].set_xticks(np.arange(1e-10,1.1e-9,1e-10))
		ax[imod].set_ylabel(ylabel)
		ax[0].legend(lines, legends)
		'''
	'''
	# compare pixel-wise sigma, same settings as liske, with liske
	print('zqsos',z_genspecs)
	print('sigma',dz2dv(np.array(sqrtvinvsuminvs),zqso))
	print('liske',sigma_liske_empir(z_genspecs))
	ratio = sigma_liske_empir(z_genspecs)/dz2dv(np.array(sqrtvinvsuminvs),zqso)
	diff = sigma_liske_empir(z_genspecs)-dz2dv(np.array(sqrtvinvsuminvs),zqso)
	print('ratio',ratio)
	plt.plot(z_genspecs,ratio,'o')
	plt.plot(z_genspecs,diff,'o')
	plt.show()
	print('ratmn',np.mean(sigma_liske_empir(z_genspecs)/dz2dv(np.array(sqrtvinvsuminvs),zqso)))
	'''
	# ----- sqrtvinvsuminvs vs R ----------
	'''
	ax1.errorbar(reses, dz2dv(np.array(sqrtvinvsuminvs),zqso), yerr=dz2dv(np.array(sqrtvinvsuminvs_err),zqso), fmt='o') # vs R
	ax1.plot([2e4],[dz2dv(1.31813370475e-10,zqso)],'rx',ms=10) #steve's point
	ax1.set_xlabel('spectral resolution')
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSresIntR2e+04_bs_unilam'+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+nphottxt+restxt+testrangetxt+'.pdf'
	####### modify tauboo below
	tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSres_bs_unilam'+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+nphottxt+restxt+testrangetxt+'_tauboo.pdf'
	#tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSresIntRkeck_unilam'+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+nphottxt+restxt+testrangetxt+'.pdf'
	'''
	# ----- sqrtvinvsuminvs vs nphot ----------
	'''
	ax1.errorbar(nphots, dz2dv(np.array(sqrtvinvsuminvs),zqso), yerr=dz2dv(np.array(sqrtvinvsuminvs_err),zqso), fmt='o') # vs R
	#ax1.plot([5e12],[dz2dv(1.31813370475e-10,zqso)],'rx',ms=10) #steve's point
	ax1.set_xlabel('Number of photons')
	ax1.set_xscale('log')
	#ax1.set_yscale('log')
	if use_genspec: tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSnphot'+spectypetxt+addnoisetxt+restxt+testrangetxt+'.pdf'
	else: tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSnphot_bs_unilam'+spectypetxt+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+restxt+testrangetxt+'.pdf'
	'''
	# ----- sqrtvinvsuminvs vs zqso(z_genspec) ----------
	'''
	ax1.errorbar(z_genspecs, dz2dv(np.array(sqrtvinvsuminvs),zqso), yerr=dz2dv(np.array(sqrtvinvsuminvs_err),zqso), fmt='o') # vs zqso
	#ax1.plot([5e12],[dz2dv(1.31813370475e-10,zqso)],'rx',ms=10) #steve's point
	ax1.set_xlabel('zqso')
	#ax1.set_xscale('log')
	#ax1.set_yscale('log')
	if use_genspec: tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSzqso'+nphottxt+addnoisetxt+restxt+testrangetxt+'.pdf'
	else: tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSzqso_bs_unilam'+nphottxt+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+restxt+testrangetxt+'.pdf'
	'''
	# ----- sqrtvinvsuminvs vs same experiments ----------
	'''
	ax1.errorbar(range(len(sqrtvinvsuminvs)), dz2dv(np.array(sqrtvinvsuminvs),zqso), yerr=dz2dv(np.array(sqrtvinvsuminvs_err),zqso), fmt='o')
	ax1.axhline(y=dz2dv(1.31813370475e-10,zqso),color='r') #steve's point
	ax1.set_xlabel('experiments')
	tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSexp_bs_unilam_noaddline'+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+nphottxt+restxt+testrangetxt+'.pdf'
	'''
	# ----- sqrtvinvsuminvs vs diff line set ----------
	'''
	ax1.errorbar(i_refpars, dz2dv(np.array(sqrtvinvsuminvs),zqso), yerr=dz2dv(np.array(sqrtvinvsuminvs_err),zqso), fmt='o')
	ax1.axhline(y=dz2dv(1.31813370475e-10,zqso),color='r') #steve's point
	ax1.set_xlabel('different sets of added lines')
	tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSlineset_bs_unilam'+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+nphottxt+restxt+testrangetxt+'.pdf'
	'''
	# ----- sqrtvinvsuminvs vs delta_dz ----------
	'''
	ax1.errorbar(np.array(delta_dzs)*3e10, dz2dv(np.array(sqrtvinvsuminvs),zqso), yerr=dz2dv(np.array(sqrtvinvsuminvs_err),zqso), fmt='o')
	#ax1.scatter(np.array(delta_dzs)*3e10, dz2dv(np.array(sqrtvinvsuminvs),zqso)) # vs dz scale
	ax1.plot([dz2dv(2e-10,zqso)],[dz2dv(1.31813370475e-10,zqso)],'rx',ms=10) #steve's point
	ax1.set_xlabel(xlabel)
	#ax1.set_xlim([0, dz2dv(1.1e-9,zqso)])
	tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSdz_bs_unilam'+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+nphottxt+restxt+testrangetxt+'.pdf'
	'''
	# ----- sqrtvinvsuminvs vs divide ----------
	'''
	ax1.errorbar(divides, dz2dv(np.array(sqrtvinvsuminvs),zqso), yerr=dz2dv(np.array(sqrtvinvsuminvs_err),zqso), fmt='o') # vs divide
	ax1.plot([8.],[dz2dv(1.31813370475e-10,zqso)],'rx',ms=10) #steve's point
	ax1.set_xlabel('divide line width by')
	tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSdivide_bs_unilam'+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+nphottxt+restxt+testrangetxt+'.pdf'
	'''
	# ----- sqrtvinvsuminvs vs addline ----------
	'''
	ax1.errorbar(addlines, dz2dv(np.array(sqrtvinvsuminvs),zqso), yerr=dz2dv(np.array(sqrtvinvsuminvs_err),zqso), fmt='o') # vs divide
	ax1.plot([250],[dz2dv(1.31813370475e-10,zqso)],'rx',ms=10) #steve's point
	ax1.set_xlabel('number of added lines')
	tosave1 = path+'/plots/shiftspec_sqrtvinvsuminvVSaddline_bs_unilam'+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+nphottxt+restxt+testrangetxt+'.pdf'
	'''
	# ---------------
	figtitle = saveid+smoothtxt+addnoisetxt+nphottxt+restxt+testrangetxt
	'''
	fig.suptitle(figtitle)
	tosave = path+'/plots/shiftspec_cmperr_unilam'+addnoisetxt+'_'+saveid+smoothtxt+'_dzdl_chunkbyline'+nphottxt+restxt+testrangetxt+'.pdf'
	fig.savefig(tosave); print('Saved:',tosave)
	'''
	#ax1.set_ylabel('$\sigma$ (cm/s)')
	#fig1.suptitle(figtitle)
	#fig1.savefig(tosave1); print('Saved:',tosave1)
	#plt.show()
	print('Total time: %.2f min'%((time.time()-t1)/60.))
'''
	# statistical properties of best dzs found
	b1=np.mean(zfs,axis=1)
	b2=np.mean(zfs**2, axis=1)
	v=b2-(b1**2) # variance
	sd=v**0.5 # standard deviation
	#print(sd)
	x=np.arange(len(v))
	plt.scatter(x,np.log10(1./sd))
	#plt.scatter(x,np.log10(v))
	plt.show()
	print(np.min(v))
	vinv=1./v
	j=[k for k in range(len(sd)) if sd[k]<7e-9 and sd[k]>1e-11]
	dum1=np.sum(vinv[j])
	print((1./dum1)**0.5)
'''

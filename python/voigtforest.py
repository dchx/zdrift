import time,pickle,gzip,os
import numpy as np
import lmfit as lf
import matplotlib.pyplot as plt
import astropy.constants as c
import vamp
from astropy.modeling.models import Voigt1D
import generate_spec as gs
from utils import *

# settings
voigt_is_tau = 1 # whether to treat sum(Voigt1Ds) as tau and flux=exp(-tau)
para_set = 'fLfG' # 'blgNHI' (used in generate_spec.voigt1d) or 'fLfG' (used in astropy.Voigt1D)
# derived
if para_set=='blgNHI': voigt_is_tau = True

# spectrum shift transformation
def dvel2dlam(dvel, lam0, unit='km/s'): return lam0 * dvel / c.c.to(unit).value
def dlam2dvel(dlam, lam0, unit='km/s'): return c.c.to(unit).value * dlam / lam0
# line width transformation
def fwhm2b(fwhm): return fwhm / 2. / np.sqrt(np.log(2.))
def b2fwhm(b): return 2.*np.sqrt(np.log(2.)) * b

def fv(fl, fg):
	'''
	compute fwhm for Voigt profile (fv) from fwhms of Lorentz (fl) and Gaussian (fg)
	'''
	return 0.5346*fl+np.sqrt(0.2166*fl**2.+fg**2.)

def lam_space(lam_start, lam_end, R=35800., pix_per_R=2.):
	'''
	Generate a sequence of wavelengths
	'''
	lam = lam_start
	lams = []
	while lam <= lam_end:
		lams.append(lam)
		dlam = lam / R
		pixscale = dlam / pix_per_R 
		lam += pixscale
	return np.array(lams)

typical_lw = 20. # km/s, typical fwhm line width
typical_lw = 50. # km/s, typical fwhm line width

def generate_test_spec(lam_start=1000., lam_end=1010., nlines=10, R=35800., pix_per_R=2., Nphot=1e3, linewidth=typical_lw):
	'''
	linewidth - (km/s) one value
	'''
	lam = lam_space(lam_start, lam_end, R, pix_per_R)
	lam0s = np.random.uniform(min(lam),max(lam),nlines)
	lam0s_mid = np.sort(lam0s)[int(round(nlines/2))]
	lam0s = np.r_[lam0s, lam0s_mid + 5*dvel2dlam(linewidth, lam0s_mid)]
	lam0s = np.r_[lam0s, lam0s_mid - 5*dvel2dlam(linewidth, lam0s_mid)]
	flux_noerr = np.zeros(len(lam))
	for lam0 in lam0s:
		lw_lam = dvel2dlam(linewidth, lam0) # typical linewidth in wavelength
		#lw_lam = np.abs(np.random.normal(lw_lam, np.sqrt(lw_lam))) # make it random
		if para_set=='fLfG': flux_noerr += Voigt1D(x_0=lam0, amplitude_L=1., fwhm_L=lw_lam, fwhm_G=lw_lam)(lam) # positive
		elif para_set=='blgNHI': flux_noerr += gs.voigt1d(lam0, lgNHI=13., b=lw_lam)(lam)
	cont = 1.
	flux_noerr = gs.form_spec(cont, flux_noerr, voigt_is_tau=voigt_is_tau)
	if min(flux_noerr) < 0: flux_noerr = (flux_noerr - min(flux_noerr))/(1. - min(flux_noerr)) # scale to [0,1]
	# add noise
	flux, noise = add_shot_noise(flux_noerr, Nphot, return_error=True)
	return lam, flux, noise

if para_set=='fLfG':
	para_prefix = ['lam0_', 'AL_', 'fL_', 'fG_']
	v1d = Voigt1D
elif para_set=='blgNHI':
	para_prefix = ['lam0_', 'lgNHI_', 'b_']
	v1d = gs.voigt1d
def para_list(p, paraind=0):
	'''
	Convert lf.Parameters() to parameter list (len=4)
	paraind - index of one line's parameter
	'''
	s = str(int(round(paraind)))
	plist = [p[prefix+s].value for prefix in para_prefix]
	return plist

def singlevoigt_paras(p, lam, paraind=0):
	args = para_list(p, paraind)
	return v1d(*args)(lam)

def num_paras(paras): return len([key for key in paras.keys() if para_prefix[0] in key])

def multivoigt_paras(p, lam):
	parray = paras2parray([p])
	return gs.multivoigt_parray(parray, lam, v1d=v1d)

def voigt_residual(p, voigtfunc, lam, flux, noise, continuum=1., *args):
	model = gs.form_spec(continuum, voigtfunc(p, lam, *args), voigt_is_tau=voigt_is_tau)
	res = (flux - model) / noise
	return res

def addaline(paras, il, lam, flux, ipeak, continuum=1.):
	if type(continuum) != np.ndarray or type(continuum) != list: continuum = continuum * np.ones(len(lam)) # make continuum an array
	lam0 = lam[ipeak]
	typical_lamwidth = dvel2dlam(typical_lw, lam0)
	min_lw = 0. #8.
	min_lamwidth = dvel2dlam(min_lw, lam0) # width to be greater than resolution
	max_lw_factor = 5. # 20.
	paras.add(para_prefix[0]+str(il), value=lam0, min=lam0-typical_lamwidth, max=lam0+typical_lamwidth) # lam0
	if para_set=='fLfG':
		if voigt_is_tau: paras.add(para_prefix[1]+str(il), value=-1.5*(np.log((flux[ipeak] if flux[ipeak]>0 else 1e-20)/continuum[ipeak])), min=0., max=2e2) # AL as tau_0
		else: paras.add(para_prefix[1]+str(il), value=-1.5*(flux[ipeak]-continuum[ipeak]), min=0., max=1.5) # AL, positive
		paras.add(para_prefix[2]+str(il), value=0.5*typical_lamwidth, min=min_lamwidth, max=max_lw_factor*typical_lamwidth) # fL
		paras.add(para_prefix[3]+str(il), value=0.5*typical_lamwidth, min=min_lamwidth, max=max_lw_factor*typical_lamwidth) # fG
	elif para_set=='blgNHI':
		paras.add(para_prefix[1]+str(il), value=13., min=12., max=16.) # lgNHI
		paras.add(para_prefix[2]+str(il), value=0.5*typical_lw, min=min_lw, max=max_lw_factor*typical_lw) # b

def aicc(fitresult):
	'''
	Compute AICC: AIC with Correction for small sample size
	'''
	p = fitresult.nvarys # number of parameters
	n = fitresult.ndata
	aicc = fitresult.aic + 2*p*(p+1.)/(n-p-1.)
	return aicc

def fit_region(lam, flux, noise, ipeaks, continuum=1., addline=True):
	'''
	Fit absorption lines in a region
	ipeaks - (array) indexes of lam for line peaks
	addline - whether try to add lines after fit
	'''
	# initialize parameters
	paras = lf.Parameters()
	for il in range(len(ipeaks)): addaline(paras, il, lam, flux, ipeaks[il], continuum=continuum)
	flux_guess = gs.form_spec(continuum, multivoigt_paras(paras, lam), voigt_is_tau=voigt_is_tau)
	# fit voigt for this region
	if len(flux) <= len(paras): return None # must have ndata > npara for leastsq
	fitresult = lf.minimize(voigt_residual, paras, args=(multivoigt_paras, lam, flux, noise, continuum))
	fitresult.initparams = paras

	# decide whether to add line
	if addline:
		max_fails = 2 # how many times of fails allowed for adding a line at a time
		fails = 0
		if len(ipeaks) == 0: il = -1 # no line detected in region
		while fails < max_fails: # add a line at a time
			il += 1
			addaline(paras, il, lam, flux, len(lam)/2, continuum=continuum)
			paras[para_prefix[0]+str(il)].set(value=np.mean(lam), min=min(lam), max=max(lam)) # adjust lam0 to be mean(lam)
			if para_set=='fLfG': paras[para_prefix[1]+str(il)].set(value=(np.mean(flux)-np.mean(continuum))) # adjust AL to be mean(flux-cont)
			if len(flux) - len(paras) <= 1: break # must have ndata - npara > 1 for aicc
			print('\tTrying to add line %d ...'%(il+1)),
			addlineresult = lf.minimize(voigt_residual, paras, args=(multivoigt_paras, lam, flux, noise, continuum))
			addlineresult.initparams = paras
			if aicc(addlineresult) < aicc(fitresult):
				fitresult = addlineresult
				fails = 0
				print('success')
			else:
				fails += 1
				print('fail')

	flux_fit = gs.form_spec(continuum, multivoigt_paras(fitresult.params, lam), voigt_is_tau=voigt_is_tau)
	plot = 0
	if plot:
		plt.plot(lam, flux_fit, 'r', lw=0.5)
		plt.plot(lam, flux_guess, 'b', lw=0.5)
		plt.plot(lam, flux, 'k', lw=0.5)
		plt.plot(lam[ipeaks], flux[ipeaks], 'vb')
		plt.show()
	return fitresult

def paras2parray(paralist):
	'''
	Inputs: paralist - list of lf.Parameters
	Outputs: parray - parameter array, dim:[nparas, nlines]
	'''
	parray = []
	for paras in paralist:
		npara = num_paras(paras)
		for il in range(npara): parray.append(para_list(paras, il))
	return np.array(parray).T

def results2parray(results): return paras2parray([result.params for result in results]) # parameters array
def results2initparray(results): return paras2parray([result.initparams for result in results]) # inital parameters array
def pfile2parray(pfile): return results2parray(pkloadgzip(pfile))
def pfile2flux(pfile, lam, continuum=1.): return gs.parray2flux(pfile2parray(pfile), lam, continuum, voigt_is_tau=voigt_is_tau, v1d=v1d)

def fit_forest(lam, flux, noise, continuum=1., tosave=None, addline=True, CSLcut=1.5, plot=False, chkind=False, verbose=True):
	'''
	Spectrum should be normalized to [0,1]
	addline - whether try to add lines after fit
	chkind - whether to return region_indlims from find_regions
	'''
	# ----------- divide spectra to regions -----------
	tosave_findreg = tosave[:tosave.rfind('.')] + '_findreg' + tosave[tosave.rfind('.'):]
	typical_pixwidths = dvel2dlam(typical_lw, lam) / np.gradient(lam) # typical linewidth in pixels for each lam
	max_pixwidth = np.mean(typical_pixwidths)+3*np.std(typical_pixwidths) # 3 sigma upper bound
	region_lamlims, region_indlims, region_indpks = \
	    vamp.find_regions(lam, flux, noise, continuum=0.99*continuum, extend=True, \
	    peak_dist=1, N_sigma=CSLcut, max_pixwidth=max_pixwidth, plot=0, tosave=tosave_findreg) # [[lam_start, lam_end]], [[i_start, i_end]] (flux[i_start:i_end])
	'''
	if np.sum([len(ipk) for ipk in region_indpks]) < 100:
		print('Number of lines less than 100, pass.')
		return None
	'''
	if plot: # plot spectrum, regions and line peaks
		plt.plot(lam, flux, 'k', lw=0.5)
		plt.plot(lam, noise, 'g', lw=0.5)
		for ireg in range(len(region_lamlims)):
			plt.fill_betweenx([0,continuum],region_lamlims[ireg][0],region_lamlims[ireg][1],alpha=0.3,color='y')
			plt.plot(lam[region_indlims[ireg][0]:region_indlims[ireg][1]][region_indpks[ireg]],\
			         flux[region_indlims[ireg][0]:region_indlims[ireg][1]][region_indpks[ireg]], 'vb')

	# ----------- fit voigts for each region -----------
	if tosave and os.path.exists(tosave):
		results = pkloadgzip(tosave, verbose=verbose)
	else:
		results = []
		t1 = time.time()
		for ireg, [start, end] in enumerate(region_indlims): # loop through each region
			tosave_reg = tosave[:tosave.rfind('.')]+'_reg%d'%ireg+tosave[tosave.rfind('.'):]
			if tosave and os.path.exists(tosave_reg):
				result_reg = pkloadgzip(tosave_reg, verbose=verbose)
			else:
				lam_reg = lam[start:end]
				flux_reg = flux[start:end]
				flux_reg = np.ma.masked_where(flux_reg < 0., flux_reg) # mask negative values
				noise_reg = noise[start:end]
				ipeaks_reg = region_indpks[ireg]
				if len(ipeaks_reg) == 0: continue
				#         fit continuum
				#ppoly = fit_poly([lam_reg, flux_reg], poly_deg=3)
				#continuum = np.polyval(ppoly, lam_reg)
				#         fit line
				print('Fitting %d lines for region %d/%d ...'%(len(ipeaks_reg),ireg+1,len(region_indlims)))
				t2 = time.time()
				result_reg = fit_region(lam_reg, flux_reg, noise_reg, ipeaks_reg, continuum=continuum, addline=addline)
				print('%.2f minutes. Total %.2f minutes.'%((time.time()-t2)/60., (time.time()-t1)/60.))
				if tosave: pkdumpgzip(result_reg, tosave_reg)
			if result_reg != None: results.append(result_reg)
			plotreg = 0
			if plotreg:
				lam_reg = lam[start:end]
				flux_reg = flux[start:end]
				plt.plot(lam_reg, flux_reg, 'k', lw=0.5)
				flux_fit = gs.parray2flux(results2parray([result_reg]), lam_reg, continuum=continuum, voigt_is_tau=voigt_is_tau, v1d=v1d)
				plt.plot(lam_reg, flux_fit, 'r', lw=0.5)
				plt.show()
		if tosave: pkdumpgzip(results, tosave)

	# ----------- analysis -----------
	if plot: # plot fitted flux
		bestp_tot_arr = results2parray(results)
		lam0, al, fl, fg = bestp_tot_arr
		al_cut = -1.2
		fg_cut = dvel2dlam(20., lam0) # AA
		import sse_lya_sims_zlines_27jun2019_steve as sse
		fv_intrin, fv_para = sse.intrinsic_fwhm(fl, fg, lam0, sse.smooth)
		fwhm_intrin = dlam2dvel(fv_intrin, lam0)
		fwhm_para = dlam2dvel(fv_para, lam0)
		fwhm_cut = 20. # km/s
		initp_tot_arr = results2initparray(results)
		flux_fit = gs.parray2flux(bestp_tot_arr, lam, continuum=continuum, voigt_is_tau=voigt_is_tau, v1d=v1d)
		flux_guess = gs.parray2flux(initp_tot_arr, lam, continuum=continuum, voigt_is_tau=voigt_is_tau, v1d=v1d)
		#    plot fitted spec
		lam = obs_frame(lam, sse.z)
		plt.figure(figsize=(12,3))
		plt.axhline(1,color='k') # continuum
		plt.plot(lam, flux, 'k.', ms=3)#, lw=0.5)
		plt.plot(lam, flux_fit, 'r')
		#plt.plot(lam, flux_guess, 'b', lw=0.5)
		#plt.plot(lam, flux - flux_fit - 0.5, 'k', lw=0.5) # residuals
		#plt.axhline(-0.5,color='k',lw=0.5) # residual zero-points
		#    plot linewidth hist
		'''
		plt.figure()
		plt.hist(fwhm_intrin, bins=np.arange(0,max(fwhm_intrin)+1,1), histtype='step', label='intrinsic')
		#plt.hist(fwhm_para[fwhm_para>fwhm_cut], color = 'r', bins=np.arange(0,max(fwhm_para)+1,1), histtype='step')
		plt.hist(fwhm_para, bins=np.arange(0,max(fwhm_para)+1,1), histtype='step', label='observed')
		plt.legend()
		print('median observed line width: %.2f km/s'%np.median(fwhm_para))
		plt.xlabel('line width FWHM (km/s)')
		'''
		plt.axis([4800.,5000.,-0.1,1.1])
		plt.xlabel('$\lambda$ ($\AA$)')
		plt.ylabel('Normalized flux')
		plt.tight_layout()

	if chkind: return results, region_indlims
	else: return results

if __name__ == '__main__':
	### generate test spectra
	#lam, flux, noise = generate_test_spec(nlines=50)
	### use keck spectra
	from norm_koa import koa_normed_spec
	from continuum_fit import get_keck_spec
	import sse_lya_sims_zlines_27jun2019_steve as sse
	ind = int(sse.saveid[:2])
	#koa_spec = koa_normed_spec(ind)
	koa_spec = get_keck_spec(ind, local_dist=sse.fitcont_dist, poly_deg=sse.fitcont_deg, fitcont_mode=sse.fitcont_mode)
	lam = koa_spec[0]; flux = koa_spec[1]; noise = koa_spec[2]
	'''
	plt.axhline(1,c='k')
	plt.plot(lam,flux)
	plt.show()
	'''

	# fit or load voigt parameters
	plot = 1
	results = fit_forest(lam, flux, noise, tosave=sse.voigtforest_pfile, addline=sse.fitaddline, CSLcut=sse.CSL_cut, plot=plot)
	tosave = path + 'plots/voigtforest_spec_%s.pdf'%sse.saveid
	#plt.savefig(tosave);print('Saved:%s'%tosave)
	#plt.show()

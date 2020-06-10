from utils import *
from cosmology import dzdt,dz2dv,liske_cosmo
from generate_spec import generate_spec
from sse import two_epoch_templates, dvfits2sigma

def spec_two_epoch(nphot, z_genspec=None, period=10., addnoise=2, cosmo=liske_cosmo, ispec=None):
	'''
	give one spectral template, output two spectra at different epochs
	period - (year) time between two epochs
	addnoise - (int) 0: donot add noise; 1: add noise to 1st epoch; 2: add noise to both epochs
	'''
	linelistmode = 'keck' if z_genspec==None else 'genspec'
	class genspec_args:
		zqso = z_genspec
		ispec = ispec
	class keck_args:
		ind = 7
		fitcont_dist = 100
		fitcont_deg = 10
		fitcont_mode = 'poly'
		vfaddline = False
		CSL_cut = 1.5
		smooth = True
	lam, flux1, flux2 = two_epoch_templates(linelistmode, 'resele', 2e4, 0.0125, 'dt', period, genspec_args, keck_args, cosmo)

	if addnoise >= 1: flux1, error1 = add_shot_noise(flux1, nphot, return_error=1)
	if addnoise == 2: flux2, error2 = add_shot_noise(flux2, nphot, return_error=1)
	# return
	if addnoise == 0: return lam, flux1, flux2
	elif addnoise == 1: return lam, flux1, flux2, error1
	elif addnoise == 2: return lam, flux1, flux2, error1, error2

def liske_dvi(lam, flux1, flux2):
	'''
	compute dv_i (for each pixel i) by Liske208
	flux1, flux2 are observations in two epochs
	'''
	slope = np.gradient(flux1, lam)
	dvi = c.c.to('cm/s').value * (flux1 - flux2) / (slope * lam)
	return dvi

def liske_dvfits(ntrial, nphot, z_genspec, period, ispec=None):
	'''
	compute dvfits for ntrials
	dvfits - [npixel, ntrial]
	'''
	dvfits = []
	for itrial in range(ntrial):
		specpack = spec_two_epoch(nphot, z_genspec=z_genspec, period=period, addnoise=2, ispec=ispec)
		dvi = liske_dvi(*specpack[:3])
		dvfits.append(dvi) # shape:(ntrial, npixel)
	return np.array(dvfits).T # shape:(npixel, ntrial)

def slope_estimate(lam, flux, flux_err):
	'''
	assuming uniform lam grid
	'''
	slope = np.gradient(flux, lam)
	# compute slope error
	err2 = flux_err**2. # error square
	slope_err = np.sqrt(err2[2:] + err2[:-2]) / (lam[2:] - lam[:-2])
	slope_err = np.insert(slope_err, 0, np.sqrt(err2[0] + err2[1]) / (lam[1] - lam[0])) # first point
	slope_err = np.append(slope_err, np.sqrt(err2[-1] + err2[-2]) / (lam[-1] - lam[-2])) # last point
	return slope, slope_err

def overall_sigma(sig2s):
	'''
	sig2s - (Npixel,) array of individual sigma^2s
	Return
	sigma - 1 value
	'''
	sigma = np.sqrt(1./np.nansum(1./sig2s)) # overall sigma
	return sigma

def liske_sigma(lam, flux1, flux2, error1, error2):
	'''
	compute sigma by Liske2008
	flux1, flux2 are observations in two epochs
	'''
	slope, slope_err = slope_estimate(lam, flux1, error1)
	# compute sigma
	#### liske paper
	left = (c.c.to('cm/s').value / lam / slope)**2.
	right = error1**2. + error2**2. + (flux1 - flux2)**2. / slope**2. * slope_err**2.
	sig2s = left * right # sigma_i^2
	sigma = overall_sigma(sig2s) # overall sigma
	#print('paper',sigma)
	#### hand calculation
	'''
	left = (c.c.to('cm/s').value / lam)**2.
	right1 = (error1**2. + error2**2.) / (flux2 - flux1)**2.
	right2 = (slope_err / slope)**2.
	sig2s = left * (right1 + right2)
	sigma = np.sqrt(1./np.nansum(1./sig2s)) # overall sigma
	print('calculation',sigma)
	'''

	#print('err2s',error1**2. + error2**2.)
	#print('slopes',(flux2 - flux1)**2. / slope**2. * slope_err**2.)
	#print('1/flux diff2',1./(flux2 - flux1)**2.)
	#print('1/slope2',1./slope**2.)
	#print('slope_err2',slope_err**2.)
	#print('right',right)
	#print('slope snr',slope/slope_err)
	return sigma

def sigma_liske_empir(zqso, snr=1.3e4, nqso=1):
	'''
	Empirical sigma expression in Liske 2008
	output sigma (cm/s)
	'''
	sigma = 2.*(2370./snr) * (30./nqso)**0.5 * (5./(1.+zqso))**1.7
	return sigma

# --- multiple epochs ---
def wa(x, flux_err):
	'''
	weighted average. eq 19 of Liske2008
	x - (Nepoch, ?) to be averaged
	flux_err - (Nepoch, Npixel)
	Returns
	xbar - (Npixel,)
	'''
	if len(x.shape)==1: x = np.atleast_2d(x).T # reshape x from (Nepoch,) to (Nepoch, 1)
	sigmaN2 = flux_err ** -2.
	xbar = np.sum(x * sigmaN2, axis=0) / np.sum(sigmaN2, axis=0) # shape (Npixel,)
	return xbar

def mi_estimate(flux, flux_err, dt):
	'''
	eq 18 of Liske2008
	flux, flux_err - (Nepoch, Npixel)
	dt - (Nepoch,) or (Nepoch, 1), sequence of dts, unit should be internally consistent
	Returns
	mi - (Npixel,)
	'''
	if len(dt.shape)==1: dt = np.atleast_2d(dt).T # reshape dt from (Nepoch,) to (Nepoch, 1)
	dtbar = wa(dt, flux_err)
	above = wa(flux * dt, flux_err) - wa(flux, flux_err) * dtbar
	below = wa(dt ** 2., flux_err) - dtbar ** 2.
	mi = above / below
	return mi

def sigma_mi2(flux_err, dt):
	'''
	square of sigma_mi. eq 20 of Liske2008
	flux_err - (Nepoch, Npixel)
	dt - (Nepoch,) or (Nepoch, 1), sequence of dts, unit should be internally consistent
	Return
	sigma_mi2 - (Npixel,)
	'''
	if len(dt.shape)==1: dt = np.atleast_2d(dt).T # reshape dt from (Nepoch,) to (Nepoch, 1)
	sigmaN2 = flux_err ** -2.
	dtbar = wa(dt, flux_err)
	dtfactor = wa(dt ** 2., flux_err) - dtbar ** 2. # shape (Npixel,)
	sigma_mi2 = 1./np.sum(sigmaN2 * dtfactor, axis=0) # shape (Npixel,)
	return sigma_mi2

def sigma_vdoti2(lam, flux, flux_err, dt):
	'''
	eq 21 of Liske2008
	lam - (Npixel,)
	flux, flux_err - (Nepoch, Npixel)
	dt - (Nepoch,) or (Nepoch, 1), sequence of dts, unit should be internally consistent
	'''
	if len(dt.shape)==1: dt = np.atleast_2d(dt).T # reshape dt from (Nepoch,) to (Nepoch, 1)
	flux1 = flux[0, :]
	error1 = flux_err[0, :]
	slope, slope_err = slope_estimate(lam, flux1, error1) # (Npixel,)
	left = (c.c.to('cm/s').value / lam / slope)**2. # (Npixel,)
	right = sigma_mi2(flux_err, dt) + mi_estimate(flux, flux_err, dt)**2. / slope**2. * slope_err**2. # (Npixel,)
	sigma_vdoti2 = left * right # (Npixel,)
	return sigma_vdoti2

def sigma_multiepoch(lam, flux, flux_err, dt):
	'''
	lam - (Npixel,)
	flux, flux_err - (Nepoch, Npixel)
	dt - (Nepoch,) or (Nepoch, 1), sequence of dts, unit should be internally consistent
	'''
	sig2_vdoti = sigma_vdoti2(lam, flux, flux_err, dt) # (Npixel,)
	sig_vdot = overall_sigma(sig2s) # 1 value
	dt0 = dt[-1] # 1 value
	sig_v = sig_vdot * dt0 # 1 value
	return sig_v

if __name__ == '__main__':
	z_genspec = None # if ==None then use keck line list
	time = 10. # year
	nphot = 1.69e8
	ispec = None
	ntrial = 100
	sigma_mode = 'estimate' # 'estimate' to estimate from dvfits array or 'compute' to compute from Liske equation

	zqsos = np.array([2., 2.5, 3., 3.5, 4.])
	liskepaper_sigma = sigma_liske_empir(zqsos)
	sig2zqso = []
	sig2zqso_err = []
	sigmas = []

	# compute dv
	#for z_genspec in zqsos:
	#for ispec in range(10): # mean of 10 different sets of spectral parameters
	if 1:
		if sigma_mode=='compute': # compute sigma from equation in Liske2008
			specpack = spec_two_epoch(nphot, z_genspec=z_genspec, period=time, addnoise=2, ispec=ispec)
			sigma = liske_sigma(*specpack)
		elif sigma_mode=='estimate': # estimate sigma from computed dvi for ntrials
			dvfits = liske_dvfits(ntrial, nphot, z_genspec, period=time, ispec=ispec)
			sigma = dvfits2sigma(dvfits, relaerr_all=None, relaerr_thrshld=1., dvstd_thrshld=np.inf, bs_time=1)
		print('z: %s ispec: %s sigma: %.4f cm/s'%(z_genspec, ispec, sigma))
		sigmas.append(sigma)
	sig2zqso.append(np.mean(sigmas))
	sig2zqso_err.append(np.std(sigmas,ddof=1)) # unbiased std
	#plt.errorbar(zqsos, sig2zqso, yerr=sig2zqso_err)
	# used to hand plot
	print('====== Results: ======')
	print('zqso',zqsos)
	print('sigma',sig2zqso)
	print('sigma err',sig2zqso_err)
	print('lower',np.array(sig2zqso)-np.array(sig2zqso_err))
	print('upper',np.array(sig2zqso)+np.array(sig2zqso_err))
	print('liske',liskepaper_sigma)
	print('ratio',liskepaper_sigma/np.array(sig2zqso))
	print('ratio mean',np.mean(liskepaper_sigma/np.array(sig2zqso)))
	plt.xlabel('zqso')
	plt.ylabel('sigma (cm/s)')
	#plt.ylim([1.5,5.8])
	tosave = path + '/plots/sigmaVSzqso_genspec_nphot%.2e_epoch%dyr.pdf'%(nphot,time)
	#plt.savefig(tosave);print('Saved: %s'%tosave)
	#plt.show()

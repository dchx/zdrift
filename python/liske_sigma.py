from utils import *
from cosmology import dzdt,dz2dv
from generate_spec import generate_spec

class liske_cosmo:
	Om0 = 0.3
	Ode0 = 0.7
	H0 = 70. * u.km/u.Mpc/u.s

def getspec(dz, z_genspec=None, ispec=None):
	'''
	get one spectra nomatter what method
	ispec: index of spectra, used in generate_spec
	'''
	import sse_lya_sims_zlines_27jun2019_steve as sse
	if z_genspec != None: lam, flux = generate_spec(z_genspec, res=2e4, dlam=0.0125, fix='resele', dz=dz, rest_frame=False, shiftmode='dz', ispec=ispec, verbose=True)
	else: flux, lam = sse.mk_qso_spec3_dz(sse.z,dz,refparams=np.zeros([4,0]),res=2e4,addline=0,divide=1.,mode='dz')
	return lam, flux

def spec_two_epoch(nphot, z_genspec=None, period=10., addnoise=2, cosmo=liske_cosmo, ispec=None):
	'''
	give one spectral template, output two spectra at different epochs
	period - (year) time between two epochs
	addnoise - (int) 0: donot add noise; 1: add noise to 1st epoch; 2: add noise to both epochs
	'''
	from sse_lya_sims_zlines_27jun2019_steve import z as ssez
	zqso = z_genspec if z_genspec != None else ssez
	dz = dzdt(zqso, cosmo) * period
	lam, flux1 = getspec(0., z_genspec, ispec)
	_, flux2 = getspec(dz, z_genspec, ispec)
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

def liske_sigma(lam, flux1, flux2, error1, error2):
	'''
	compute sigma by Liske2008
	flux1, flux2 are observations in two epochs
	'''
	dlam = np.diff(lam)[0]
	slope = np.gradient(flux1, lam)
	# compute slope error
	err2 = error1**2.
	slope_err = np.sqrt(err2[2:] + err2[:-2]) / (2. * dlam)
	slope_err = np.insert(slope_err, 0, np.sqrt(err2[0] + err2[1]) / dlam) # first point
	slope_err = np.append(slope_err, np.sqrt(err2[-1] + err2[-2]) / dlam) # last point
	# compute sigma
	#### liske paper
	left = (c.c.to('cm/s').value / lam / slope)**2.
	right = error1**2. + error2**2. + (flux1 - flux2)**2. / slope**2. * slope_err**2.
	sig2s = left * right # sigma_i^2
	sigma = np.sqrt(1./np.nansum(1./sig2s)) # overall sigma
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

if __name__ == '__main__':
	from sse import dvfits2sigma
	z_genspec = 3. # if ==None then use keck line list
	time = 10. # year
	nphot = 1.69e8
	ispec = None
	ntrial = 100
	sigma_mode = 'compute' # 'estimate' to estimate from dvfits array or 'compute' to compute from Liske equation

	zqsos = np.array([2., 2.5, 3., 3.5, 4.])
	liskepaper_sigma = sigma_liske_empir(zqsos)
	sig2zqso = []
	sig2zqso_err = []
	sigmas = []

	# compute dv
	from sse_lya_sims_zlines_27jun2019_steve import z as ssez
	zqso = z_genspec if z_genspec != None else ssez
	dz = dzdt(zqso, liske_cosmo) * time
	dv = dz2dv(dz, zqso); print('real dv: %.3f cm/s'%dv)
	#for z_genspec in zqsos:
	#for ispec in range(10): # mean of 10 different sets of spectral parameters
	if 1:
		if sigma_mode=='compute': # estimate sigma from computed dvi for ntrials
			specpack = spec_two_epoch(nphot, z_genspec=z_genspec, period=time, addnoise=2, ispec=ispec)
			sigma = liske_sigma(*specpack)
		elif sigma_mode=='estimate': # compute sigma from equation in Liske2008
			dvfits = liske_dvfits(ntrial, nphot, z_genspec, period=time, ispec=ispec)
			sigma = dvfits2sigma(dvfits, relaerr_all=None, relaerr_thrshld=1., dvstd_thrshld=np.inf, bs_time=1)
		print('z: %.1f ispec: %s sigma: %.4f cm/s'%(z_genspec, ispec, sigma))
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

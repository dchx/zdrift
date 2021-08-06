from utils import *
from spec_utils import *
import spec_utils as su
import cosmology as cs
import generate_spec as gs
import flux_unit_change as fuc
import sse

def spec_two_epoch(nphot=None, fluxpars=None, z_genspec=None, period=10., addnoise=2, cosmo=cs.liske_cosmo, ispec=None, aperture=None, exptime=None):
	'''
	give one spectral template, output two spectra at different epochs
	nphot or fluxpars - atleast have one
	period - (year) time between two epochs
	addnoise - (int) 0: donot add noise; 1: add noise to 1st epoch; 2: add noise to both epochs
	'''
	linelistmode = 'keck' if z_genspec==None else 'genspec'
	class genspec_args:
		zqso = z_genspec
		ispec = ispec
	class keck_args:
		item = re.top10_df().iloc[0] # pd.Series
		fitcont_dist = 100
		fitcont_deg = 10
		fitcont_mode = 'poly'
		vfaddline = False
		CSL_cut = 1.5
		smoothwidth = 2
	lam, flux1, flux2 = sse.two_epoch_templates(linelistmode, 'resele', 2e4, 0.0125, 'dt', period, genspec_args, keck_args, cosmo)

	if type(fluxpars)!=type(None): # make nphot the continuum shape
		contflux = cf.fluxpars2flux(fluxpars, lam)
		nphot = fuc.flux2nphot(lam, contflux, aperture, exptime)
	if addnoise >= 1: flux1, error1 = su.add_shot_noise(flux1, nphot, return_error=1)
	if addnoise == 2: flux2, error2 = su.add_shot_noise(flux2, nphot, return_error=1)
	# return
	if addnoise == 0: return lam, flux1, flux2
	elif addnoise == 1: return lam, flux1, flux2, error1, np.zeros(len(flux2))
	elif addnoise == 2: return lam, flux1, flux2, error1, error2

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

def liske_dvi(lam, flux1, flux2, error1=None, error2=None):
	'''
	compute dv_i and its uncertainties sigma_dvi (for each pixel i) by Liske208
	flux1, flux2 are observations in two epochs
	'''
	if not np.any(error1): # make zero errors
		error1 = np.zeros(len(flux1))
		error2 = np.zeros(len(flux2))
	slope, slope_err = slope_estimate(lam, flux1, error1)
	cslopelam = c.c.to('cm/s').value / (slope * lam) # c / (slope*lam)
	fluxdiff = flux1 - flux2
	# compute dv
	dvi = cslopelam * fluxdiff
	# compute sigma2
	left = cslopelam**2.
	right = error1**2. + error2**2. + fluxdiff**2. / slope**2. * slope_err**2.
	sig2s = left * right # sigma_i^2
	return dvi, sig2s # dvi, (sigma_dvi)^2. shape:npixel

def cosmomc_product(lam, dv, dv_err, period, tosave, fmt='dv'):
	'''
	generate zdrift observations data for CosmoMC
	fmt - dv or dz
	output columns: z, z_err, dt, dt_err, dv(dz), dv_err(dz_err)
	'''
	zs = su.lam2z(lam)
	if fmt=='dz':
		dz = cs.dv2dz(dv, zs)
		dz_err = cs.dv2dz(dv_err, zs)
	zeros = np.zeros(len(zs))
	period = np.ones(len(zs)) * period
	if fmt=='dv':   out = np.vstack([zs, zeros, period, zeros, dv, dv_err]).T
	elif fmt=='dz': out = np.vstack([zs, zeros, period, zeros, dz, dz_err]).T
	np.savetxt(tosave, out, fmt='%.12e'); print('Saved: %s'%tosave)

def liske_dvfits(ntrial, nphot, z_genspec, period, ispec=None):
	'''
	compute dvfits for ntrials
	dvfits - [npixel, ntrial]
	'''
	dvfits = []
	for itrial in range(ntrial):
		specpack = spec_two_epoch(nphot, z_genspec=z_genspec, period=period, addnoise=2, ispec=ispec)
		dvi, _ = liske_dvi(*specpack[:3])
		dvfits.append(dvi) # shape:(ntrial, npixel)
	return np.array(dvfits).T # shape:(npixel, ntrial)

def overall_sigma(sig2s):
	'''
	compute overall sigma across pixels
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
	dvi, sig2s = liske_dvi(lam, flux1, flux2, error1, error2)
	sigma = overall_sigma(sig2s) # overall sigma
	return sigma

# --- empirical relations ---
def sigma_liske_empir(zqso, snr=1.3e4, nqso=1, lyb2lya=True):
	'''
	Empirical sigma expression (eq 15) in Liske 2008
	assuming observing Lyb to Lya
	output sigma (cm/s)
	'''
	if lyb2lya: factor = 2.
	else: factor = 1.35
	sigma = factor * (2370./snr) * (30./nqso)**0.5 * (5./(1.+zqso))**1.7
	return sigma

def sigma_liske_empir_multiepoch(zqso, snr_tot=1.8e4, nepoch=2, nqso=1, lyb2lya=True):
	'''
	Empirical sigma expression (eq 24)
	'''
	if lyb2lya: factor = 2.
	else: factor = 1.35
	g = np.sqrt(3. * (nepoch - 1.) / (nepoch + 1.))
	sigma = factor * (3350./snr_tot) * (30./nqso)**0.5 * (5./(1.+zqso))**1.7 * g
	return sigma

def snr_empir(rmag, t_tot, diameter, efficiency=0.18):
	'''
	Empirical SNR (Liske eq 26) from SDSS r mag
	t_tot - in s, total exptime
	diameter - in m
	'''
	snr = 700 * (10**(0.4*(16. - rmag)) * (diameter / 42.)**2. * (t_tot / (10. * 3600.)) * (efficiency / 0.25))**0.5
	return snr

# --- multiple epochs ---
def wa(x, flux_err):
	'''
	weighted average for 2d. eq 19 of Liske2008
	x - (Nepoch, ?) to be averaged
	flux_err - (Nepoch, Npixel)
	Returns
	xbar - (Npixel,)
	'''
	if len(x.shape)==1: x = np.atleast_2d(x).T # reshape x from (Nepoch,) to (Nepoch, 1)
	sigmaN2 = flux_err ** -2.
	xbar = np.sum(x * sigmaN2, axis=0) / np.sum(sigmaN2, axis=0) # shape (Npixel,)
	return xbar

def wa_1d(x, sig2):
	'''
	weighted average for 1d
	x - 1d array
	sig2 - error^2
	Returns
	xbar - scalar
	'''
	sumup = 0.
	sumdown = 0.
	for ind in range(len(x)):
		if not np.isnan(x[ind]) and not np.isnan(sig2[ind]) and not np.isinf(sig2[ind]):
			sumup += x[ind] / sig2[ind]
			sumdown += 1. / sig2[ind]
	xbar = sumup / sumdown
	#xbar = np.sum(x / sig2) / np.sum(1. / sig2)
	return xbar

def mi_estimate(flux, flux_err, dt):
	'''
	eq 18 of Liske2008
	flux, flux_err - (Nepoch, Npixel)
	dt - (Nepoch,) or (Nepoch, 1), sequence of dts, unit should be internally consistent
	Returns
	mi - (Npixel,) unit: flux / dt
	'''
	if len(dt.shape)==1: dt = np.atleast_2d(dt).T # reshape dt from (Nepoch,) to (Nepoch, 1)
	dtbar = wa(dt, flux_err)
	above = wa(flux * dt, flux_err) - wa(flux, flux_err) * dtbar
	below = wa(dt ** 2., flux_err) - dtbar ** 2.
	mi = - above / below
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

def liske_dvdti(lam, flux, flux_err, dt):
	'''
	dvdt and sigma2_dv/dt, eq 21 of Liske2008
	lam - (Npixel,)
	flux, flux_err - (Nepoch, Npixel)
	dt - (Nepoch,) or (Nepoch, 1), sequence of dts, unit should be internally consistent
	Return: dvdti, sigma_dvdti2 - (Npixel,)
	'''
	if len(dt.shape)==1: dt = np.atleast_2d(dt).T # reshape dt from (Nepoch,) to (Nepoch, 1)
	flux1 = flux[0, :]
	error1 = flux_err[0, :]
	slope, slope_err = slope_estimate(lam, flux1, error1) # (Npixel,)
	cslopelam = c.c.to('cm/s').value / (slope * lam) # c / (slope*lam)
	mi = mi_estimate(flux, flux_err, dt)
	# compute dv/dt
	dvdti = cslopelam * mi # cm/s/yr
	# compute sigma_dvdt2
	left = cslopelam**2. # (Npixel,)
	right = sigma_mi2(flux_err, dt) + mi**2. / slope**2. * slope_err**2. # (Npixel,)
	sigma_dvdti2 = left * right # (Npixel,)
	return dvdti, sigma_dvdti2

def liske_sigma_multiepoch(lam, flux, flux_err, dt):
	'''
	lam - (Npixel,)
	flux, flux_err - (Nepoch, Npixel)
	dt - (Nepoch,) or (Nepoch, 1), sequence of dts, unit should be internally consistent
	'''
	dvdti, sig2_dvdti = liske_dvdti(lam, flux, flux_err, dt) # (Npixel,)
	sig_dvdt = overall_sigma(sig2_dvdti) # 1 value
	dt0 = dt[-1] # 1 value
	sig_v = sig_dvdt * dt0 # 1 value
	return sig_v

def cosmomc_product_multiepoch(zs, dvdt, dvdt_err, tosave, fmt='dv'):
	'''
	generate zdrift observations data for CosmoMC
	fmt - dv or dz
	output columns: z, z_err, dvdt(dzdt), dvdt_err(dzdt_err)
	'''
	if fmt=='dz':
		dzdt = cs.dv2dz(dvdt, zs)
		dzdt_err = cs.dv2dz(dvdt_err, zs)
	zeros = np.zeros(len(zs))
	if fmt=='dv':   out = np.vstack([zs, zeros, dvdt, dvdt_err]).T
	elif fmt=='dz': out = np.vstack([zs, zeros, dzdt, dzdt_err]).T
	np.savetxt(tosave, out, fmt='%.12e'); print('Saved: %s'%tosave)

# --- binning sigma by z/lam ---
def sigma_binned(zol, dvs, sig2s, nbins=None, binwidth=None, return_binedges=False):
	'''
	zol - z or lam
	dvs - dv or dv/dt
	sig2s - sig2_dv or sig2_dvdt
	nbins and binwidth (in zol unit) should have one defined
	Return
	zdvsigma - array, (3, nbins) [[z, z, ...], [dv, dv, ...], [sigma, sigma, ...]]
	'''
	left = np.min(zol)
	right = np.max(zol)
	if nbins!=None: binedges = np.linspace(left, right, nbins+1)
	elif binwidth!=None: binedges = np.arange(left, right+binwidth/2., binwidth)
	zdvsigma = []
	for ind in range(len(binedges)-1):
		crit = (binedges[ind] < zol) & (zol <= binedges[ind+1])
		z_thisbin = (binedges[ind] + binedges[ind+1]) / 2.
		#dv_thisbin = np.nanmedian(dvs[crit])
		dv_thisbin = wa_1d(dvs[crit], sig2s[crit])
		sigma_thisbin = overall_sigma(sig2s[crit])
		zdvsigma.append([z_thisbin, dv_thisbin, sigma_thisbin])
	zdvsigma = np.array(zdvsigma).T # (3, nbins)
	if return_binedges: return zdvsigma, binedges
	else: return zdvsigma

if __name__ == '__main__':
	sigma_mode = 'compute' # 'estimate' to estimate from dvfits array or 'compute' to compute from Liske equation
	import read_elqs as re
	import continuum_fit as cf

	item = re.top10_df().iloc[0]
	z_genspec = item['z'] # if ==None then use keck line list
	ispec = None
	ntrial = 100
	time = 10. # year
	cosmomc_tosave = None
	# nphot
	nphot = 1.69e8
	fluxpars = pkload(cf.fluxpar_filename(item['KOAjobID'])) # if ==None then use nphot value
	fluxpars = None
	aperture = 0
	exptime = 0

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
			specpack = spec_two_epoch(nphot=nphot, fluxpars=fluxpars, z_genspec=z_genspec, period=time, addnoise=2, ispec=ispec, aperture=aperture, exptime=exptime)
			dvi, sig2s = liske_dvi(*specpack)
			sigma = overall_sigma(sig2s) # overall sigma
			if cosmomc_tosave!=None: cosmomc_product(specpack[0], dvi, np.sqrt(sig2s), time, cosmomc_tosave, fmt='dv')
		elif sigma_mode=='estimate': # estimate sigma from computed dvi for ntrials
			dvfits = liske_dvfits(ntrial, nphot, z_genspec, period=time, ispec=ispec)
			sigma = sse.dvfits2sigma(dvfits, relaerr_all=None, relaerr_thrshld=1., dvstd_thrshld=np.inf, bs_time=1)
		print('z: %s ispec: %s sigma: %.4f cm/s'%(z_genspec, ispec, sigma))
		sigmas.append(sigma)
	sig2zqso.append(np.mean(sigmas)) # list of mean sigmas
	sig2zqso_err.append(np.std(sigmas,ddof=1)) # unbiased std
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
	'''
	plt.errorbar(zqsos, sig2zqso, yerr=sig2zqso_err)
	plt.xlabel('zqso')
	plt.ylabel('sigma (cm/s)')
	plt.ylim([1.5,5.8])
	tosave = path + '/plots/sigmaVSzqso_genspec_nphot%.2e_epoch%dyr.pdf'%(nphot,time)
	plt.savefig(tosave);print('Saved: %s'%tosave)
	plt.show()
	'''

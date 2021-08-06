from utils import *
import spec_utils as su

def fnu2m1450(F_nu, zqso, cosmo=cosmology.Planck15):
	'''
	F_nu in Jy (1 Jy = 1e-26 W / (m2 Hz) = 1e-23 erg / (s cm2 Hz))
	'''
	mu = cosmo.distmod(zqso).value
	M_1450 = -2.5 * np.log10(F_nu/3631.) - mu + 2.5 * np.log10(1. + zqso)
	return M_1450

def m14502fnu(M_1450, zqso, cosmo=cosmology.Planck15):
	'''
	F_nu in Jy (1 Jy = 1e-26 W / (m2 Hz) = 1e-23 erg / (s cm2 Hz))
	'''
	mu = cosmo.distmod(zqso).value
	F_nu = 3631. * 10**(- 0.4 * (M_1450 + mu)) * (1. + zqso)
	return F_nu

def m14502sdss(m1450, zqso, cosmo=cosmology.Planck15):
	'''
	convert m1450 to sdss flux (1e-17 erg/(cm2 s AA))
	'''
	if type(m1450)==pd.Series: m1450 = m1450.to_numpy()
	if type(zqso)==pd.Series: zqso = zqso.to_numpy()
	Jy = m14502fnu(m1450, zqso, cosmo)
	lam_1450 = 1450.*(1. + zqso)
	f_sdss = su.convert_flux(Jy*u.Jy, '1e-17 erg/(cm2 s AA)', lam_1450*u.AA)
	return f_sdss.value

def norm2f1450(spec, f1450, return_factor=False):
	'''
	normalize a spectra to fit in a given f1450 value
	lam - should be in rest frame
	'''
	lam = spec[0]; flux = spec[1]
	lamrange = pd.Interval(min(lam), max(lam), closed='both')
	extrapolate = 1450. not in lamrange
	
	if extrapolate: # assuming spectra is to the left (red) of 1450 AA
		# use last 2 AA of flux
		tomedian = flux[(lam >= lamrange.right - 2.) & (lam <= lamrange.right)]
	else: # get median value of 1450 +/- pm AA
		pm = 1.
		while True:
			tomedian = flux[(lam >= (1450. - pm)) & (lam <= (1450. + pm))]
			if len(tomedian) > 0: break
			pm += 2.
	flux_1450 = np.median(tomedian) # uncorrected flux at 1450
	factor = f1450 / flux_1450
	if return_factor: return factor
	spec[1:3] *= factor # for flux, flux_err
	return spec

from utils import *

class liske_cosmo:
	Om0 = 0.3
	Ode0 = 0.7
	H0 = 70. * u.km/u.Mpc/u.s
	def efunc(z):
		zp1 = z + 1.
		m = liske_cosmo.Om0 * zp1**3.
		de = liske_cosmo.Ode0 * zp1
		Ez = np.sqrt(m + de)
		return Ez

'''
def dzdt(z, cosmo=cosmology.Planck15):
	zp1 = z + 1.
	m = cosmo.Om0 * zp1**3.
	de = cosmo.Ode0 * zp1
	Ez = np.sqrt(m + de)
	dzdt = (zp1 - Ez) * cosmo.H0
	return dzdt.to('yr-1').value
'''

def dzdt(z, cosmo=cosmology.Planck15):
	'''
	return dzdt (yr-1)
	'''
	dzdt = ((1. + z) - cosmo.efunc(z)) * cosmo.H0
	return dzdt.to('yr-1').value

def dz2dv(dz, z):
	dv =  c.c.to('cm s-1').value * dz / (1. + z)
	return dv

def dvdt(z, cosmo=cosmology.Planck15):
	'''
	LCDM model
	return dvdt (cm s-1 yr-1)
	'''
	dz = dzdt(z, cosmo)
	return dz2dv(dz, z)

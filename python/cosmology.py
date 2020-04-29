from utils import *

def dzdt(z, cosmo=cosmology.Planck15):
	'''
	LCDM model
	return dzdt (yr-1)
	'''
	zp1 = z + 1.
	m = cosmo.Om0 * zp1**3.
	de = cosmo.Ode0 * zp1
	Ez = np.sqrt(m + de)
	dzdt = (zp1 - Ez) * cosmo.H0
	return dzdt.to('yr-1').value

'''
def dzdt(z, cosmo=cosmology.Planck15):
	dzdt = ((1. + z) - cosmo.efunc(z)) * cosmo.H0
	return dzdt.to('yr-1').value
'''

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

from utils import *
from spec_utils import *
import spec_utils as su
import astropy.units as u
import astropy.constants as c

def bigO_liske(aperture, t_tot, efficiency):
	'''
	Liske2008 eq 27
	aperture - values or Quantity, default unit: m
	t_tot - values or Quantity, default unit: s
	'''
	if type(aperture)!=u.Quantity: aperture *= u.m
	if type(t_tot)!=u.Quantity: t_tot *= u.s
	O = (aperture / (42. * u.m))**2. * efficiency / 0.25 * t_tot / (2000. * u.h)
	O = O.to(u.one).value
	return O

def flux2nphot(lam, flux, aperture, exptime, efficiency=1., disp=None):
	'''
	Change flux unit from energy to Nphoton/pixel
	lam - should be in observed frame. values or Quantity, default unit: AA
	flux - values or Quantity, default unit: 1e-17 erg cm-2 s-1 AA-1
	aperture - values or Quantity, default unit: m
	exptime - values or Quantity, default unit: s
	'''
	if type(lam)!=u.Quantity: lam *= u.AA
	if type(flux)!=u.Quantity: flux = flux * 1e-17 * u.Unit('erg cm-2 s-1 AA-1')
	if type(aperture)!=u.Quantity: aperture *= u.m
	if type(exptime)!=u.Quantity: exptime *= u.s
	if np.any(disp): 
		if type(disp)!=u.Quantity: disp *= u.AA
	else: disp = np.gradient(lam) # dispersion in AA (per pixel)
	#print('disp min mean max: %.4f %.4f %.4f'%(np.min(disp).value, np.mean(disp).value, np.max(disp).value))
	area = (np.pi * (aperture/2.)**2.).to('cm2') # telescope area
	Ephot = (c.h * c.c / lam).to('erg') # electron energy

	nphot = (flux / Ephot * area * exptime * disp * efficiency).to(u.one).value
	return nphot

def FluxUnitChange(spec,z_plot_rest_frame=0.,mode='e2p'):
	'''
	Change flux unit between energy and Nphoton/pixel
	INPUT
	    spec - (lam, [flux, flux_err, ...], disp, exptime)
	        lam unit: A
	        flux unit:
	    	sdss_flux_label='Flux ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)'
		                 erg cm-2 s-1 AA-1
	        disp unit: Angstrom / pixel
	        exptime unit: seconds
	    mode -
	        'e2p': energy [10^-17 erg / (Angstrom cm2 s)] to photon/pixel
	        'p2e': photon/pixel to energy 
	OUTPUT
	    outfluxes - [flux, flux_err, ...]
	    flux unit:
		photon / pixel
	'''
	# define units
	upix=u.def_unit('pixel')
	uphot=u.def_unit('photon') # unit of number of photons
	uSpecBin=u.def_unit('SpecBin') # spectral bin
	uEnergy=1e-17*u.erg*u.cm**-2/u.s/u.Angstrom # flux unit in energy [10^-17 erg / (Angstrom cm2 s)]

	lam,flux_package,disp,exptime=spec
	L = su.obs_frame(lam,z_plot_rest_frame)*u.Angstrom # lam observed
	D=disp*u.Angstrom/upix # dispersion [Angstrom / pixel]
	T=exptime*u.s

	Ephot=(c.h*c.c/L/uphot).to(u.erg/uphot) # photon energy [erg / photon]
	keckApertureArea=np.pi*(996./2.)**2.*u.cm**2. # cm2

	### specbin width
	# gradient
	SpecBinWidth=np.gradient(L)/uSpecBin # [Angstrom / SpecBin]
	# diff
	#diff=np.diff(L.value)
	#SpecBinWidth=np.hstack([diff[0],diff])*L.unit/uSpecBin # [Angstrom / SpecBin]

	# calculate converting factor (len(lam))
	factor=uEnergy # [10^-17 erg / (Angstrom cm2 s)]
	factor=factor/Ephot # [photon / (Angstrom cm2 s)] [photon cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]
	factor*=(keckApertureArea*T) # [photon / Angstrom] [photon $\AA^{-1}$]
	#factor*=SpecBinWidth # [photon / SpecBin] [photon SpecBin$^{-1}$]
	#factor=factor.to(uphot/uSpecBin)
	factor*=D # [photon / pix] [photon pixel$^{-1}$]
	factor=factor.to(uphot/upix)

	outfluxes=[]
	for flux in flux_package:
		if mode=='e2p':
			F_NphotPerPixel=flux*factor
			outfluxes.append(F_NphotPerPixel.value)
		elif mode=='p2e':
			F_energy=flux/factor
			outfluxes.append(F_energy.value)

	return outfluxes

def nelectron_R(R, F_lam=1., lam=5000., npix_perres=3, D=15., t_int=1., efficiency=0.25, electron_perphot=1):
	'''
	compute number of electrons as function of R, using my paper eq 6 (Nphot = ...)
	Nphot = 1.33e6 * (F_lam / 1e-15 erg cm-2 s-1 AA-1) * (lam / 5000 AA) * (dlam / 0.033... AA) * (D / 15m)^2 * (t_int / 100 h) * (efficiency / 0.25)
	Nphot = 1.33e6 * (F_lam / 1e-15 erg cm-2 s-1 AA-1) * (lam / 5000 AA)^2 * (3 / npix_perres) * (50,000 / R) * (D / 15m)^2 * (t_int / 100 h) * (efficiency / 0.25)
	Inputs
	    F_lam - in 1e-15 erg cm-2 s-1 AA-1 (range 0.1~1+ max10+)
	    lam - in AA
	    npix_perres - number of pixels per resolution element
	    D - aperture, in m
	    t_int - integration time, in hour
	    efficiency - total throughput
	'''
	nphot = 1.33e6 * F_lam * (lam / 5000.)**2. * (3. / npix_perres) * (5e4 / R) * (D / 15.)**2. * (t_int / 100.) * (efficiency / 0.25)
	nelectron = electron_perphot * nphot
	return nelectron

if __name__ == '__main__':
	'''
	plot Nelectron vs R
	'''
	paras = {
		'F_lam': 0.1, # 1e-16 erg cm-2 s-1 AA-1
		't_int': 1./60, # 1 minute
	}
	fig, ax = plt.subplots()

	R = np.linspace(1e4, 1e5, 10)
	setups = [{'D': 15., 't_int': 1./60},
	          {'D': 10., 't_int': 1./60},
	          {'D':  5., 't_int': 1./60}]
	labels = ['%d m, %d min'%(dic['D'], dic['t_int']*60.) for dic in setups]

	# dark current and read-out noise
	# HIRES: https://www2.keck.hawaii.edu/inst/hires/hires_data.pdf
	#    DC  2 e-/pixel/hour
	#    RON < 3 e-
	# ESPRESSO: https://www.eso.org/sci/facilities/paranal/instruments/espresso/
	#    DC  1 e-/pixel/hour
	#    RON(slow) 3(blue) 2(red) e-/pixel
	#    RON(fast) 8(blue) 5(red) e-/pixel
	dc = 1. # e-/pixel/hour
	ron = 3. # e-/pixel

	for iset in range(len(setups)):
		ne = nelectron_R(R, F_lam=0.3, **setups[iset])
		e_noise = np.sqrt(ne)
		ax.plot(R / 1e3, e_noise, label=labels[iset])

		t_int = setups[iset]['t_int']
		dc_int = dc * t_int
		ax.axhline(ron + dc_int, color='b')#, label='RON + DC')
		ax.axhline(ron, color='r')#, label='Readout noise')
		ax.axhline(dc_int, color='k')#, label='Dark current')

	#ax.set_xscale('log')
	#ax.set_yscale('log')
	ylim = ax.get_ylim()
	ax.set_ylim([0, ylim[1]])

	ax.set_xlabel('Spectral resolution / 1000')
	ax.set_ylabel('Photon noise ($e^{-}$/pixel)')
	ax.legend(loc=2)
	fig.tight_layout()
	plt.show()

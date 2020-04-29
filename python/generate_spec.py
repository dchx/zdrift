'''
Generate Lya forest from literature parameter distributions following Kim et al. 2001
'''
from __future__ import print_function
from utils import *
from scipy.stats import rv_continuous
from scipy import integrate,special
from read_koa import flux_smooth
LiskeDist = 1

# ----------- b distribution -----------

bpars = pd.DataFrame(np.array([[1.61, 7.37, 23.01, 12.61, 28.14, 34.56],\
[1.98, 7.98, 23.83, 13.04, 26.04, 29.10],\
[2.13, 7.46, 23.61, 12.95, 25.34, 29.57],\
[2.66, 6.82, 24.09, 13.25, 28.30, 30.10],\
[2.87, 7.09, 27.75, 15.30, 28.74, 34.05],\
[3.75, 6.72, 22.41, 12.31, 28.90, 30.70]]),\
columns=['zmean','B_HR','b_sigma','b_HR','b_meda','b_medb'])
bpars = bpars.set_index('zmean', drop=False)

zmean_default = bpars.zmean.iloc[5] # default zmean = 3.75, bhr = 6.72, bsig = 22.41
zmean_default = bpars.zmean.iloc[4] # default zmean = 2.87, bhr = 7.09, bsig = 27.75
def b_dist(b, bhr=bpars.B_HR[zmean_default], bsig=bpars.b_sigma[zmean_default]):
	'''
	b distribution analytical function (Hui & Rutledge 1999)
	'''
	bsig4 = bsig**4.
	dndb = bhr * bsig4 / b**5. * np.exp(- bsig4 / b**4.)
	# normalize
	int_dndb = bhr / 4. # int_0^inf dndb db
	dndb = dndb / int_dndb # normalize to 1
	return dndb

class b_gen(rv_continuous):
	'''
	b distribution generator
	'''
	def _pdf(self, b):
		if b <= 0: return 0.
		else: return b_dist(b)
b_distribution = b_gen(name='b', a=0.)

# ----------- N_HI distribution -----------

NHI_left = 1e12
NHI_right = 1e16
a_dndNHI = 4.9e7
a_dndNHI = 3.537165443330732e8 # to give 100 lines with 13.64 < log N H I (cm−2 ) < 16 per unit redshift
b_dndNHI = 1.46 # Kim2001
if LiskeDist: b_dndNHI = 1.5 # LiskeDist
def dndNHI(NHI):
	'''
	from Kim et al. 2001, Hu et al. 1995
	'''
	return a_dndNHI * NHI**(-b_dndNHI)
def int_dndNHI(left=NHI_left, right=NHI_right):
	'''
	int_left^right dndNHI dNHI
	'''
	l_b = 1. - b_dndNHI # 1 - b_dndNHI
	integral = a_dndNHI / l_b * (right ** l_b - left ** l_b) # int_left^right dndNHI dNHI
	return integral

def NHI_dist(NHI, left=NHI_left, right=NHI_right):
	'''
	NHI distribution from Hu et al. 1995
	'''
	return dndNHI(NHI) / int_dndNHI(left, right) # normalize to 1
class NHI_gen(rv_continuous):
	'''
	NHI distribution generator
	'''
	def _pdf(self, NHI, left, right):
		if NHI < left or NHI > right: return 0.
		else: return NHI_dist(NHI, left, right)
NHI_distribution = NHI_gen(name='NHI', a=NHI_left, b=NHI_right)

# ----------- redshift distribution -----------

a_dndz = 9.06
b_dndz = 2.19 # Kim2001
if LiskeDist: b_dndz = 2.2 # LiskeDist
z_left = 2. # left bound to integrate dndz
z_left = 0. # left bound to integrate dndz
def dndz(z):
	'''
	from Kim et al. 2001
	'''
	return a_dndz * (1. + z)**b_dndz
def int_dndz(zleft, zright):
	'''
	int_zleft^zright dndz dz
	'''
	bp1 = b_dndz + 1. # b + 1
	return a_dndz / bp1 * ((1. + zright)**bp1 - (1. + zleft)**bp1)

def z_lyb(z_qso):
	'''
	redshift of lyman beta
	'''
	lyb2lya = 27. / 32. # (1 - 1/2^2) / (1 - 1/3^2)
	return lyb2lya * (1. + z_qso) - 1.

def z_dist(z, z_qso, zleft=z_left):
	'''
	z - the redshift to calculate pdf
	z_qso - z of qso, the right bound to ingegrate dndz
	'''
	left = max(zleft, z_lyb(z_qso))
	return dndz(z) / int_dndz(left, z_qso) # normalize to 1
class z_gen(rv_continuous):
	'''
	z distribution generator
	'''
	def _pdf(self, z, zleft, z_qso):
		left = max(zleft, z_lyb(z_qso))
		if z < left or z > z_qso: return 0.
		else: return z_dist(z, z_qso, left)

# ----------- number of lines -----------

def nlines(z_qso, zleft=z_left, NHIleft=NHI_left, NHIright=NHI_right):
	'''
	number of lines in the lya forest as function of z_qso
	'''
	# number of lines in the range 13.64 < log10(NHI) < 16
	left = max(zleft, z_lyb(z_qso))
	nl = int_dndz(left, z_qso)
	# convert nlines in 13.64 < log10(NHI) < 16 to designated NHI range
	#NHI_norm_factor = int_dndNHI(NHIleft, NHIright) / int_dndNHI(10.**13.64, 1e16) 
	#nl = nl * NHI_norm_factor
	#nl = np.random.poisson(nl)
	nl = 1
	return int(round(nl))

# ----------- combined distribution -----------

def fzNb(z, NHI, b):
	'''
	Liske2008 eq.7
	NHI in cm-2, b in km/s
	'''
	gamma = 2.2
	beta = 1.5
	b_bar = 30. # km/s
	sigma_b = 8. # km/s
	return (1. + z)**gamma * NHI**(-beta) * np.exp(- (b - b_bar)**2. / (2. * sigma_b**2.))

# ----------- line profile -----------

gamma_lya = 7618.*1e8 / lya_wave # s-1, gamma_{ul}
flu_lya = 0.4164
# intrisic line profile
def phi_intrin(nu, nu0=lya_freq, gamma=gamma_lya):
	return 4. * gamma / (16. * np.pi**2. * (nu - nu0)**2. + gamma**2.)
def phi_intrin_lam(lam, nu0=lya_freq, gamma=gamma_lya):
	nu = aa2nu(lam)
	return intrinsic_profile(nu, nu0, gamma)
# voigt line profile
def voigt_tointegrate(v, b, nu, nu0=lya_freq, gamma=gamma_lya):
	'''
	v, b in km/s
	'''
	gauss_part = np.exp(- v**2. / b**2.) / (b * np.sqrt(np.pi)) 
	lorentz_part = 4. * gamma / (16. * np.pi**2. * (nu - (1. - v * 1e3 / c.c.value) * nu0)**2. + gamma**2.)
	return gauss_part * lorentz_part
def phi_nu(nu, b, nu0=lya_freq, gamma=gamma_lya):
	nu = np.atleast_1d(nu)
	# use integrate
	#intout = np.array([integrate.quad(voigt_tointegrate, -np.inf, np.inf, args=(b, nui, nu0, gamma))[0] for nui in nu])
	# sum
	vs = np.linspace(-100, 100, 1000)
	dv = vs[1] - vs[0]
	intout = np.array([dv * np.sum(voigt_tointegrate(vs, b, nui, nu0, gamma)) for nui in nu])
	return intout
	
def phi_lam(lam, b, nu0=lya_freq, gamma=gamma_lya):
	nu = aa2nu(lam)
	return phi_nu(nu, b, nu0, gamma)

# ----------- voigt profile approximation -----------

def K(x, y):
	'''
	Voigt function
	'''
	z = complex(x,y)
	return special.wofz(z).real
def voigt_profile_onevalue(nu, b, nu0=lya_freq, gamma=gamma_lya):
	'''
	b in km/s
	'''
	lightspd = c.c.value/1e3 # km/s
	x = lightspd * (nu0-nu) / (nu0 * b)
	y = gamma * lightspd / (4. * np.pi * nu0 * b)
	phi = lightspd / (np.sqrt(np.pi) * nu0 * b**2.) * K(x, y)
	return phi
def voigt_profile(nu, b, nu0=lya_freq, gamma=gamma_lya):
	nu = np.atleast_1d(nu)
	return np.array([voigt_profile_onevalue(nui, b, nu0, gamma) for nui in nu])
def voigt_profile_lam(lam, b, nu0=lya_freq, gamma=gamma_lya):
	nu = aa2nu(lam)
	return voigt_profile(nu, b, nu0, gamma)

# ----------- optical depth -----------

def tau0(NHI, b):
	'''
	optical depth at line center for Lya (Draine eq 9.10)
	assuming gaussian only
	NHI in cm-2, b in km/s
	'''
	return 0.758 * (NHI / 1e13) * (10. / b)
def tau_nu(nu, b, NHI, nu0=lya_freq, gamma=gamma_lya):
	phi_nu = voigt_profile(nu, b, nu0, gamma)
	phi_nu_norm = phi_nu / abs(integrate.trapz(phi_nu, nu)) # normalize to one
	#pie2mec = np.pi * c.e.value**2. / (c.m_e.cgs.value * c.c.cgs.value)
	pie2mec = 1.497e-2 * np.sqrt(np.pi)
	taunu = pie2mec * flu_lya * NHI * phi_nu_norm
	return taunu
def tau_lam(lam, b, NHI, lam0=lya_wave, gamma=gamma_lya):
	nu = aa2nu(lam)
	nu0 = aa2nu(lam0)
	return tau_nu(nu, b, NHI, nu0, gamma)
def voigt1d(lam0, lgNHI, b, gamma=gamma_lya):
	NHI = 10.**lgNHI
	def v1d(lam): return tau_lam(lam, b, NHI, lam0, gamma)
	return v1d

def multivoigt_parray(parray, lam, v1d=voigt1d, verbose=False):
	'''
	Compute multivoigt from an parameter array
	Inputs: parray - [nparas, nlines]
	'''
	tau = np.zeros(len(lam))
	if verbose: print('%d lines'%len(parray.T))
	for iline, args in enumerate(parray.T):
		if verbose: print('line %d'%iline, args)
		tau += v1d(*args)(lam)
	return tau

# ----------- spectra generation -----------

def form_spec(continuum, multivoigts, voigt_is_tau=True):
	'''
	form spectra given multivoigt profiles, whether to treat voigts as tau or not
	'''
	if voigt_is_tau: return continuum * np.exp(-multivoigts)
	else: return continuum - multivoigts
def parray2flux(parray, lam, continuum=1., voigt_is_tau=True, v1d=voigt1d, tosave=None):
	if tosave and os.path.exists(tosave):
		flux = pkload(tosave)
	else:
		tau = multivoigt_parray(parray, lam, v1d=v1d)
		flux = form_spec(continuum, tau, voigt_is_tau=voigt_is_tau)
		if tosave: pkdump(flux, tosave)
	return flux
def parray2flux_dz(parray, lam, shiftmode, dz, zqso, continuum=1., voigt_is_tau=True, v1d=voigt1d, tosave=None):
	'''
	parray - shape:(nparas, nlines) [lam0, lgNHI, b] (for v1d=voigt1d) or [lam0, AL, fL, fG] (for v1d=Voigt1D)
	dz - redshift drift
	'''
	if tosave and os.path.exists(tosave):
		flux = pkload(tosave, verbose=False)
	else:
		# shift lam0s for dl
		if 'dl' in shiftmode: parray[0] += dz
		else: # dz or dv
			# shift lam0s
			if 'dz' in shiftmode: factor = 1. + dz / (1. + zqso)
			elif 'dv' in shiftmode: factor = 1. + dz / c.c.to('cm/s').value
			else: raise ValueError("shiftmode should be dz, dl ov dv, not %s."%mode)
			parray[0] *= factor
			# shift line widths
			if 'lw' in shiftmode:
				parray[2] *= factor
				if v1d.__name__=='Voigt1D': parray[3] *= factor
		# flux
		flux = parray2flux(parray, lam, continuum=continuum, voigt_is_tau=voigt_is_tau, v1d=v1d)
		if tosave: pkdump(flux, tosave)
	return flux

def generate_spec(zqso, res=2e4, dlam=0.0125, fix='res', dz=0., rest_frame=True, shiftmode='dz', ispec=None, verbose=True):
	'''
	zqso - redshift of quassar
	res - spectral resolution, used if fix=='res'
	dlam - AA per pixel, 0.0125 for Liske2008, used if fix=='resele'
	fix - 'res' or 'resele'
	dz - shift in z or lam
	fix - if 'res', fix resolution and use different resolution element sizes at different lams;
	      if 'resele', fix resolution element size and use different resolution R values at different lams;
	rest_frame - whether to generate spectra in rest frame (or in observed frame)
	shiftmode - 'dz' if dz is shift in z, 'dl' if dz is shift in lambda
	ispec - index of spectra, if want to generate multiple spectra with same parameters
	
	Returns
	lam, flux - spectral template, not added noise
	'''
	restframetxt = '_restframe' if rest_frame else ''
	restxt = '_dlam%.3e'%dlam if fix=='resele' else '_R%.1e'%res if fix == 'res' else ''
	ispectxt = '_spec%d'%ispec if ispec != None else ''
	liskedisttxt = '_LiskeDist' if LiskeDist else ''
	suffix = ispectxt + liskedisttxt
	if fix == 'res':
		# dlam = 1e3 / 2. / res # for 2 pix / resolution element at 1000 AA
		dlam = 5e3 / 4. / res # for 4 pix / resolution element at 5000 AA (Liske2008)
	elif fix == 'resele':
		res_ele = 4. # resolution element size in pixel
	else: raise Exception("argument 'fix' should be either 'res' or 'resele'")

	spec_file = path + 'data/gen_spec/gen_spec%s_zqso%.1f%s_%s%.3e%s.pickle'%(restframetxt,zqso,restxt,shiftmode,dz,suffix)
	if os.path.exists(spec_file):# and __name__!='__main__': # load previously saved spectra, all not added noise
		lam, flux = pkload(spec_file, verbose=verbose)
	else:
		left = max(z_left, z_lyb(zqso)) # spectra left bound in z
		# wavelength range
		if rest_frame:
			lam_left = lya_wave * (1. + left) / (1. + zqso)
			lam_right = lya_wave
		else:
			lam_left = lya_wave * (1. + left)
			lam_right = lya_wave * (1. + zqso)
		# lam points
		nlam = int(round((lam_right - lam_left) / dlam))
		lam = np.linspace(lam_left, lam_right, nlam)
		if fix == 'res': res_ele = np.mean(lam / dlam / res) # in pixels
		elif fix == 'resele': res = lam / dlam / res_ele # no unit
		# line parameter file
		blgNHIz_file = path + 'paras/blgNHIz_zqso%.1f%s%s.pickle'%(zqso,ispectxt,liskedisttxt)
		# --- get b, NHI and z parameters for each line ---
		if os.path.exists(blgNHIz_file) and __name__!='__main__': # load previously saved parameters
			bs, lgNHIs, zs = pkload(blgNHIz_file, verbose=False)
		else:
			# number of lines
			nl = nlines(zqso)
			# generate bs
			bs = b_distribution.rvs(size=nl) # generated line width bs
			if LiskeDist: 
				bs = []
				while len(bs)<nl:
					newb = np.random.normal(30., 8.)
					if 15. < newb and newb < 100.: bs.append(newb)
				bs = np.array(bs)
			# generate NHIs
			NHIs = NHI_distribution.rvs(size=nl, left=NHI_left, right=NHI_right)
			lgNHIs = np.log10(NHIs)
			# generate lam0s
			z_distribution = z_gen(name='z', a=left, b=zqso)
			zs = z_distribution.rvs(size=nl, zleft=left, z_qso=zqso)
			pkdump((bs, lgNHIs, zs), blgNHIz_file)
		if rest_frame: lam0s = lya_wave * (1. + zs) / (1. + zqso)
		else: lam0s = lya_wave * (1. + zs)
		# form parray
		parray = np.array([lam0s, lgNHIs, bs])
		# --- form spectrum from parameters ---
		flux = parray2flux_dz(parray, lam, shiftmode, dz, zqso, voigt_is_tau=True, v1d=voigt1d)
		# convolve with resolution element
		flux = flux_smooth(flux, res_ele)
		# save spectra
		pkdump((lam, flux), spec_file)
	return lam, flux

def compare_Voigt1D_taulam():
	lam = np.arange(4995,5005,0.0125)
	lam0 = 5000.
	tau = tau_lam(lam, b=20., NHI=1e13, lam0=lam0)
	flux_tau = np.exp(-tau)
	voigt = Voigt1D(x_0=lam0, amplitude_L=38.,fwhm_L=0.01,fwhm_G=0.7)(lam)
	voigt = Voigt1D(x_0=lam0, amplitude_L=34.,fwhm_L=0.01,fwhm_G=0.63)(lam)
	flux_voigt1d = 1. - voigt
	#flux_voigt1d = np.exp(-voigt)
	plt.plot(lam, flux_tau, label='generate')
	plt.plot(lam, flux_voigt1d, label='Voigt1D')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	'''
	# b distribution
	bs = np.linspace(1e-7,100,100)
	dndb = b_dist(bs)
	bsample = b_distribution.rvs(size=1000)
	plt.figure()
	plt.plot(bs, dndb)
	plt.hist(bsample,bins=50,density=True)
	plt.xlabel('b (km/s)')

	# NHI distribution
	NHIs = np.logspace(np.log10(NHI_left), np.log10(NHI_right), 100)
	dndNHI = NHI_dist(NHIs)
	NHIsample = NHI_distribution.rvs(size=1000, left=NHI_left, right=NHI_right)
	plt.figure()
	plt.loglog(NHIs, dndNHI)
	bins = np.logspace(np.log10(NHI_left), np.log10(NHI_right), 50)
	plt.hist(NHIsample, bins=bins, log=True, density=True)
	plt.xlabel('NHI (cm -2)')

	# z distribution
	z_qso = 4.
	left = max(z_left, z_lyb(z_qso))
	z_distribution = z_gen(name='z', a=left, b=z_qso)
	#
	zs = np.linspace(left, z_qso, 100)
	dndzs = z_dist(zs, z_qso, left)
	zsample = z_distribution.rvs(size=1000, zleft=left, z_qso=z_qso)
	plt.figure()
	plt.loglog(zs, dndzs)
	bins = np.logspace(np.log10(left), np.log10(z_qso), 50)
	plt.hist(zsample, bins=bins, log=True, density=True)
	plt.xlabel('z')
	'''

	# generate line profile
	'''
	lams = np.linspace(1215., 1216., 1000)
	nus = aa2nu(lams)
	b = 25
	NHI = 1e13
	plt.figure()
	#plt.plot(lams, np.exp(-phi_lam(lams, b)))
	#plt.plot(lams, np.exp(-voigt_profile_lam(lams, b)))
	#plt.plot(lams, tau_lam(lams, b, NHI))
	#plt.plot(lams, voigt_profile_lam(lams, b))
	#plt.plot(lams, tau_nu(nus, b, NHI))
	plt.plot(lams, np.exp(-tau_nu(nus, b, 1e12)), label='NHI=1e12')
	plt.plot(lams, np.exp(-tau_nu(nus, b, 1e13)), label='NHI=1e13')
	plt.plot(lams, np.exp(-tau_nu(nus, b, 1e14)), label='NHI=1e14')
	plt.plot(lams, np.exp(-tau_nu(nus, b, 1e15)), label='NHI=1e15')
	plt.legend()
	plt.figure()
	plt.plot(lams, np.exp(-tau_nu(nus, 15, NHI)), label='b=15')
	plt.plot(lams, np.exp(-tau_nu(nus, 25, NHI)), label='b=25')
	plt.plot(lams, np.exp(-tau_nu(nus, 35, NHI)), label='b=35')
	plt.plot(lams, np.exp(-tau_nu(nus, 45, NHI)), label='b=45')
	plt.legend()
	'''

	# generate spectra
	zqso = 3.4
	lam, flux = generate_spec(zqso, rest_frame=0, fix='resele')
	flux_wnoise = add_shot_noise(flux, nphot=1.69e8) # 5e12,1e4,1.69e8
	# estimate snr
	'''
	crit = flux>0.97
	#crit = flux>-1.
	#snr = np.nanmedian(flux[crit] / np.abs(flux_wnoise[crit] - flux[crit]))
	snr = np.mean(flux[crit])/np.std(flux_wnoise[crit] - flux[crit])
	print('flux mean: ',np.mean(flux[crit]))
	print('snr: ',snr)
	'''
	# ^^^ estimate snr
	plt.close('all')
	plt.figure(figsize=(12,3))
	plt.axhline(1,c='k')
	#plt.plot(lam,flux_wnoise,lw=0.5)
	plt.plot(lam,flux,'r')
	plt.axis([4800.,5000.,-0.1,1.1])
	plt.xlabel('$\lambda$ ($\AA$)')
	plt.ylabel('Normalized flux')
	plt.tight_layout()
	tosave = path + 'plots/generate_spec_zqso%.1f.pdf'%zqso
	plt.savefig(tosave);print('Saved:%s'%tosave)
	plt.show()

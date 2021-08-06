from utils import *
import itertools
from scipy.signal import gaussian

def nu2aa(nu): return c.c.value / nu * 1e10
def aa2nu(aa): return c.c.value / aa * 1e10
def lya2lyb(lam_lya): return lam_lya * 27./32.
def z_lyb(zqso): return lya2lyb(1. + zqso) - 1.
lya_wave = 1215.67 # Angstrom
lya_freq = aa2nu(lya_wave) # s-1
lyb_wave = lya2lyb(lya_wave) # 1025.72 Angstrom

def rest_frame(lam_obs, z):
	# input observed wavelengths lam_obs, provide redshift z
	# output rest frame wavelengths
	lam_emit = lam_obs/(1.+z)
	return lam_emit 

def obs_frame(lam_emit, z):
	# input rest frame wavelengths lam_emit, provide redshift z
	# output observed wavelengths
	lam_obs=lam_emit*(1.+z)
	return lam_obs

def lam2z(lam):
	'''
	convert a series of wavelengths (lam) in a Lya forest to redshifts of clouds on the path
	assuming lam in observed frame
	'''
	zs = lam / lya_wave - 1.
	return zs

def z2lam(zs):
	lam = (zs + 1.) * lya_wave
	return lam

def rest_fram_pars(redsft,plot_rest_frame=True):
	if plot_rest_frame: 
		z_plot_rest_frame=redsft #redshift if plot in rest frame, else zero
		lya_toplot=lya_wave # lya 1215.67 Angstrom
	else: 
		z_plot_rest_frame=0.
		lya_toplot=obs_frame(lya_wave,redsft)
	return z_plot_rest_frame,lya_toplot

def cut_lyaforest(spec,lya_toplot,searchrange=50.,adjust_ind=-10,searchlya=False):
	'''
	cut a spec to return only Lya forest section (from Lyb to Lya)
	spec: (lam,flux,flux_err(,disp,exptime))
	searchrange: search lya peak at (lya-searchrange,lya+searchrange), in Angstrom
	return only lya forest spectrum
	adjust_ind: how many more values to the right of lya peak / left of lyb peak
	'''
	lam=spec[0]
	flux=spec[1]

	if searchlya:
		# search for lya peak by the max value
		therange = [lya_toplot-searchrange,lya_toplot+searchrange]
		searchindex = np.where((lam>therange[0])*(lam<therange[1]))[0]
		if len(searchindex)!=0:
			#raise ValueError("No Lya in the spectrum.")
			lyaindex = searchindex[np.argmax(flux[searchindex])]
			foundlya = lam[lyaindex]
			lya_toplot = foundlya
		else: foundlya = lya_toplot

	lyb_toplot = lya2lyb(lya_toplot)
	cuttedind = np.where((lyb_toplot <= lam) & (lam <= lya_toplot))[0]
	if len(cuttedind)==0:
		newspec = [np.array([])] * len(spec)
	else:
		lybindex = cuttedind[0]
		lyaindex = cuttedind[-1]

		right_edge = np.min([len(lam), (lyaindex + adjust_ind + 1)])# right wave index to cut
		left_edge = np.max([0, (lybindex - adjust_ind)])# left wave index to cut
		newspec=[]
		for item in range(len(spec)): newspec.append(spec[item][left_edge:right_edge]) # do cut
	newspec = np.array(newspec)
	if searchlya: return newspec, foundlya
	else: return newspec

def fit_poly(spec, local_dist=5, poly_deg=4):
	'''
	spec: lam, flux, flux_err(, disp, exptime)
	local_dist: min_distance for peak_local_max
	'''
	lam=spec[0]
	flux=spec[1]
	# correct poly degree for small sample
	npoints = len(lam)
	if npoints <= poly_deg: poly_deg = npoints - 1

	# use only local max
	ipeak=peak_local_max(flux,min_distance=local_dist).flatten()
	if len(ipeak)==0: ipeak=list(range(len(lam))) # can't find local max: use whole spec
	lam_topoly=lam[ipeak]
	flux_topoly=flux[ipeak]

	ppoly=np.polyfit(lam_topoly,flux_topoly,poly_deg)
	return ppoly

def cut_spec(spec,lamlim):
	'''
	cut a spec [lam, flux, ...] with lamlim [min, max]
	'''
	spec=np.array(spec)
	initial_dim=len(spec.shape)
	spec=np.atleast_2d(spec)
	specut=spec.T[(spec[0]>=lamlim[0])*(spec[0]<=lamlim[1])].T
	if specut.shape[0]==1 and initial_dim==1: specut=np.squeeze(specut,axis=0) # if spec has only lam
	return specut

def connect_chunks_old(specs):
	'''
	connect chunked spectra to one
	------
	Input specs: [(lam, flux, flux_err, ...), (lam, flux, flux_err, ...), ...]
	Output connected_spec: [lam, flux, flux_err, ...] connected
	'''
	connected_spec = []
	for idim in range(len(specs[0])): # loop through lam, flux, ...
		connected_dim = []
		for chunk in specs: connected_dim.append(chunk[idim]) # loop through chunks
		connected_spec.append(np.hstack(connected_dim))
	indsortlam = np.argsort(connected_spec[0])
	for idim in range(len(connected_spec)): connected_spec[idim] = connected_spec[idim][indsortlam] # sort by lam
	return connected_spec

def connect_chunks(specs):
	'''
	connect chunked spectra to one
	------
	Input specs: [2darray(lam, flux, flux_err, ...), 2darray(lam, flux, flux_err, ...), ...]
	Output connected_spec: 2darray(lam, flux, flux_err, ...) connected
	'''
	newspec = np.hstack(specs)
	newspec = newspec.T[np.argsort(newspec[0])].T
	return newspec

def add_shot_noise(flux, nphot, sky=1e-12, return_error=False):
	'''
	flux should be normalized to [0, 1]
	sky - value in [0, 1], squeezes the spectrum to [sky, 1]
	'''
	if np.all(nphot==np.inf): # no error
		flux_werr = flux
		error = np.zeros(flux.shape)
	else:
		flux_nphot = (flux + sky) / (1. + sky) * nphot
		err_nphot = flux_nphot**0.5
		#print('flux percentage error after adding noise:', np.mean(np.abs(err_nphot*np.random.normal(0.0,1.0,len(flux_nphot)))/flux_nphot))
		#flux_nphot=flux_nphot+err_nphot*np.random.normal(0.0,1.0,len(flux_nphot)) # use normal with sigma=sqrt(flux_nphot)
		flux_nphot_werr = np.random.poisson(flux_nphot) # use poisson
		flux_werr = flux_nphot_werr / nphot * (1. + sky) - sky
		error = err_nphot / nphot
	if return_error: return flux_werr, error
	else: return flux_werr

def convert_flux(toconvert, tounit, lamornu):
	'''
	Convert between F_nu and F_lam
	toconvert - astropy quantity to be converted
	tounit - string or astropy unit, the unit to convert to
	lamornu - astropy quantity, the wavelength or frequency at which to convert the flux
	Returns - astropy quantity, the converted quantity
	'''
	return toconvert.to(tounit, equivalencies=u.spectral_density(lamornu))

def detect_gap(lam, threshold=0.5):
	'''
	detect gaps in the spectra
	threshold - (AA) for gap detection
	Return - {igroup: [laminds], ...} e.g. {0: [0, 1, 2], 1: [3, 4, 5, 6], ...}
	'''
	diff = np.diff(lam)
	criteria = (np.abs(diff) >= threshold) # [False, True, ...] values
	gapstartlams = lam[:-1][criteria] # size:ngap
	def assign_groupind(lamind):
		for igroup, gapstartlam in enumerate(gapstartlams):
		    if lam[lamind] <= gapstartlam: return igroup
		return len(gapstartlams)
	groups = {igroup: list(laminds) for igroup, laminds in itertools.groupby(range(len(lam)), assign_groupind)} # {igroup: [laminds], ...}
	return groups

def divide_spec(spec, groups):
	'''
	Divide spectra by given groups
	spec - lam (, flux, ...) 1d or 2d array
	groups - {igroup: [inds], ...} e.g. {0: [0, 1, 2], 1: [3, 4, 5, 6], ...}
	Return - [spec1, spec2, ...] divided by groups
	'''
	spec = np.atleast_2d(spec)
	divided = [spec.T[groups[igroup]].T for igroup in groups.keys()]
	return divided

def fill_gap(gaplamranges, parray, linedensity=None):
	'''
	fill gap for KOA template
	gaplamranges - [[left, right], ...] in AA
	parray - lam0, ... shape:(nparas, nlines)
	linedensity - nlines per AA, if None, compute from parray (should be in rest frame)
	'''
	if type(linedensity)==type(None): # compute line density by parray
		lam0s = parray[0]
		nlines = len(lam0s)
		lamwidth_withline = (lya_wave - lyb_wave) - np.diff(gaplamranges, axis=1).sum() # lamwidth_withline = lamwidth_lyaforest - lamwidth_gaps
		linedensity = nlines / lamwidth_withline
	for gaplamrange in gaplamranges:
		nl_gap = int(np.round(linedensity * float(np.diff(gaplamrange)))) # nlines in the gap

		# generate paras in the gap
		lam0s_gap = np.random.uniform(*gaplamrange, nl_gap)
		otherpara_gap = np.array([np.random.choice(x, nl_gap) for x in parray[1:]]) # random choice with replacement
		parray_gap = np.vstack([lam0s_gap, otherpara_gap])

		parray = np.hstack([parray, parray_gap])

	return parray

def within_range(arr, therange, closed='both'):
	if closed=='both': criteria = ((therange[0] <= arr) & (arr <= therange[1]))
	elif closed=='left': criteria = ((therange[0] <= arr) & (arr < therange[1]))
	elif closed=='right': criteria = ((therange[0] < arr) & (arr <= therange[1]))
	elif closed.lower()=='no' or closed.lower()=='none': criteria = ((therange[0] < arr) & (arr < therange[1]))
	return criteria

def mask_lamrange(spec, lamranges, method='delete'):
	'''
	spec - lam, flux, ...
	lamranges - [[lam_left, lam_right], [lam_left, lam_right], ...], ranges that should be masked
	method - 'delete' of 'maskedarray'
	'''
	if len(lamranges)==0: return spec
	lam = spec[0]
	masked = np.any(np.array([within_range(lam, x) for x in lamranges]), axis=0)
	if method=='delete': return spec.T[~masked].T
	masked = np.repeat(np.atleast_2d(masked), len(spec), axis=0) # reshape to spec.shape
	if method=='maskedarray': return np.ma.masked_where(masked, spec)
	else: raise Exception('method not recognized')

def add_datapoint(spec_sparse, spec_dense, toadd):
	'''
	add points to peak max points for continuum fitting
	spec_* - assuming only have lam and flux
	spec_sparse - peak max points
	spec_dense - spectra
	toadd - list of scalar or list; if scalar: wavelength point to add; if list: [lam, flux] of point to add, flux in keck flux (if have SDSS spectra) or norm by M1450 (if don't have SDSS spectra)
	'''
	lam_sparse = spec_sparse[0]; flux_sparse = spec_sparse[1]
	lam_dense = spec_dense[0]; flux_dense = spec_dense[1]
	for item in toadd:
		if np.array(item).size==1: # item==lam0
			lamdiff = np.abs(lam_dense - item)
			if min(lamdiff) > 1.: break # no lam_dense close to item (lam0)
			ind = np.argmin(lamdiff)
			lam_sparse = np.append(lam_sparse, lam_dense[ind])
			flux_sparse = np.append(flux_sparse, flux_dense[ind])
		else: # item==[lam0, flux0]
			lamdiff = np.abs(lam_dense - item[0])
			if min(lamdiff) > 1.: break # no lam_dense close to item (lam0)
			lam_sparse = np.append(lam_sparse, item[0])
			flux_sparse = np.append(flux_sparse, item[1])
	spec_sparse = np.array([lam_sparse, flux_sparse])
	# sort lam
	spec_sparse = spec_sparse.T[lam_sparse.argsort()].T
	return spec_sparse

def flux_smooth(flux, width):
	# smooth flux with 2-sigma width (in pixels)
	w = np.ones(int(round(width))) # flat kernel
	w = gaussian(int(round(width*4.)), width/2.) # +/- 4sigma range
	return np.convolve(w/w.sum(),flux,mode='same')

def varysigma_smooth(flux, sigma):
	'''
	do gaussian convolution with different sigmas at different pixels, where gaussian defined by pixel
	sigma - in pixel
	'''
	npix = len(flux)
	flux_out = np.zeros(npix)
	for ipix in range(npix):
		npix_kernel = int(round(sigma[ipix] * 10.))
		if (npix_kernel % 2) == 0: npix_kernel += 1 # make it always odd
		kernel = gaussian(npix_kernel, sigma[ipix]) # +/- 5sigma range
		kernel = flux[ipix] * kernel / kernel.sum() # normalized to pixel flux value
		ind_mid = int((npix_kernel - 1) / 2) # index in kernel defining mid point
		iflux_tochange = np.arange(npix_kernel) - ind_mid + ipix # indexes of flux to be changed by this kernel, size:npix_kernel
		# exclude edge pixels
		within_flux = (iflux_tochange >=0) & (iflux_tochange < npix) # exclude edge pixels, size <= npix_kernel
		kernel = kernel[within_flux]
		iflux_tochange = iflux_tochange[within_flux]
		# assign kernel
		flux_out[iflux_tochange] += kernel
	return flux_out

def varysigma_smooth_bylam(lam, flux, sigma):
	'''
	do gaussian convolution with different sigmas at different pixels, where gaussian defined by lambda
	sigma - in lam unit
	'''
	tosave = path + 'paras/smoothed_flux_%.10e.pickle'%flux[0]
	if os.path.exists(tosave):
		flux_out = pkload(tosave)
	else:
		npix = len(lam)
		flux_out = np.zeros(npix)
		for ipix in range(npix):
			kernel = Gaussian1D(mean=lam[ipix], stddev=sigma[ipix])(lam)
			kernel = flux[ipix] * kernel / kernel.sum() # normalized to pixel flux value
			flux_out += kernel
		pkdump(flux_out, tosave)
	return flux_out

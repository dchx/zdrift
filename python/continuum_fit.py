from utils import *
import m1450
import read_koa as rk
import read_sdss as rs
import read_elqs as re
import spec_utils as su
from scipy import interpolate
from scipy.ndimage import median_filter

def trim_koaspec(koajobid, stackchan=0, plot_rest_frame=True, plot_lya_forest=True, smooth=True, keck_catalog=keck_catalog):
	'''
	input koajobid, output [lam, flux, flux_err, disp, exptime, arclamp] connected
	'''
	matched = get_matched(keck_catalog)[1]
	item = matched[matched['KOAjobID']==koajobid]
	z_plot_rest_frame, lya_toplot = su.rest_fram_pars(item['z'],plot_rest_frame)
	if plot_lya_forest: lya_tocut = lya_toplot
	else: lya_tocut = None
	koa_data = rk.read_koa_jobid(koajobid,stackchan=stackchan,z_plot_rest_frame=z_plot_rest_frame,lya_tocut=lya_tocut,smooth=smooth, keck_catalog=keck_catalog)
	if len(koa_data) == 0: raise IndexError('koa_data is empty.')
	koa_data = rk.cut_wave_by_snr(koa_data) # cut wavelength by snr
	koa_spec = su.connect_chunks_old(koa_data) # connect chunks
	return koa_spec

def npt2deg(npt, func='abs'): 
	'''
	For polynomial fitting, assign poly_degree (deg) from number of data points (npt),
	using a sigoid function (logistic, arctan or abs) with slope of 1 at 0 and upper limit of 99 (max for polyfit)
	'''
	upper_limit = 99.
	x = npt - 1.
	# logistic function, largest degs
	if func=='logistic':
		deg = upper_limit * (2. / (1. + np.exp(- 2. * x / upper_limit)) - 1.)
	# arctan function, middle degs
	if func=='arctan':
		factor = upper_limit / (np.pi/2.) 
		deg = factor * np.arctan(x / factor)
	# abs function, smallest degs
	elif func=='abs':
		deg = x / (1. + np.abs(x / upper_limit))
	deg = np.round(deg)
	return deg 

def contpar2cont(contpar, mode, lam):
	if mode=='poly': return np.polyval(contpar, lam)
	else: return interpolate.interp1d(*contpar, mode, fill_value='extrapolate')(lam)

def fit_continuum(lam, flux, local_dist=100, poly_deg=None, mode='poly', plot=False, figaxes=None, contpar_tosave=None, spec_nosmooth=None, ppoly_ratio=[1.], args=None):
	'''
	local_dist: min_distance for peak_local_max in pixel
	mode: poly, linear or cubic
	figaxes: if not None, use these to plot, else create fig
	args - args class for masking and adding points
	spec_nosmooth - only used in plotting
	poly_deg - degree for polynomial fit. if None, automatically set poly_deg
	'''
	#flux = su.flux_smooth(flux, width=10) # width in pixels
	ipeak = np.sort(np.r_[0,peak_local_max(flux,min_distance=local_dist).flatten()])
	ipeak = np.append(ipeak, -1)
	if len(ipeak)==0: ipeak=list(range(len(lam))) # can't find local max: use whole spec
	lam_peak = lam[ipeak]
	flux_peak = flux[ipeak]

	# mask local max points
	if type(args)!=type(None):
		lam_tofit, flux_tofit = su.mask_lamrange(np.vstack([lam_peak, flux_peak]), args.contmask, 'delete') # mask some peak points
		lam_tofit, flux_tofit = su.add_datapoint([lam_tofit, flux_tofit], [lam, flux], args.contadd) # add points to fit continuum
	else:
		lam_tofit, flux_tofit = lam_peak, flux_peak
	# fit polynomial
	if mode=='poly':
		npoints = len(lam_tofit)
		if poly_deg is None: poly_deg = npt2deg(npoints) # redefine poly_deg
		print('poly_deg:', poly_deg)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			contpar = np.polyfit(lam_tofit, flux_tofit, poly_deg)
	# connect local max by interpolation, no residuals
	else: contpar = [lam_tofit, flux_tofit]
	contpar_pack = [contpar, mode]
	if contpar_tosave!=None: pkdump(contpar_pack, contpar_tosave)
	flux_fitted = contpar2cont(*contpar_pack, lam)
	flux_tofit_fitted = contpar2cont(*contpar_pack, lam_tofit)

	if plot:
		if np.any(figaxes): fig, axes = figaxes
		else:
			if mode=='poly': fig,axes = plt.subplots(3,1,figsize=(12,6),sharex=True,gridspec_kw={'hspace':0,'height_ratios':[0.4,0.4,0.2]})
			else: fig,axes = plt.subplots(2,1,figsize=(12,4),sharex=True,gridspec_kw={'hspace':0})
		sdssflux = True # whether to plot flux in sdss unit
		# plot unsmoothed spec
		if type(spec_nosmooth)!=type(None):
			lam_nosmooth = spec_nosmooth[0]
			flux_nosmooth = spec_nosmooth[1]
			flux_fit_nosmooth = contpar2cont(*contpar_pack, lam_nosmooth)
			if sdssflux: axes[0].plot(lam_nosmooth, flux_nosmooth / np.polyval(ppoly_ratio, lam_nosmooth), 'k', lw=0.2)
			else: axes[0].plot(lam_nosmooth, flux_nosmooth, 'k', lw=0.2)
			axes[1].plot(lam_nosmooth, flux_nosmooth/flux_fit_nosmooth, 'k', lw=0.2)
		# plot fit
		pfmt = {'linestyle': '', 'marker': 'o', 'mfc': 'w', 'mec': 'g', 'ms': 5} # color of local max points
		fitted_fmt = 'b'
		if sdssflux:
			axes[0].plot(lam, flux / np.polyval(ppoly_ratio, lam), 'r', lw=1)
			axes[0].plot(lam, flux_fitted / np.polyval(ppoly_ratio, lam), fitted_fmt) # fitted
			axes[0].plot(lam_tofit, flux_tofit / np.polyval(ppoly_ratio, lam_tofit), **pfmt)
			axes[0].set_ylim([-0.2*np.max(flux_fitted / np.polyval(ppoly_ratio, lam)), 1.2*np.max(flux_fitted / np.polyval(ppoly_ratio, lam))])
			axes[0].set_ylabel('Flux\n($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
		else:
			axes[0].plot(lam, flux, 'r', lw=1)
			axes[0].plot(lam, flux_fitted, fitted_fmt) # fitted
			axes[0].plot(lam_tofit, flux_tofit, **pfmt)
			axes[0].set_ylim([-0.3*np.max(flux_fitted), 1.3*np.max(flux_fitted)])
			axes[0].set_ylabel('Flux (counts)')
		axes[0].set_xlim([su.lyb_wave - 5., su.lya_wave + 5.])
		# plot normed
		axes[1].axhline(1, c=fitted_fmt) # fitted
		axes[1].plot(lam, flux/flux_fitted, 'r', lw=1) # plot normed spec
		axes[1].plot(lam_tofit, flux_tofit/flux_tofit_fitted, **pfmt)
		axes[1].set_ylim([-0.4, 1.4])
		axes[1].set_ylabel('Normalized flux')
		# plot residual
		if mode=='poly':
			axes[2].axhline(1,c=fitted_fmt) # fitted
			axes[2].plot(lam_tofit, flux_tofit/flux_tofit_fitted, **pfmt)
			axes[2].set_ylabel('Residuals')
			ylim = axes[2].get_ylim()
			axes[2].set_ylim(min(0.88, ylim[0]), max(1.12, ylim[1]))
		axes[-1].set_xlabel('Rest frame wavelength ($\mathrm{\AA}$)')
		fig.tight_layout()

		# sanity check: plot residual vs flux
		plot_res_vs_f = True
		if plot_res_vs_f:
			residual = flux_tofit - flux_tofit_fitted
			figt, axest = plt.subplots(1, 2, figsize=(12, 6))
			if sdssflux:
				# absolute residual vs flux
				axest[0].plot(flux_tofit / np.polyval(ppoly_ratio, lam_tofit), np.abs(residual) / np.polyval(ppoly_ratio, lam_tofit), 'ok')
				axest[0].set_xlabel('Flux ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
				axest[0].set_ylabel('abs(Residual) ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
				# relateve residual vs flux
				axest[1].plot(flux_tofit / np.polyval(ppoly_ratio, lam_tofit), np.abs(residual)/flux_tofit_fitted / np.polyval(ppoly_ratio, lam_tofit), 'ok')
				axest[1].set_xlabel('Flux ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
				axest[1].set_ylabel('abs(Residual/Flux)')
			figt.tight_layout()
			polydegtxt = '' if poly_deg is None else '_deg%d'%poly_deg
			res_vs_f_tosave = path + 'plots/contfit_residual_vs_flux_36576%s.pdf'%(polydegtxt)
			figt.savefig(res_vs_f_tosave); print('Saved: %s'%res_vs_f_tosave)
		
	return flux_fitted, contpar_pack

def scale_flux(item, spec):
	'''
	if no SDSS spectra, scale flux according to m1450
	spec - lam should be in rest frame
	'''
	f1450 = m1450.m14502sdss(item['M1450'], item['z']) # 1450AA flux in 1e-17 erg / (cm2 s AA)
	spec = m1450.norm2f1450(spec, f1450)
	return spec

def fluxpars2flux(fluxpars, lam):
	'''
	fluxpars - lamranges, contpar_packs, ppoly_ratios
	    lamranges - [[lam_min, lam_max], ...]
	    contpar_packs - [[contpar, fitcontmode], ...]
	    ppoly_ratios - [ppoly_ratio, ...]
	'''
	fluxpars = copy.deepcopy(fluxpars)
	lamranges = fluxpars[0]
	contpar_packs = fluxpars[1]
	if len(fluxpars)==3:
		havesdss = True
		ppoly_ratios = fluxpars[2]
	else: havesdss = False
	contfuncs = []
	for ispec in range(len(lamranges)):
		# wrapup contfit and flux calibration
		ppoly_ratio = ppoly_ratios[ispec] if havesdss else None
		contpar_pack = contpar_packs[ispec]
		def cont_calibrated_func(lam, contpar_pack=contpar_pack, ppoly_ratio=ppoly_ratio):
			cont_fit = contpar2cont(*contpar_pack, lam)
			if type(ppoly_ratio)!=type(None):
				ratio_fit = np.polyval(ppoly_ratio, lam)
				cont_fit = cont_fit / ratio_fit
			return cont_fit
		contfuncs.append(cont_calibrated_func)

	# connect gaps in continuum by a straight line
	# ---- middle gaps
	ispec = 0
	while ispec < len(contfuncs) - 1:
		# assume specs sorted by lam
		gap_lamrange = [lamranges[ispec][1], lamranges[ispec + 1][0]]
		gap_fluxrange = [contfuncs[ispec + ind](lam) for ind, lam in enumerate(gap_lamrange)]
		gap_ppoly = np.polyfit(gap_lamrange, gap_fluxrange, 1)
		def gap_func(lam, gap_ppoly=gap_ppoly): return np.polyval(gap_ppoly, lam)
		contfuncs.insert(ispec + 1, gap_func)
		lamranges.insert(ispec + 1, gap_lamrange)
		ispec += 2
	# ---- left edge
	lam_leftedge = lamranges[0][0]
	flux_leftedge = contfuncs[0](lam_leftedge)
	def leftedge_func(lam, flux_leftedge=flux_leftedge):
		try: return flux_leftedge * np.ones(len(lam))
		except TypeError: return flux_leftedge
	lamranges.insert(0, [-np.inf, lam_leftedge])
	contfuncs.insert(0, leftedge_func)
	# ---- right edge
	lam_rightedge = lamranges[-1][-1]
	flux_rightedge = contfuncs[-1](lam_rightedge)
	def rightedge_func(lam, flux_rightedge=flux_rightedge):
		try: return flux_rightedge * np.ones(len(lam))
		except TypeError: return flux_rightedge
	lamranges.append([lam_rightedge, np.inf])
	contfuncs.append(rightedge_func)

	# connected continuum function, save cont function
	# --- parameters are contfuncs, lamranges
	flux = []
	for ispec in range(len(contfuncs)):
		lam_ispec = lam[(lamranges[ispec][0] < lam) & (lam <= lamranges[ispec][1])]
		flux_ispec = contfuncs[ispec](lam_ispec)
		flux = np.append(flux, flux_ispec)
	if len(lam)!=len(flux): raise Exception('generated continuum len(lam)!=len(flux)')
	return flux

def fluxpar_filename(name, polydegtxt=''):
	'''
	name - sdss_filename if use sdss spectra else KOAjobID
	polydeg - degree of polyfit for continuum fit
	'''
	#polydegtxt = '' if polydeg is None else '_deg%d'%(polydeg)
	name = '.'.join(name.split('.')[:-1]) if type(name)==str else 'koa%d'%(name)
	prefix = path + 'paras/fluxpar_%s_restframe.pickle'%(name + polydegtxt)
	return prefix

def fit_sdss_cont(sdss_filename, local_dist=20, poly_deg=10, fitcont_mode='poly', plot=False, save_fluxpars=True):
	'''
	fit sdss spectra continuum
	  sdss_spec - lam, flux, flux_err
	'''
	sdssfile = path + 'data/sdss/' + sdss_filename
	sdss = rs.read_sdss_file_class(sdssfile)
	sdss.lam = su.rest_frame(sdss.lam, sdss.zqso) # to rest frame
	sdss_spec = np.vstack([sdss.lam, sdss.flux, sdss.flux_err]) # (3, npix)

	# add cont mask and contadd, in rest frame
	class args: contmask = []; contadd = []; voigtmask = []
	if sdss_filename == 'spec-5328-55982-0562.fits': # SDSS J095937.11+131215.5
		args.contmask = [[1025., 1026.]]
		args.contadd = [1029.11, 1189.54, 1195.52, 1208.55, 1211.35]

	# cut spectra from lyb to lya, in rest frame
	spec_cut = su.cut_lyaforest(sdss_spec, su.lya_wave, adjust_ind=0, searchlya=False) # (3, npix)

	# fit continuum in rest frame
	flux_fitted, contpar_pack = fit_continuum(*spec_cut[:2], local_dist=local_dist, poly_deg=poly_deg, mode=fitcont_mode, plot=plot, contpar_tosave=None, args=args)

	# save fluxpars
	if save_fluxpars:
		lamrange = [np.min(spec_cut[0]), np.max(spec_cut[0])]
		fluxpars = [[lamrange], [contpar_pack]]
		fluxpar_tosave = fluxpar_filename(sdss_filename)
		pkdump(fluxpars, fluxpar_tosave)

def get_keck_spec(item, local_dist=100, poly_deg=None, fitcont_mode='poly', detect_gap=True, rest_frame=True, smoothwidth=30, plot=False, normalize=True, save_fluxpars=False, return_gaprange=False):
	'''
	detect_gap - whether to detect gaps in the spectra and divide spectra when fitting continuum
	use item[KOAjobID, catalog, z, SDSS, M1450]
	if not rest_frame: use item[z_origin]
	'''
	koajobid = item['KOAjobID']
	try: catalog = item['catalog']
	except Exception: item.at['catalog'] = 'elqs'
	koa_spec = rk.read_koa_df(item, rest_frame=True, cut_lya=False, smoothwidth=smoothwidth) # get 1d spectra connected, array, lam, flux, error, disp; use item[z, KOAjobID, catalog]
	koa_spec_nosmooth = rk.read_koa_df(item, rest_frame=True, cut_lya=False, smoothwidth=0) # get 1d spectra connected, array, lam, flux, error, disp

	#  if no SDSS, scale flux to m1450
	havesdss = rs.check_item_havesdss(item) # use item[SDSS]
	havem1450 = ('M1450' in item.keys()) and not np.isnan(item['M1450'])
	if not havesdss and havem1450: # scale flux to m1450
		print('Scale flux for:', item)
		koa_spec = scale_flux(item, koa_spec) # should be in rest frame, use item[M1450, z]
		koa_spec_nosmooth = scale_flux(item, koa_spec_nosmooth) # should be in rest frame

	# wrap up koa_spec
	if len(koa_spec[0]) == 0: raise Exception('spectrum is zero length')
	if not normalize:
		if not rest_frame: koa_spec[0] = su.obs_frame(koa_spec[0], item['z_origin'])
		return koa_spec[:3] # lam, flux, flux_err, disp in rest frame

	# if have SDSS, convolve keck_spec_nosmooth flux by SDSS resolution (in obs frame), and compute keck.flux_smoothbysdss / sdss.flux_keckgrid ratio
	if havesdss:
		sdssfile = path + 'data/sdss/' + item['SDSS']
		sdss = rs.read_sdss_file_class(sdssfile)
		keck, sdss = rs.smoothKeckBysdss_calcRatio(su.obs_frame(koa_spec_nosmooth[0], item['z']), koa_spec_nosmooth[1], sdss) # sdss in obs frame, convert keck to obs frame
		koa_spec_nosmooth = np.vstack([koa_spec_nosmooth, keck.flux_smoothbysdss, keck.ratio]) # lam, flux, error, disp, flux_smoothbysdss, ratio, in rest frame

	# detect gaps and divide spec
	if detect_gap:
		gap_threshold = 1.6
		specs = su.divide_spec(koa_spec, su.detect_gap(koa_spec[0], threshold=gap_threshold)) # [spec1, spec2, ...] divided by groups
		specs_nosmooth = su.divide_spec(koa_spec_nosmooth, su.detect_gap(koa_spec_nosmooth[0], threshold=gap_threshold)) # [spec1, spec2, ...] divided by groups
	else:
		specs = [koa_spec]
		specs_nosmooth = [koa_spec_nosmooth]
	if koajobid==7260: # delete a small chunk
		specs.pop(1)
		specs_nosmooth.pop(1)

	# if have SDSS, smooth the keck.flux_smoothbysdss / sdss.flux_keckgrid ratio, no matter frame
	if havesdss:
		for ispec in range(len(specs_nosmooth)):
			ratio = specs_nosmooth[ispec][-1]
			ratio_filt = median_filter(ratio, 5001) # median filtered ratio
			specs_nosmooth[ispec] = np.vstack([specs_nosmooth[ispec], ratio_filt]) # lam, flux, error, disp, flux_smoothbysdss, ratio, ratio_filt

	# cut spectra from lyb to lya, in rest frame
	if koajobid in [104297, 125810, 20463, 32118, 119681, 9075]: searchlya = True # search for lya peak
	else: searchlya = False
	# get general searched lya from smoothed spec
	if searchlya:
		_, lya_tocut = su.cut_lyaforest(koa_spec, su.lya_wave, searchrange=5., adjust_ind=0, searchlya=searchlya)
		zqso = lya_tocut / su.lya_wave * (1. + item['z']) - 1.
	else:
		lya_tocut = su.lya_wave
		zqso = item['z']
	# cut individual sections
	specs_cut = []
	specs_nosmooth_cut = []
	for ispec in range(len(specs)):
		spec_cut = su.cut_lyaforest(specs[ispec], lya_tocut, adjust_ind=0, searchlya=False) # cut lya peak, in rest frame
		if len(spec_cut[0])!=0: # only include non-zero-size spec sections
			spec_nosmooth_cut = su.cut_lyaforest(specs_nosmooth[ispec], lya_tocut, adjust_ind=0, searchlya=False) # cut lya peak, in rest frame
			if searchlya: # correct lam by lya_found in rest frame
				spec_cut[0] = su.rest_frame(su.obs_frame(spec_cut[0], item['z']), zqso)
				spec_nosmooth_cut[0] = su.rest_frame(su.obs_frame(spec_nosmooth_cut[0], item['z']), zqso)
			specs_cut.append(spec_cut)
			specs_nosmooth_cut.append(spec_nosmooth_cut)
	specs = specs_cut # list of specs, separated by gaps, within lyb to lya
	specs_nosmooth = specs_nosmooth_cut # list of specs

	# setup plot for keck flux calibration by sdss
	if havesdss:
		if plot:
			fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, gridspec_kw={'hspace':0, 'height_ratios':[0.4,0.4,0.2]})

	# if have SDSS, fit polynomial to ratio, in rest frame
	if havesdss:
		ppoly_ratios = []
		for ispec in range(len(specs_nosmooth)):
			klam = specs_nosmooth[ispec][0]
			kflux = specs_nosmooth[ispec][1]
			ratio_filt = specs_nosmooth[ispec][-1]
			kflux_smoothbysdss = specs_nosmooth[ispec][4]
			# fit polynomial to ratio
			polydeg_ratio = 4
			if item['KOAjobID']==32118 and ispec==1: polydeg_ratio = 6
			ppoly_ratio = np.polyfit(klam, ratio_filt, polydeg_ratio) # input klam (rest frame) and ratio_filt
			ppoly_ratios.append(ppoly_ratio)
			ratio_fit = np.polyval(ppoly_ratio, klam)
			kflux_calibrate = kflux / ratio_fit
			if plot:
				# axes 0 (keck flux)
				axes[0].plot(klam, kflux, 'k', lw=0.2) # keck spec nosmooth
				axes[0].plot(klam, kflux_smoothbysdss, 'r') # keck spec smoothed by sdss
				axes[0].set_ylim([-0.2*np.max(kflux_smoothbysdss), 1.2*np.max(kflux_smoothbysdss)])
				axes[0].set_ylabel('Keck flux\n(counts)')
				# axes 1 (sdss flux)
				axes[1].plot(klam, kflux_calibrate, 'k', lw=0.2) # calibrated keck flux
				if ispec == len(specs_nosmooth)-1:
					axes[1].plot(su.rest_frame(sdss.lam, zqso), sdss.flux, 'r') # sdss spec
				sdssmax = sdss.flux[np.isclose(su.rest_frame(sdss.lam, zqso), su.lya_wave, atol=5.)] # sdss lya peak flux
				axes[1].set_ylim([-0.2*np.max(sdssmax), 1.2*np.max(sdssmax)])
				axes[1].set_ylabel('SDSS flux\n($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
				# axes 2 (ratio)
				axes[2].plot(klam, specs_nosmooth[ispec][5], 'r', lw=1) # ratio
				#axes[2].plot(klam, ratio_filt, 'c', lw=3) # smoothed (median-filtered) ratio
				axes[2].plot(klam, ratio_fit, 'k') # fitted ratio
				axes[2].set_ylim([0.1*np.min(ratio_filt), 1.6*np.max(ratio_filt)])
				axes[2].set_ylabel('Flux ratio')
				axes[-1].set_xlabel('Rest frame wavelength ($\mathrm{\AA}$)')
		if plot:
			axes[0].set_xlim([su.lyb_wave - 5., su.lya_wave + 5.])
			fig.tight_layout()

	# setup plot for fit continuum
	plot_fitcont = plot
	#plot_fitcont = False
	if plot_fitcont:
		if fitcont_mode=='poly': figaxes = plt.subplots(3,1,figsize=(12,6),sharex=True,gridspec_kw={'hspace':0,'height_ratios':[0.4,0.4,0.2]})
		else: figaxes = plt.subplots(2,1,figsize=(12,4),sharex=True,gridspec_kw={'hspace':0})
	else: figaxes = None

	# fit continuum and normalize, in rest frame
	# ---- mask or add args for continuum fit
	if searchlya and not item['KOAjobID']==119681: args = re.mask_add_item(item, zqso) # use item[KOAjobID, z]
	else: args = re.mask_add_item(item) # use item[KOAjobID]
	contpar_packs = []
	lamranges = [] # lam range for each gapped section
	for ispec in range(len(specs)):
		lamranges.append([np.min(specs[ispec][0]), np.max(specs[ispec][0])])
		ppoly_ratio = ppoly_ratios[ispec] if havesdss else [1.]
		# fit continuum on smoothed spec
		flux_fitted, contpar_pack = fit_continuum(*specs[ispec][:2], local_dist=local_dist, poly_deg=poly_deg, mode=fitcont_mode, plot=plot_fitcont, figaxes=figaxes, contpar_tosave=None, spec_nosmooth=specs_nosmooth[ispec], ppoly_ratio=ppoly_ratio, args=args)
		contpar_packs.append(contpar_pack)
		# normalize
		specs[ispec][1:3] /= flux_fitted # normalize for flux, flux_err
		flux_fit_nosmooth = contpar2cont(*contpar_pack, specs_nosmooth[ispec][0])
		specs_nosmooth[ispec][1:3] /= flux_fit_nosmooth # normalize for flux, flux_err

	# generate fluxpars (and save)
	fluxpars = [lamranges, contpar_packs]
	if havesdss: fluxpars.append(ppoly_ratios)

	polydegtxt = '' if poly_deg is None else '_deg%d'%poly_deg
	if plot:
		# plot continuum with random lam grid
		'''
		randlam = np.linspace(su.lyb_wave, su.lya_wave, 1000)
		randflux = fluxpars2flux(fluxpars, randlam)
		if havesdss: axes[1].plot(randlam, randflux, 'k') # plot on flux calibration plot
		else: figaxes[1][0].plot(randlam, randflux, 'k') # plot on fit continuum plot
		'''
		# add twiny axis showing observed frame (of substituting quasars)
		# --- for calibrate flux
		if havesdss:
			axp = axes[0].twiny()
			lamlim_obsframe = su.obs_frame(np.array(axes[0].get_xlim()), item['z'])
			axp.set_xlim(lamlim_obsframe)
			axp.set_xlabel('Observed frame wavelength ($\mathrm{\AA}$)')
			fig.tight_layout()
		# --- for continuum fit
		axp = figaxes[1][0].twiny()
		lamlim_obsframe = su.obs_frame(np.array(figaxes[1][0].get_xlim()), item['z'])
		axp.set_xlim(lamlim_obsframe)
		axp.set_xlabel('Observed frame wavelength ($\mathrm{\AA}$)')
		figaxes[0].tight_layout()
		# savefig
		debug = True
		if havesdss:
			plot_cali_tosave = path + 'plots/calibrate_flux_%d.pdf'%item['KOAjobID']
			if not debug:
				fig.savefig(plot_cali_tosave); print('Saved: %s'%plot_cali_tosave)
		plot_contfit_tosave = path + 'plots/contfit_%d%s.pdf'%(item['KOAjobID'], polydegtxt)
		figaxes[0].savefig(plot_contfit_tosave); print('Saved: %s'%plot_contfit_tosave)

	if save_fluxpars:
		contfunc_tosave = fluxpar_filename(item['KOAjobID'], polydegtxt) # save continuum paras
		pkdump(fluxpars, contfunc_tosave)
		return fluxpars

	# record gap lam ranges [[gapleft, gapright], ...], in rest frame
	edgepoints = np.array(lamranges).flatten()
	edgepoints = np.insert(edgepoints, 0, su.lyb_wave) # insert lyb point, should in rest frame
	edgepoints = np.append(edgepoints, su.lya_wave) # insert lya point, should in rest frame
	gaplamranges = [[edgepoints[i], edgepoints[i+1]] for i in range(0,len(edgepoints),2)] # edgepoints to gaplamranges
	gaplamranges = np.array(gaplamranges)[[~np.isclose(*i, atol=1e-2) for i in gaplamranges]] # drop verry narrow gaps (<0.01AA)
	#gaplamranges = [[np.max(specs_nosmooth[ispec][0]), np.min(specs_nosmooth[ispec+1][0])] for ispec in range(len(specs_nosmooth)-1)]
	# connect again
	koa_spec = su.connect_chunks(specs) # normalized
	koa_spec_nosmooth = su.connect_chunks(specs_nosmooth) # normalized
	# go to observed frame if needed
	if not rest_frame:
		koa_spec[0] = su.obs_frame(koa_spec[0], item['z_origin'])
		koa_spec_nosmooth[0] = su.obs_frame(koa_spec_nosmooth[0], item['z_origin'])
	if return_gaprange: return koa_spec_nosmooth[:3], gaplamranges
	else: return koa_spec_nosmooth[:3] # lam, flux, flux_err. normalized, in rest frame

if __name__ == '__main__':
	import read_ps as rp
	import realll_vs_simll as rvs

	#item = rp.top10vds_N_nosub[rp.top10vds_N_nosub.KOAjobID==9075].iloc[0]
	#item.at['catalog'] = 'elqs'
	kids = rvs.kid_withspec(exclude_weak=False)
	kid = 36576
	poly_deg = 10
	#for kid in kids:
	if 1:
		#item = rp.top10vds_N[rp.top10vds_N.KOAjobID==kid].iloc[0] # if kid in top10vds_N
		item = df_all[df_all['KOAjobID']==kid].iloc[0] # if kid not in top10vds_N; but cannot set rest_frame=False
		print(item.Name)
		#if 1: # main
		for poly_deg in [10]: # debug
			#poly_deg = 20 if kid in [32118, 54447] else 10
			spec = get_keck_spec(item, plot=True, save_fluxpars=False, poly_deg=poly_deg) # plot
			#spec = get_keck_spec(item, plot=False, save_fluxpars=True, poly_deg=poly_deg) # save fluxpars
			#plt.show()
			#tosave = path + 'plots/contfit_%d_%d.pdf'%(ind+1, item['KOAjobID'])
			#plt.savefig(tosave); print('Saved: %s'%tosave)

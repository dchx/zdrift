from utils import *
from scipy import interpolate
from read_koa import cut_wave_by_snr,read_koa_jobid,flux_smooth

def trim_koaspec(koajobid, stackchan=0, plot_rest_frame=True, plot_lya_forest=True, smooth=True, keck_catalog=keck_catalog):
	'''
	input koajobid, output [lam, flux, flux_err, disp, exptime, arclamp] connected
	'''
	matched = get_matched(keck_catalog)[1]
	item = matched[matched['KOAjobID']==koajobid]
	z_plot_rest_frame, lya_toplot = rest_fram_pars(item['z'],plot_rest_frame)
	if plot_lya_forest: lya_tocut = lya_toplot
	else: lya_tocut = None
	koa_data = read_koa_jobid(koajobid,stackchan=stackchan,z_plot_rest_frame=z_plot_rest_frame,lya_tocut=lya_tocut,smooth=smooth, keck_catalog=keck_catalog)
	if len(koa_data) == 0: raise IndexError('koa_data is empty.')
	koa_data = cut_wave_by_snr(koa_data) # cut wavelength by snr
	koa_spec = connect_chunks(koa_data) # connect chunks
	return koa_spec

def fit_continnum(lam, flux, local_dist=100, poly_deg=10, mode='poly', plot=False):
	'''
	local_dist: min_distance for peak_local_max in pixel
	mode: poly, linear or cubic
	'''
	#flux = flux_smooth(flux, width=10) # width in pixels
	ipeak = np.sort(np.r_[0,peak_local_max(flux,min_distance=local_dist).flatten()])
	ipeak = np.append(ipeak, -1)
	if len(ipeak)==0: ipeak=list(range(len(lam))) # can't find local max: use whole spec
	lam_tofit = lam[ipeak]
	flux_tofit = flux[ipeak]

	# fit polynomial
	if mode=='poly': flux_fitted = np.polyval(np.polyfit(lam_tofit, flux_tofit, poly_deg),lam)
	# connect local max by interpolation
	else: flux_fitted = interpolate.interp1d(lam_tofit, flux_tofit, mode, fill_value='extrapolate')(lam)

	if plot:
		if mode=='poly': fig,axes=plt.subplots(3,1,figsize=(12,6),sharex=True,gridspec_kw={'hspace':0,'height_ratios':[0.4,0.4,0.2]})
		else: fig,axes=plt.subplots(2,1,figsize=(12,4),sharex=True,gridspec_kw={'hspace':0})
		# plot fit
		axes[0].plot(lam,flux,lw=0.5);axes[0].plot(lam,flux_fitted,'k');axes[0].plot(lam[ipeak],flux[ipeak],'.r')
		#axes[0].set_yscale('log')
		axes[0].set_ylabel('Keck flux (counts)')
		# plot normed
		axes[1].axhline(1,c='k');axes[1].plot(lam,flux/flux_fitted,lw=0.5);axes[1].plot(lam[ipeak],flux[ipeak]/flux_fitted[ipeak],'.r')
		axes[1].set_ylabel('Normalized\nflux')
		# plot residual
		if mode=='poly':
			axes[2].axhline(1,c='k');axes[2].plot(lam[ipeak],flux[ipeak]/flux_fitted[ipeak],'.r')
			axes[2].set_ylabel('Residuals')
		axes[-1].set_xlabel('$\lambda$ ($\AA$)')
		fig.tight_layout()
	return flux_fitted

def get_keck_spec(i, local_dist=100, poly_deg=10, fitcont_mode='poly', plot_rest_frame=True, plot_lya_forest=True, smooth=True, plot=False, normalize=True):
	z_plot_rest_frame, lya_toplot = rest_fram_pars(matched['z'][i],plot_rest_frame)
	if plot_lya_forest: lya_tocut = lya_toplot
	else: lya_tocut = None

	koajobid = matched['KOAjobID'][i]
	koa_spec = trim_koaspec(koajobid, plot_rest_frame=plot_rest_frame, smooth=smooth) # make spec connected
	koa_spec, lya_found = cut_lya(koa_spec, lya_toplot, adjust_ind=-100, searchlya=True) # cut lya peak
	if len(koa_spec[0]) == 0: raise Exception('spectrum is zero length')
	lam = koa_spec[0]; flux = koa_spec[1]; flux_err = koa_spec[2]
	if not normalize: return lam, flux, flux_err

	# normalize
	flux_fit = fit_continnum(lam, flux, local_dist=local_dist, poly_deg=poly_deg, mode=fitcont_mode, plot=plot)
	flux_normed = flux / flux_fit
	flux_err_normed = flux_err / flux_fit
	return lam, flux_normed, flux_err_normed

if __name__ == '__main__':
	i = 7
	mode = 'poly'
	tosave = path+'plots/contfit_%s_%s.pdf'%(saveid_func(i),mode)
	lam, flux, _ = get_keck_spec(i, fitcont_mode=mode, plot=1, plot_rest_frame=False)
	plt.savefig(tosave);print('Saved:',tosave)
	plt.show()

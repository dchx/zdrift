from utils import *
from spec_utils import *
import spec_utils as su
import astropy.stats as astats
from scipy.ndimage import median_filter

# reference: http://adalace.org/posts/the-sdss-1d-spectrum-of-q1357%2B0525/
sdss_wave_label='Wavelength ($\AA$)'
sdss_flux_label='Flux ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)'

def read_ned_file(filepathname, z_plot_rest_frame=0.):
	'''
	read ned spectra in NED-ASCII format
	'''
	data = np.loadtxt(filepathname, converters={3:bool}).T # converter make '|' to 1
	spec_lam = data[:3] # lam (AA), flux (erg/(cm2 s AA)), flux_err
	spec_nu = data[4:] # nu (Hz), f_nu (W/(m2 Hz)), f_nu_err
	spec_lam[0] = rest_frame(spec_lam[0], z_plot_rest_frame)
	spec_lam[1:3] /= 1e-17 # flux, flux_err to 10**-17 erg/(cm2 s AA)
	return spec_lam

def get_resolution_size(fitsfile):
	'''
	return SDSS resolution element size in AA
	'''
	with fits.open(fitsfile) as hdu: 
		data = hdu[1].data 
		z = hdu[2].data['Z'][0]
		lam = 10**data['loglam'] 
		wdisp = data['wdisp']
		plt.plot(lam, np.gradient(lam)*wdisp) 
		plt.axvline(su.obs_frame(su.lya_wave, z), c='k')
		plt.axvline(su.obs_frame(su.lyb_wave, z), c='k')
		plt.ylabel('resolution element size in AA') 
		plt.show()

def check_item_havesdss(item):
	havesdss = (('SDSS' in item.keys()) and type(item['SDSS'])==str and item['SDSS'].endswith('fits'))
	return havesdss

def read_sdss_file_class(sdssfile):
	class sdss: pass
	# read sdss
	with fits.open(sdssfile) as hdu:
		sdss.zqso = hdu[2].data['Z'][0]
		sdss.lam = 10**hdu[1].data['loglam'] # Angstrom
		sdss.flux = hdu[1].data['flux'] # 10**-17 erg cm-2 s-1 AA-1
		with np.errstate(divide='ignore'):
			sdss.flux_err = 1. / np.sqrt(hdu[1].data['ivar'])
		sdss.wdisp = hdu[1].data['wdisp'] # N sdss pixel per resolution element
	return sdss

def smoothKeckBysdss_calcRatio(klam, kflux, sdss):
	'''
	should be in obs frame
	'''
	class keck:
		lam = klam
		flux = kflux
	sdss.resscale = np.gradient(sdss.lam) * sdss.wdisp # AA per sdss resolution element, in sdss grid
	# exclude 0 resscale values (and their +/- pixels)
	reszero = (sdss.resscale == 0)
	reszero = (reszero | np.hstack([reszero[1:], False]) | np.hstack([False, reszero[:-1]]))
	sdss.lam, sdss.flux, sdss.wdisp, sdss.resscale = np.array([sdss.lam, sdss.flux, sdss.wdisp, sdss.resscale]).T[~reszero].T
	# interp scale to keck grid
	keck.sdssresscale = np.interp(keck.lam, sdss.lam, sdss.resscale) # AA per sdss resolution element, in keck grid
	# smooth keck flux
	keck.flux_smoothbysdss = su.varysigma_smooth_bylam(keck.lam, keck.flux, keck.sdssresscale)
	# interp sdss flux to keck grid
	sflux_kgrid = np.interp(keck.lam, sdss.lam, sdss.flux)
	# compute ratio, should be gap separated
	keck.ratio = keck.flux_smoothbysdss / sflux_kgrid
	return keck, sdss

def calibrate_keck_by_sdss(klam, kflux, sdssfile):
	'''
	calibrate Keck flux by SDSS flux, the two spectra should be of the same object
	keck spec should be without gaps
	'''
	# read sdss
	sdss = read_sdss_file_class(sdssfile)
	# smooth keck flux by sdss resolution and comput flux ratio
	keck, sdss = smoothKeckBysdss_calcRatio(klam, kflux, sdss)

	# detect gap and divide keck spec
	# smooth flux ratio
	ratio_filt = median_filter(keck.ratio, 10001)

	# cut lyb to lya
	keck.lam_cut, ratio_cut, ratio_filt_cut = su.cut_lyaforest(np.vstack([keck.lam, keck.ratio, ratio_filt]), su.obs_frame(su.lya_wave, sdss.zqso), adjust_ind=0, searchlya=False)

	# fit polynomial to ratio
	ratio_cut_fit = np.polyval(np.polyfit(keck.lam_cut, ratio_filt_cut, 3), keck.lam_cut)

	# plot
	fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
	axes[0].plot(keck.lam, keck.sdssresscale)
	axis = axes[0].axis()
	axes[0].plot(sdss.lam, sdss.resscale, lw=1)
	axes[0].set_ylabel('AA per sdss resolution element')
	axes[1].plot(keck.lam, keck.flux, lw=0.5)
	axes[1].plot(keck.lam, keck.flux_smoothbysdss)
	axes[1].plot(sdss.lam, sdss.flux, 'k')
	axes[1].set_ylabel('Keck flux')
	axes[2].plot(keck.lam, keck.ratio)
	axes[2].plot(keck.lam_cut, ratio_filt_cut, lw=1)
	axes[2].plot(keck.lam_cut, ratio_cut_fit, lw=0.5)
	axes[2].set_ylabel('Flux ratio')
	axes[0].axis(axis)
	for ax in axes:
		ax.axvline(su.obs_frame(su.lya_wave, sdss.zqso), c='k')
		ax.axvline(su.obs_frame(su.lyb_wave, sdss.zqso), c='k')
	fig.tight_layout()
	plt.show()

def read_sdss_file(filepathname,z_plot_rest_frame=0.):
	hdus=fits.open(filepathname)
	wave=10**hdus[1].data['loglam'] # Angstrom
	wave=rest_frame(wave,z_plot_rest_frame)
	flux=hdus[1].data['flux'] # 10**-17 erg cm-2 s-1 AA-1
	with np.errstate(divide='ignore'): flux_err=1./np.sqrt(hdus[1].data['ivar'])
	return wave,flux,flux_err

def read_sdss_plate(plate,mjd,fiberid,z_plot_rest_frame=0.):
	#filepathname=path+'data/SDSS/dr14/all/spec-%04d-%05d-%04d.fits'%(plate,mjd,fiberid)
	filepathname=path+'data/sdss/spec-%s-%s-%04d.fits'%(plate,mjd,fiberid)
	return read_sdss_file(filepathname,z_plot_rest_frame=z_plot_rest_frame)

def read_sdss_koajobid(koajobid,z_plot_rest_frame=0.):
	imatched=np.where(koajobid==matched['KOAjobID'])[0][0] # select numtest
	return read_sdss_plate(matched['plate'][imatched],matched['mjd'][imatched],matched['fiberid'][imatched],z_plot_rest_frame=z_plot_rest_frame)
def read_sdss_top11koajobid(koajobid, plot_rest_frame=False):
	top11 = pd.read_csv(path + 'data/top11_substitutes.csv')
	item = top11[top11.KOAjobID==koajobid].iloc[0]
	sdssurl = item.SDSS
	z_plot_rest_frame, _ = rest_fram_pars(item.z, plot_rest_frame)
	if type(sdssurl)==str: # sdss data
		pmf = sdssurl.split('spec-')[1].rstrip('.fits').split('-') # plate, mjd, fiberid
		spec = read_sdss_plate(*pmf, z_plot_rest_frame=z_plot_rest_frame)
	else: # ned data
		rank = item.Rank
		nedfile = glob.glob(path + 'data/sdss/%d.sub.*_NED.txt'%rank)
		if len(nedfile)==1:
			nedfile = nedfile[0]
			spec = read_ned_file(nedfile, z_plot_rest_frame)
	return spec

def sdss_axis_labels(ax,z_plot_rest_frame):
	if z_plot_rest_frame==0: wavelength_descrip='Observed ' # plot in observed frame
	else: wavelength_descrip='Rest Frame ' # plot in rest frame
	ax.set_xlabel(wavelength_descrip+sdss_wave_label)
	ax.set_ylabel(sdss_flux_label)
	
def plot_sdss_spec(sdss_data, ax=None, z_plot_rest_frame=0.):
	# z_plot_rest_frame: redshift if plot in rest frame, else zero
	if type(ax)==type(None):
		generate_fig = True
		fig, ax = plt.subplots(figsize=(12, 4))
	wave, flux, flux_err = sdss_data
	lines=ax.plot(wave,flux,'k',lw=0.5)
	sdss_axis_labels(ax,z_plot_rest_frame)
	if generate_fig:
		fig.tight_layout()
		return fig, ax
	return lines # a list

def main(z_plot_rest_frame=0.):
	for i in range(len(matched))[:2]:
		sdss_data=read_sdss_koajobid(matched['KOAjobID'][i],z_plot_rest_frame=z_plot_rest_frame)
		fig,ax=plt.subplots(figsize=(12,8))
		sdss_lines=plot_sdss_spec(sdss_data,ax,z_plot_rest_frame=z_plot_rest_frame)
		plt.show()
		plt.close(fig)

if __name__=='__main__': main()

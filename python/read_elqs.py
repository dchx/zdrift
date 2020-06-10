from utils import *
from continuum_fit import trim_koaspec
from astropy.coordinates import SkyCoord, Angle

def flux2m1450(F_nu, z, cosmo=cosmology.Planck15):
	'''
	F_nu - flux at rest frame 1450AA in Jy (10^-23 erg s-1 cm-2 Hz-1)
	'''
	mu = cosmo.distmod(z) # distance modulus
	m1450 = -2.5 * np.log10(F_nu / 3631. / (1. + z)) - mu
	return m1450

def m14502flux(m1450, z, cosmo=cosmology.Planck15):
	'''
	Return flux at rest frame 1450AA in Jy (10^-23 erg s-1 cm-2 Hz-1)
	'''
	mu = cosmo.distmod(z) # distance modulus
	f1450 = 3631. * (1. + z) * 10.**((m1450 + mu)/ -2.5)
	return f1450

def within_range(value, therange):
	if therange[0] <= value and value <= therange[1]: return True
	else: return False

def recover_koajobid():
	'''
	recover koajobid after hard disk broken
	'''
	koajobid_notfound = [41580, 5707, 113104, 123929, 124485, 125016, 14150, 27266]
	kn = [123929, 125016]
	koafolders = glob.glob(path+'data/Keck/'+keck_catalog+'/KOA_*/')
	searchradius = Angle(30, unit='arcsec')
	d_coor = SkyCoord(d['RASdeg'], d['DESdeg'], unit=u.deg)
	for koafolder in koafolders:
		koajobid = int(koafolder.split('KOA_')[1].rstrip('/'))
		fitslist = glob.glob(koafolder + '/HIRES/raw/sci/*.fits')
		d_found = pd.DataFrame(columns = d.columns)
		coors = []
		for fitsfile in fitslist:
			head = fits.getheader(fitsfile)
			coor = SkyCoord(head['RA'], head['DEC'], unit=(u.hourangle, u.deg))
			coors.append(coor)
			within_radius = coor.separation(d_coor) < searchradius
			d_f = d[within_radius]
			for item in d_f.iloc:
				if len(d_found)==0 or item['No'] in d_found['No']:
					d_found = d_found.append(item)
		print('\n'+koafolder)
		print(d_found)
		if len(d_found)==1:
			index = d_found.index[0]
			kj = d_found.at[index, 'KOAjobID']
			if np.isnan(kj):
				d.at[index, 'KOAjobID'] = koajobid
				print('updated:\n', pd.DataFrame(d.loc[index]).T)
			elif kj != koajobid:
				print('Error: koajobid in folder %d and in table %d do not match'%(koajobid, kj))
				continue
		elif d_found.empty:
			for coord in coors: print(coord.to_string())
			dkj = d[d['KOAjobID']==koajobid]
			if dkj.empty:
				if koajobid==123929: index = 116
				elif koajobid==125016: index = 151
				d.at[index, 'KOAjobID'] = koajobid
				print('updated:\n', pd.DataFrame(d.loc[index]).T)
			else: print(dkj)
		else:
			print('Error: found more than one target.')
			continue
	#d.to_csv(path+'data/elqs_full_sortM1450_addmore.csv')

def plot_elqs(koajobidlist):
	koajobidlist = np.atleast_1d(koajobidlist)
	fig, ax = plt.subplots(*subplot_shape(len(koajobidlist)), figsize=(12, 8))
	ax = np.atleast_1d(ax).flatten()
	for iplot, koajobid in enumerate(koajobidlist):
		item = d[d['KOAjobID']==koajobid].iloc[0]
		titles = 'ind: %d KOAjobID: %d use: %s'%(item['No'] - 1, koajobid, item['use'])
		print(titles)
		ax[iplot].set_title(titles)
		try: data = trim_koaspec(koajobid, plot_lya_forest=False)
		except Exception as e:
			print(type(e).__name__, ':', e)
			continue
		lam = data[0]
		flux = data[1]
		ax[iplot].plot(lam, flux, lw=0.5)
		ax[iplot].axvline(lya_wave, c='k')
		ax[iplot].axvline(lyb_wave, c='k')
		ax[iplot].axvline(1450, c='r')
		# set ylim
		flux_min = max(-15, ax[iplot].get_ylim()[0])
		flux_max = min(2000, ax[iplot].get_yli()[1])
		ax[iplot].set_ylim([flux_min, flux_max])
	plt.tight_layout()
	plt.show()

def check_spec():
	toplot = d[-np.isnan(d['KOAjobID']) & d['extracted']==True]
	plot_elqs(toplot['KOAjobID'])
		
if __name__ == '__main__': check_spec()

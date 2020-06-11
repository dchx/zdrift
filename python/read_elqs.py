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

def plot_elqs(koajobidlist, keck_catalog_list=keck_catalog, titles=None):
	koajobidlist = np.atleast_1d(koajobidlist)
	keck_catalog_list = np.atleast_1d(keck_catalog_list)
	fig, ax = plt.subplots(*subplot_shape(len(koajobidlist)), figsize=(12, 8))
	ax = np.atleast_1d(ax).flatten()
	for iplot, koajobid in enumerate(koajobidlist):
		d = get_matched(keck_catalog_list[iplot])[0]
		item = d[d['KOAjobID']==koajobid].iloc[0]
		if titles==None: title = 'ind: %d KOAjobID: %d use: %s'%(item.name, koajobid, item['use'])
		else: title = titles[iplot]
		print(title)
		ax[iplot].set_title(title)
		try: data = trim_koaspec(koajobid, plot_lya_forest=False, keck_catalog=keck_catalog_list[iplot])
		except Exception as e:
			print(type(e).__name__+':', e)
			continue
		lam = data[0]
		flux = data[1]
		ax[iplot].plot(lam, flux, lw=0.5)
		ax[iplot].axvline(lya_wave, c='k')
		ax[iplot].axvline(lyb_wave, c='k')
		ax[iplot].axvline(1450, c='r')
		#ax[iplot].set_xbound([lyb_wave, lya_wave])
		# set ylim
		flux_min = max(-15, ax[iplot].get_ylim()[0])
		flux_max = min(2000, ax[iplot].get_ylim()[1])
		ax[iplot].set_ylim([flux_min, flux_max])
	plt.tight_layout()
	plt.show()

def find_substitute():
	'''
	look for substitutes for top 11 elqs QSOs
	'''
	top11 = pd.read_csv(path + 'data/elqs_full_sortM1450_addmore.csv')[:11]
	for item in top11.iloc:
		if str(item['use'])!='True':
			print('\n----------')
			print(pd.DataFrame(item).T.to_string())
			criteria_base = (item.No != d.No) & (d.KOAjobID != 0) & (-np.isnan(d.KOAjobID))
			criteria = criteria_base & (item.z <= d.z) & (d.z <= item.z + 0.1)
			if item.No == 5: criteria = criteria_base & (d.z >= item.z - 0.2)
			substitutes = d[criteria]
			print('----------')
			print(substitutes.to_string())

def plot_substitute():
	top11 = pd.read_csv(path + 'data/elqs_full_sortM1450_addmore.csv')[:11]
	dfs = {'.': get_matched('.')[0], 'elqs': get_matched('elqs')[0]}
	dic = {2: ['.', 10408], 4: ['elqs', 119681], 5: ['elqs', 43254], 7: ['elqs', 54447], 8: ['elqs', 119681], 9: ['elqs', 2315], 10: ['elqs', 119681], 11: ['.', 12850]} # No: [keck_catalog, koajobid]
	kjlist = [] # koajobid list
	kclist = [] # keck_catalog list
	titles = []
	for item in top11.iloc:
		num = item.No
		if str(item['use'])!='True' and str(item['use'])!='check':
			kj = dic[num][1]
			kc = dic[num][0]
			subitem = dfs[kc][dfs[kc]['KOAjobID']==kj]
			kjlist.append(kj)
			kclist.append(kc)
			titles.append('No: %d z=%.3f z_sub=%.3f'%(num, item.z, subitem.z))
		else:
			kjlist.append(item.KOAjobID)
			kclist.append('elqs')
			titles.append('No: %d z=%.3f'%(num, item.z))
	plot_elqs(kjlist, kclist, titles)

def check_elqs_spec():
	toplot = d[-np.isnan(d['KOAjobID']) & d['extracted']==True]
	plot_elqs(toplot['KOAjobID'])

def check_original_spec():
	ind_no_use = [0, 22, 44, 51, 55, 56, 62, 66, 73, 81, 85, 87, 103, 109, 111, 137, 148, 157, 166, 183, 184, 186, 196, 199, 211, 213, 223, 246, 283, 284, 289, 306, 340, 350, 365, 372]
	ind_check = [53, 60, 67, 69, 84, 92, 93, 101, 104, 125, 127, 145, 150, 156, 161, 164, 168, 172, 185, 202, 204, 216, 282, 286]
	ind_use = [17, 20, 47, 142, 154, 160, 163, 200, 305]
	toplot = d[(d['KOAjobID']!=0) & (d['extracted']==True)]
	plot_elqs(toplot['KOAjobID'])
		
if __name__ == '__main__': plot_substitute()

from utils import *
import spec_utils as su
import sse
import read_elqs as re
import read_ps as rp
import liske_sigma as ls
import continuum_fit as cf
import flux_unit_change as fuc
import cosmology as cs
import obs_plan as op

class genspec_args:
	ispec = None
class keck_args:
	fitcont_dist = 100
	fitcont_deg = 10 # 03/19/2022 change from None to 10
	fitcont_mode = 'poly'
	vfaddline = True
	#if item['KOAjobID']==12850: vfaddline = False
	CSL_cut = 1.5
	smoothwidth = 30
	min_lw = 5. # km/s, 4/8/2022 add argument

def vds_filename(varname, nqso, selection='vds', addtxt=''):
	'''
	varname - 'aperture' or 'period'
	nqso - int or 'plan'
	selection - 'vds' or 'm14'
	addtxt - additional texts to add
	'''
	vds_tosave = path + 'paras/vds_%s_top%s%s_N%s.pkl'%(varname, nqso, selection, addtxt)
	return vds_tosave

def multi_epoch_templates(dts, linelistmode, genspec_args, keck_args, obs_setup, fixres='res'):
	'''
	linelistmode - 'genspec' or 'keck'
	'''
	fluxes = []
	for dt in dts:
		lam, flux = sse.get_template(linelistmode, fixres=fixres, res_fixres=obs_setup.resolution, res_ele=obs_setup.res_ele, shiftmode='dtlw', dx=dt, genspec_args=genspec_args, keck_args=keck_args, verbose=False, cosmo=cosmology.Planck15)
		fluxes.append(flux)
	fluxes = np.array(fluxes)
	return lam, fluxes

def add_noise_const(flux_temps, nphot):
	fluxes = []; errs = []
	for flux_temp in flux_temps:
		flux, err = su.add_shot_noise(flux_temp, nphot, return_error=True)
		fluxes.append(flux); errs.append(err)
	fluxes = np.vstack(fluxes); errs = np.vstack(errs)
	return fluxes, errs

def add_noise(item, lam, flux_temps, obs_setup, exptime_perobs, nphot=None, zqso=None, verbose=True, substituting=True, return_nphot=False, fitcont_deg=None, return_fluxtrue=False):
	'''
	item - use z, SDSS, KOAjobID(, z_origin, M1450_origin if substituting)
	lam - observed frame
	zqso - the redshift where lam (of generated spec) is
	return_nphot - whether to return total nphot (scalar)
	fitcont_deg - degree used to fit continuum, if None automatically setted degree
	'''
	if zqso is None:
		if substituting: zqso = item['z_origin']
		else: zqso = item['z']
	# add noise, in observed frame
	fluxpar_name = item['SDSS'] if pd.isna(item['KOAjobID']) else item['KOAjobID']
	fitcontdegtxt = '' if fitcont_deg is None else '_deg%d'%fitcont_deg
	fluxparfile = cf.fluxpar_filename(fluxpar_name, fitcontdegtxt) # use KOAjobID
	fluxpars = pkload(fluxparfile, verbose=verbose)
	fluxes = []; errs = []
	# normalize exptime_perobs to array
	if np.isscalar(exptime_perobs): exptime_perobs = [exptime_perobs] * len(flux_temps)
	# true flux
	if nphot is None:
		nphot_isdefined = False
		#flux_true = 250. # constant continuum, for lower continuum of No.1, (1e-17 erg/(cm2 s AA))
		flux_true = cf.fluxpars2flux(fluxpars, su.rest_frame(lam, zqso)) # non-constant continuum
		if substituting: flux_true *= re.substitute_flux_factor(item) # from substituted flux to original spec flux, use SDSS, z, z_origin, M1450_origin
	else: nphot_isdefined = True
	# compute nphot, add noise
	nphot_total = 0.
	nphots = []
	for iepoch, flux_temp in enumerate(flux_temps):
		if not nphot_isdefined:
			#print('cont flux min mean max: %.4f %.4f %.4f'%(np.min(flux_true), np.mean(flux_true), np.max(flux_true)))
			nphot = fuc.flux2nphot(lam, flux_true, obs_setup.aperture, exptime_perobs[iepoch], obs_setup.efficiency) # (npixel,)
		nphot_total += np.sum(nphot)
		nphots.append(nphot)
		flux, err = su.add_shot_noise(flux_temp, nphot, return_error=True)
		fluxes.append(flux); errs.append(err)
	#print('Total nphot: %s'%nphot_total)
	fluxes = np.vstack(fluxes); errs = np.vstack(errs)
	toreturn = [fluxes, errs]
	if return_nphot:
		nphots = np.array(nphots) # (nepoch, npixel)
		toreturn.append(nphots)
	if return_fluxtrue:
		toreturn.append(flux_true)
	#return [fluxes, errs]
	#return [fluxes, errs, nphot_total]
	#return [fluxes, errs, flux_true]
	#return [fluxes, errs, nphot_total, flux_true]
	return toreturn

def simulation_setup(item, linelistmode, obs_setup, nepoch=121, exptime_perobs=1.584e5, O=None, cosmomc_tosave=None, ispec=None, nbins=1):
	'''
	10 hours per night, 5 nights per week, one spectrum per two months
	=> 440 hours of one spectrum per two months / 10 QSOs
	=> 44 hours per QSO per two months
	-----------
	obs_setup:
		period - in years, total observation periods
		res_ele - Npixel per resolution element
		aperture - in m
	exptime_perobs - scalar or (nepoch,), in s, default 1.584e5 = 440. (h) * 3600. / 10 (qsos) (50 h per week, or 10 hr/night, 10 qsos)
	O - defined by Liske eq 27. Total O for each QSO. If not None, then use O instead of (aperture, efficiency, exptime_perobs)
	Returns
	  zdvsigma_binned - (3, nbins) z, dv, sigma_dv for each redshift bin
	  binwidth - bin width in redshift
	  zs - (npix,) all redshifts for each lam
	  dvdti - (npix,) all dvdts for each pixel
	  sig2_dvdti - (npix,) sigma^2 for each pixel
	  fig - plot of dv vs z for bins
	'''
	# parameters
	genspec_args.zqso = item['z_origin']
	genspec_args.ispec = ispec
	keck_args.item = item
	#if item['KOAjobID']==12850: keck_args.vfaddline = False

	# time points for each epoch 
	dts = np.linspace(0., obs_setup.period, nepoch) # (nepoch,) time points for each epoch, in years
	if not np.isscalar(exptime_perobs): # adjust dts and exptimes
		time_allocated_ind = np.where(exptime_perobs) # which epochs have time allocated
		if len(time_allocated_ind)==0: # no allocated exptime for this source
			raise Exception('no allocated exptime for this source')
		else: # only keep epochs with allocated exptime
			dts = dts[time_allocated_ind]
			exptime_perobs = exptime_perobs[time_allocated_ind]

	# get spec templates for multiple epochs, observed frame
	lam, flux_temps = multi_epoch_templates(dts, linelistmode, genspec_args, keck_args, obs_setup)

	# add noise, in observed frame
	fluxes, errs, nphot = add_noise(item, lam, flux_temps, obs_setup, exptime_perobs, return_nphot=True, fitcont_deg=keck_args.fitcont_deg) # item use z, SDSS, KOAjobID, z_origin, M1450_origin

	# sigma
	verbose = False
	## two epochs
	if verbose: print('two epochs')
	dvi, sig2_dvi = ls.liske_dvi(lam, fluxes[0], fluxes[-1], errs[0], errs[-1])
	if verbose: print(np.nanmedian(errs))
	if verbose: print(dvi, np.nanmedian(np.abs(dvi)))
	sigma = ls.overall_sigma(sig2_dvi)
	if verbose: print('sigma: %.3e cm/s'%sigma)
	## multiple epochs
	if verbose: print('multiple epochs')
	dvdti, sig2_dvdti = ls.liske_dvdti(lam, fluxes, errs, dts) # in cm/s/yr
	#dvdt_est = np.nanmedian(dvdti)
	dvdt_est = ls.wa_1d(dvdti, sig2_dvdti)
	#dvdt_est = np.nanmean(dvdti)
	if verbose: print('dvdt_est: %.3f'%dvdt_est)
	sigma_dvdt = ls.overall_sigma(sig2_dvdti)
	sigma = sigma_dvdt * dts[-1]
	if verbose: print('sigma: %.3e cm/s'%sigma)

	# dvdt binned by z
	zs = su.lam2z(lam)
	binwidth = 0.2
	zdvsigma_binned, binedges = ls.sigma_binned(zs, dvdti, sig2_dvdti, nbins=nbins, return_binedges=True)
	binwidth = np.diff(binedges)
	#zdvsigma_binned = ls.sigma_binned(zs, dvdti, sig2_dvdti, binwidth=binwidth)
	plot = 1
	if plot:
		fig = plt.figure()
		plt.plot(zs, dvdti,'.')
		plt.axhline(0., color='k')
		# plot overall dvdt and sigma_dvdt
		plt.axhline(dvdt_est, color='b')
		plt.axhspan(dvdt_est-sigma_dvdt, dvdt_est+sigma_dvdt, color='r', alpha=0.5)
		# plot binned dvdt and sigma_dvdt
		if nbins!=0: plt.errorbar(*zdvsigma_binned[:2], xerr=binwidth/2, yerr=zdvsigma_binned[2], fmt='or', capsize=2)
		plt.ylim(-5., 5.) 
		plt.xlabel('z') 
		plt.ylabel('dvdt (cm/s/yr)') 
		plt.title('%d years, overall sigma_dvdt: %.2e cm/s/yr'%(obs_setup.period, sigma_dvdt))
		#plt.show() 

	if cosmomc_tosave!=None: ls.cosmomc_product_multiepoch(zs, dvdti, np.sqrt(sig2_dvdti), cosmomc_tosave, fmt='dv')
	return zdvsigma_binned, binwidth, zs, dvdti, sig2_dvdti, fig, nphot

# --- observation setup
class obs_setup:
	period = 20. # year
	resolution = 5e4 # 50k
	res_ele = 3. # Npixel per resolution element
	aperture = 15. # meter
	efficiency = 0.25 # 0.18 telescope efficiency

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	linelistmode = 'genspec' # 'keck' or 'genspec'
	nbins_perqso = 1 # default 1, if ==0 use each pixel
	selection = 'm14' # 'vds' (vdot/sigma) or 'm14' (m1450), target selection method
	ispec = None
	debug = True

	# time allocation strategies
	individual_exptime = False # whether assgin individual exptime for each qso from obs_plan
	if individual_exptime:
		#addfake = 0 # None or number (how many mag brighter) # 'top1' (add new target at top1's brightness) or 'brighter' (add new target 0.5mag brighter than top1)
		test_mimicplan = False # whether testing time allocation mimicing obs_plan
	test_concentrate_time = False # whether test making time more concentrated for each QSO than averaged over the year
	if test_concentrate_time: concentrate_mimicplan = True
                
	# exptime settings
	dt_per_obs = 2. / 12. # two months in years
	#exptime_perepoch = 300. * 3600. # in seconds, (35 h per week, or 7 hr/night)
	exptime_perepoch = 440. * 3600. # in seconds, (50 h per week, or 10 hr/night)
	if individual_exptime: exptime_factor = 0.8

	# --- dvdt/sigma target selection
	# for elqs without ps-elqs
	'''
	selection = 'dvdtsigma' # 'm1450' or 'dvdtsigma'
	top10 = re.top10_df(True, selection) 
	# old substitutions, need to edit read_elqs.top10_df.dic
	substuting = re.top10m14.sort_values('z')
	origin = re.top10vds_nosub.sort_values('z')
	substuting['z_origin'] = origin['z'].to_numpy()
	substuting['M1450_origin'] = origin['M1450'].to_numpy()
	top10 = substuting
	'''
	changeTop1 = 0 # how much brighter to change top1
	addfake = None # None or number (how many mag brighter) # 'top1' (add new target at top1's brightness) or 'brighter' (add new target 0.5mag brighter than top1)

	contdegtxt = '' if keck_args.fitcont_deg is None else '_contdeg%d'%keck_args.fitcont_deg
	#for changeTop1 in [0.5, 1, 2]:
	for addfake in [None]: # [None, 0, 0.5, 1, 2]
		for varname in ['aperture', 'period']:
			# re-assert default values
			obs_setup.period = 20.
			obs_setup.aperture = 15.

			# for elqs + ps-elqs
			if selection=='vds': top10 = copy.deepcopy(rp.top10vds_N) # for elqs + ps-elqs
			elif selection=='m14': top10 = copy.deepcopy(rp.top10m14_N) # for elqs + ps-elqs
			nqso = 5 # should be kept in the loop, because it will be change below
			#varname = 'period' # 'aperture' or 'period' or 'z'; change what and compute vds
                
			# exptimes from obs_plan
			if individual_exptime:
				exptimes = pkload(op.exptime_tosave(nqso=nqso, addfake=addfake)[0])[:, :-1] # (6, nqso) TimeDelta object, for 6 obs per year, excluding empty time (last column)
				exptimes = exptimes.to(u.s).value * exptime_factor # (6, nqso) float array in s
				if debug and (addfake is not None):
					exptimes_noadd = pkload(op.exptime_tosave(nqso=nqso, addfake=None)[0])[:, :-1] # (6, nqso) TimeDelta object, for 6 obs per year, excluding empty time (last column)
					exptimes_noadd = exptimes_noadd.to(u.s).value * exptime_factor # (6, nqso) float array in s
				if test_mimicplan:
					nqso = 3
					exptimes = exptimes[:, :nqso]
			#    test concentrated time
			if test_concentrate_time:
				nqso = 3
				#exptimes = np.tile(np.diag(np.repeat(exptime_perepoch, 3)), [2, 1]) # (6, 3) i.e. (nepoch_peryear, nqso) for nqso==3, array float
				exptimes = np.zeros([6, 3])
				if concentrate_mimicplan:
					exptimes[0, 0] = exptime_perepoch * 2.7 # top No.1
					exptimes[2, 1] = exptime_perepoch * 1.3 # top No.2
					exptimes[4, 2] = exptime_perepoch * 2.0 # top No.3
				else: exptimes[[0, 2, 4], [0, 1, 2]] = exptime_perepoch * 2. # (6, 3) i.e. (nepoch_peryear, nqso) for nqso==3, array float
			if individual_exptime and debug:
				# delete some rows
				#exptimes = exptimes[:, [0, 1, 2, 5]] # delete original no.3, no.4
				#exptimes = exptimes[:, [0, 2, 3, 4, 5]] # delete original no.1
				# exchange some columns
				#exptimes[0, 0] -= 10. * 3600.
				#exptimes[0, 1] += 10. * 3600.
				# delete some time slots
				#exptimes[0, 1] += exptimes[0, 0]
				#exptimes[0, 0] -= exptimes[0, 0]
				#exptimes[0, 0] = 0.0 # -1 Jan-Feb 0 hour
				#exptimes[1, 0] = 0.0 # -1 Mar-Apr 0 hour
				#exptimes[:2, 1:] = exptimes_noadd[:2]

				#addedtime = exptimes[:, 0].copy()
				#exptimes[:, 0] = exptimes[:, 1].copy()
				#exptimes[:, 1] = addedtime
				#exptimes[0, 1] = 0.
				# print exptimes
				print('exptimes:', exptimes.shape)
				for i in range(len(exptimes)):
					print(' '.join(['%8.2f'%t for t in exptimes[i]/3600.])) # in hour
			
			# --- select only top n
			# for elqs without ps-elqs
			'''
			topn_nosub = re.top10_df(False, selection).iloc[:nqso] # top nqso dvdt/sigma
			topn = top10[[(z in topn_nosub['z'].to_numpy()) for z in top10['z_origin']]]
			top10 = topn
			'''
			# for elqs + ps-elqs
			top10 = top10[:nqso]
			if individual_exptime:
				top10 = op.topn_addnew(top10, addfake, mode='mag', changeTop1=changeTop1)
				if debug:
					'''
					# added use no.2 instead of no.1
					add_m1450 = top10.at[-1, 'M1450_origin']
					add_z = top10.at[-1, 'z_origin']
					top10.loc[-1] = top10.loc[2]
					top10.at[-1, 'M1450_origin'] = add_m1450
					top10.at[-1, 'z_origin'] = add_z
					# delete original no.3, no.4
					top10 = top10.iloc[[0, 1, 2, 5]]
					top10 = top10.iloc[[0, 2, 3, 4, 5]] # delete original no.1
					'''
					print(top10)
				if test_mimicplan: nqso = '%smimicplan'%nqso
				else:
					mag_brighter = addfake if np.isreal(addfake) else 0 if addfake=='top1' else 0.5 if addfake=='brighter' else None # make it this mag brighter
					addfaketxt = '' if addfake is None else '_add%sMagBrighter'%mag_brighter
					changetxt = '' if (changeTop1 is None or changeTop1==0) else '_top1%sMagBrighter'%changeTop1
					nqso = 'plan' + addfaketxt + changetxt
					if debug: print(nqso) # description
			if test_concentrate_time:
				if concentrate_mimicplan: nqso = '%sconcentrateMimic'%nqso
				else: nqso = '%stestConcentrateTime'%nqso
                
			figall, axall = plt.subplots() # dvdt vs z / vdotsigma vs aperture / vdotsigma vs period
			ill = 0 if linelistmode=='genspec' else 1 # which linelist
			#linelists = ['genspec', 'keck']
			labels = ['Simulated LL', 'Real LL']
			colors_ll = ['r', 'b']
			apertures = np.arange(10., 41.) # 10 to 40 m
			periods = (np.arange(20.) + 1.) # 1 to 20 years
			vdses = [] # list of vdot/sigma

			if varname=='aperture': variable = apertures # plot vdot/sigma vs aperture
			elif varname=='period': variable = periods # plot vdot/sigma vs period
			elif varname=='z': variable = [None] # plot vdot vs period
			var = variable[-1]
			#for ispec in range(10):
			#for ispec in [None]:
			for var in variable:
				if varname=='aperture': obs_setup.aperture = var # 10 to 40 m
				elif varname=='period': obs_setup.period = var # 1 to 20 years
				elif varname=='z':
					obs_setup.aperture = 1000.
					obs_setup.period = 20.
				print('%d years, %d m'%(obs_setup.period, obs_setup.aperture))
				# template generation parameters
				#nepoch = 121 # two months per observation in 20 years
				nepoch = int(round(obs_setup.period / dt_per_obs)) + 1
				# add noise parameters
				if individual_exptime or test_concentrate_time:
					exptimes = np.tile(exptimes, [math.ceil(obs_setup.period) + 1, 1])[:nepoch] # (6, nqso) -> (nepoch, nqso)
					t_tot = np.sum(exptimes) # scalar
				else:
					exptime_perobs = exptime_perepoch / nqso

					# adjusting setup by big O from Liske
					t_tot = exptime_perepoch * nepoch # total integration time for all
				O = fuc.bigO_liske(obs_setup.aperture, t_tot, obs_setup.efficiency)
				print('O: %.2f'%O)
				O_aim = 3.4
				#obs_setup.efficiency = obs_setup.efficiency / O * O_aim; print('Changed to O = %s'%O_aim)

				zs_all = []; dvdti_all = []; sig2_dvdti_all = [] # all estimates for each pixel
				zdvsigma_binned_all = []; total_nphot = 0.
				for qso_num in range(1, len(top10)+1):
					item = top10.iloc[qso_num - 1]
					if debug:
						pass
						#print('QSO No.%d'%qso_num)
						#print(pd.DataFrame(item).T)
					cosmomc_tosave = None
					if individual_exptime or test_concentrate_time:
						exptime_perobs = exptimes[:, qso_num - 1] # (nepoch,)
						if debug:
							pass
							#print('exptime_perobs')
							#print(exptime_perobs[:6]/3600.)
						if np.sum(exptime_perobs) <= 0: continue # no time allocated for this source
					zdvsigma_binned, binwidth, zs, dvdti, sig2_dvdti, fig, nphot = simulation_setup(item, linelistmode, obs_setup, nepoch, exptime_perobs, cosmomc_tosave=cosmomc_tosave, ispec=ispec, nbins=nbins_perqso)
					if debug:
						pass
						#print('sigma for QSO %s:'%qso_num, zdvsigma_binned[2])
						#sig_thisqso = ls.overall_sigma(sig2_dvdti)
						#print(sig_thisqso)
					# append
					zs_all.append(zs) # list of redshifts of each pixel for each qso
					dvdti_all.append(dvdti) # list of dvdt of each pixel for each qso
					sig2_dvdti_all.append(sig2_dvdti) # list of sigma^2 of each pixel for each qso
					if nbins_perqso!=0: zdvsigma_binned_all.append(zdvsigma_binned) # list of z, dv, sigma for (3, nbins) each qso each bin
					total_nphot += np.sum(nphot)
					#fig.savefig(path + 'plots/dvdt_%dyr_%d_nobin%s.pdf'%(obs_setup.period, qso_num, contdegtxt))
					plt.close(fig)

					# plot binned dvdt for all qsos
					if varname=='z' and nbins_perqso!=0: axall.errorbar(*zdvsigma_binned[:2], xerr=binwidth/2, yerr=zdvsigma_binned[2], fmt='or', capsize=2)
				if debug: print('Total nphot: %s'%total_nphot)
				# ---  plot binned dvdt for all qsos combined
				# stack different qsos
				zs_all = np.hstack(zs_all) # (npixel * nqso,)
				dvdti_all = np.hstack(dvdti_all) # (npixel * nqso,)
				sig2_dvdti_all = np.hstack(sig2_dvdti_all) # (npixel * nqso,)
				if nbins_perqso!=0: zdvsigma_binned_all = np.hstack(zdvsigma_binned_all) # (3, nbins_perqso * nqso)
				else: zdvsigma_binned_all = np.vstack([zs_all, dvdti_all, np.sqrt(sig2_dvdti_all)]) # (3, npixel * nqso)
				# save cosmomc product
				if varname=='z':
					cosmomc_tosave = path + 'paras/cosmomc_zdrift_%.1fyr_%dm_top%s%s_N_%dbinperqso%s.txt'%(obs_setup.period, obs_setup.aperture, nqso, selection, nbins_perqso, contdegtxt)
					#cosmomc_tosave = None
					if cosmomc_tosave!=None: ls.cosmomc_product_multiepoch(*zdvsigma_binned_all, cosmomc_tosave, fmt='dv')

				# --- average of each qso/bin, assuming nbin==1
				sigma_perqso_mean = np.mean(zdvsigma_binned_all[2])
				vdot2sig_perqso_mean = np.mean(abs(cs.dvdt(np.array(zdvsigma_binned_all[0])))/zdvsigma_binned_all[2])
				#print('perqso mean err: %.2f'%sigma_perqso_mean, 'cm/s/yr; dvdt/sigma per qso: %.2f'%vdot2sig_perqso_mean)
				# --- nbins over redshift
				nbins = 5
				(z_binned, dvdt_binned, sigmadvdt_binned), binedges = ls.sigma_binned(zs_all, dvdti_all, sig2_dvdti_all, nbins=nbins, return_binedges=True)
				binwidth = np.diff(binedges)
				if varname=='z': axall.errorbar(z_binned, dvdt_binned, xerr=binwidth/2, yerr=sigmadvdt_binned, fmt='o', color=colors_ll[ill], label=labels[ill], capsize=2)
				#print('%d bin err:'%nbins, ('%.2f '*nbins)%tuple(sigmadvdt_binned), 'cm/s/yr; dvdt/sigma:', ('%.2f '*nbins)%tuple(abs(cs.dvdt(np.array(z_binned)))/sigmadvdt_binned))
				if varname=='z':
					cosmomc_tosave = path + 'paras/cosmomc_zdrift_%.1fyr_%dm_top%s%s_N_%dbin%s.txt'%(obs_setup.period, obs_setup.aperture, nqso, selection, nbins, contdegtxt)
					#cosmomc_tosave = None
					if cosmomc_tosave!=None: ls.cosmomc_product_multiepoch(z_binned, dvdt_binned, sigmadvdt_binned, cosmomc_tosave, fmt='dv')
				# --- 1 overall bin
				nbins = 1
				(z_binned, dvdt_binned, sigmadvdt_binned), binedges = ls.sigma_binned(zs_all, dvdti_all, sig2_dvdti_all, nbins=nbins, return_binedges=True)
				binwidth = np.diff(binedges)
				#if varname=='z': axall.errorbar(z_binned, dvdt_binned, xerr=binwidth/2, yerr=sigmadvdt_binned, fmt='o', color=colors_ll[ill], label=labels[ill], capsize=2) # dv vs z
				vds = abs(cs.dvdt(np.array(z_binned)))/sigmadvdt_binned
				vdses.append(np.squeeze(vds))
				print('1 bin err:', ('%.5f '*nbins)%tuple(sigmadvdt_binned), 'cm/s/yr; dvdt/sigma:', ('%.5f '*nbins)%tuple(vds))
				if varname=='z':
					cosmomc_tosave = path + 'paras/cosmomc_zdrift_%.1fyr_%dm_top%s%s_N_%dbin%s.txt'%(obs_setup.period, obs_setup.aperture, nqso, selection, nbins, contdegtxt)
					#cosmomc_tosave = None
					if cosmomc_tosave!=None: ls.cosmomc_product_multiepoch(z_binned, dvdt_binned, sigmadvdt_binned, cosmomc_tosave, fmt='dv')
			vdses = np.array(vdses)
			vds_product = np.vstack([variable, vdses]) # (2, len(variable))
			# --- save results for aperture or period
			save_vds = True
			if varname=='z': save_vds = False
			if save_vds:
				vds_tosave = vds_filename(varname, nqso, selection, contdegtxt)
				pkdump(vds_product, vds_tosave) # (2, len(variable)) variable (aperture or period), vdses
			# --- plot vds vs aperture or period
			if not varname=='z': axall.plot(*vds_product, 'ok') # vdot/sigma vs aperture or period
			# linear fit (only for apertrue without noise)
			if varname=='aperture':
				ppoly = np.polyfit(*vds_product, 1)
				#print('fitted coefficient: %s'%ppoly)
				var_fit = np.linspace(min(variable), max(variable), 100)
				vds_fit = np.polyval(ppoly, var_fit)
				axall.plot(var_fit, vds_fit, 'b')
			if varname=='z':
				#axall.legend()
				# plot cosmology
				zlim = axall.get_xlim()
				z_grid = np.linspace(*zlim, 1000)
				dvdt_grid = cs.dvdt(z_grid)
				axall.plot(z_grid, dvdt_grid, 'k')
				axall.set_xlim(zlim)
				# plot liske points
				liske_zs_blue = [2., 2.5, 3., 3.5]
				liske_zerr_blue = 0.25
				liske_dvdt_blue = [0.06, 0.09, -0.2, -0.4]
				liske_dvdterr_blue = [0.19, 0.32, 0.23, 0.26]
				axall.errorbar(liske_zs_blue, liske_dvdt_blue, xerr=liske_zerr_blue, yerr=liske_dvdterr_blue, fmt='o', color='#0085ba', capsize=2)
				liske_zs_yellow = [3.67, 4.25]
				liske_zerr_yellow = [[0.42, 0.14], [0.42, 0.25]]
				liske_dvdt_yellow = [-0.38, -0.7]
				liske_dvdterr_yellow = [0.15, 0.31]
				axall.errorbar(liske_zs_yellow, liske_dvdt_yellow, xerr=liske_zerr_yellow, yerr=liske_dvdterr_yellow, fmt='s', color='#ecb641', capsize=2)
				# wrapup plot
				axall.axhline(0., color='k')
				#axall.set_ylim([-1.5, 0.5])
			# set ytick frequency
			axall.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
			axall.yaxis.set_major_locator(plt.MultipleLocator(1))
			# wrapup plot
			if varname=='aperture':
				axall.set_xlabel('Aperture (m)') 
				axall.set_title('polynomials: %s'%ppoly)
			elif varname=='period': axall.set_xlabel('Time (year)') 
			elif varname=='z': axall.set_xlabel('$z$') 
			if varname=='z':
				axall.set_ylabel('$\mathrm{d}v/\mathrm{d}t$ (cm s$^{-1}$ yr$^{-1}$)') # dvdt
				tosave = path + 'plots/dvdt_%dyr_%dm_top%s%s_N%s.pdf'%(obs_setup.period, obs_setup.aperture, nqso, selection, contdegtxt)
			else:
				axall.set_ylabel('$|\dot{v}|/\sigma_\dot{v}$') # vdot/sigma
				tosave = path + 'plots/vds_%s_top%s%s_N%s.pdf'%(varname, nqso, selection, contdegtxt)
			figall.savefig(tosave); print('Saved: %s'%tosave)
			#plt.show()

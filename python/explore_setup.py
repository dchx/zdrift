from utils import *
import lmfit as lf
import spec_utils as su
import continuum_fit as cf
import read_elqs as re
import read_ps as rp
import flux_unit_change as fuc
import liske_sigma as ls
import generate_spec as gs
import cosmology as cs
import simulation_setup as ss
import realll_vs_simll as rvs

def sigma_as_zqso(zqso, const, power, addone=True):
	if addone: return const * (1. + zqso)**power
	else: return const * (zqso)**power

def compute_const2(paras, addone=True):
	if addone: const2 = paras['const1'] * (1. + paras['sep'])**(paras['power1'] - paras['power2'])
	else: const2 = paras['const1'] * (paras['sep'])**(paras['power1'] - paras['power2'])
	return const2

def sigma_as_zqso_sep(zqso, paras, sep=True, addone=True):
	'''
	sigma = [const1 * (1 + zqso_l)**power1, const2 * (1 + zqso_r)**power2]
	log(sigma) = power * log(const * (1 + zqso))
	'''
	zqso_l = zqso[zqso <= paras['sep']]
	zqso_r = zqso[zqso > paras['sep']]
	const2 = compute_const2(paras, addone=addone)
	sigma_l = sigma_as_zqso(zqso_l, paras['const1'], paras['power1'], addone=addone)
	sigma_r = sigma_as_zqso(zqso_r, const2, paras['power2'], addone=addone)
	sigma = np.zeros_like(zqso)
	sigma[zqso <= paras['sep']] = sigma_l
	sigma[zqso > paras['sep']] = sigma_r
	return sigma

def sigma_as_R(R, paras, sep):
	'''
	sigma = [const1 * R**power1, const2 * R**power2]
	if sep: same as sigma_as_zqso_sep() -- sigma = [const1 * (1 + R_l)**power1, const2 * (1 + R_r)**power2]
	else: sigma = const1 * R**power1 + const2
	'''
	if sep: return sigma_as_zqso_sep(R, paras, addone=False)
	#return paras['const1'] * R**paras['power1'] + paras['const2'] * R**paras['power2']
	return paras['const1'] * R**paras['power1'] + paras['const2']

def sigma_as_ftrue_zqso(inputs, paras, sep=True):
	'''
	sigma = [const1 * flux_trues^-0.5 * (1 + zqso)^power1,
	         const2 * flux_trues^-0.5 * (1 + zqso)^power2]
	'''
	flux_trues, zqsos = inputs
	return flux_trues**(-0.5) * sigma_as_zqso_sep(zqsos, paras, sep=sep)

def nphot_as_zqso(zqso, paras, sep=False):
	'''
	nphot = const1 * (1 + zqso)^2
	'''
	return paras['const1'] * (1. + zqso)**2.

def residual(paras, x, y, y_err, y_func, sep=True, log=True):
	if log:
		obs = np.log(y)
		pred = np.log(y_func(x, paras, sep))
		err = np.abs(y_err / y)
		res = (obs - pred) / err
	else:
		res = (y - y_func(x, paras, sep)) / np.abs(y_err)
	return res

def fit_sigma_zqso(zqso, sigma, sigma_err):
	paras = lf.Parameters()
	paras.add('const1', value=1., min=0.)
	paras.add('power1', value=-2., max=0.)
	paras.add('power2', value=-1., max=0.)
	paras.add('sep', value=4., min=2., max=5., vary=True) # zqso seperation point
	result = lf.minimize(residual, paras, args=(zqso, sigma, sigma_err, sigma_as_zqso_sep))
	return result

def fit_sigma_ftrue_zqso(flux_trues, zqsos, sigma, sigma_err):
	paras = lf.Parameters()
	paras.add('const1', value=1., min=0.)
	paras.add('power1', value=-2., max=0.)
	paras.add('power2', value=-1., max=0.)
	paras.add('sep', value=4., min=2., max=5., vary=True) # zqso seperation point
	result = lf.minimize(residual, paras, args=((flux_trues, zqsos), sigma, sigma_err, sigma_as_ftrue_zqso)) # TODO: edit residual() allow of 2 x variables
	return result

def fit_sigma_R(R, sigma, sigma_err, sep):
	paras = lf.Parameters()
	if sep:
		paras.add('const1', value=1e4, min=0.)
		paras.add('power1', value=-1.5, max=0.)
		paras.add('power2', value=-1., max=0.)
		paras.add('sep', value=1e4, min=1e3, max=1e5, vary=True) # zqso seperation point
	else:
		paras.add('const1', value=57561., min=0.)
		paras.add('const2', value=0.08, min=0., max=1.)
		paras.add('power1', value=-1.5, max=0.)
		#paras.add('power2', value=-0.1, max=0.)
	result = lf.minimize(residual, paras, args=(R, sigma, sigma_err, sigma_as_R, sep))
	return result

def fit_nphot_zqso(zqso, nphot, nphot_err):
	paras = lf.Parameters()
	paras.add('const1', value=1e7, min=0.)
	result = lf.minimize(residual, paras, args=(zqso, nphot, nphot_err, nphot_as_zqso, False, False))
	return result

def explore_setup(ivar=0, const_nphot=True, nphot=None, kid=None, fixres='res'):
	'''
	sigma vs zqso, sigma vs R
	ivar - 0 for R, 1 for zqso, modify here and the for loop below
	const_nphot - whether use const nphot or compute nphot from qso flux and exposures
	'''
	if const_nphot:
		if type(nphot)==type(None): nphot = 1e8
		sigma_as_R_sep = True # if True use two-piece power law, else use power law + constant
	else:
		if type(kid)==type(None):
			# get koajobids
			koajobids = rvs.kid_withspec()
			#kid = 104297.
			kid = koajobids[0]
		#item = re.top10m14.loc[kid] # use S5 0014+81 and its own continuum level
		item = df_all.loc[kid] # use 'SDSS' or 'KOAjobID'
		exptime_perobs = 26000. * 3600. # 50 h per week, 20 years, 2 epochs, 1 qso
		sigma_as_R_sep = False # if True use two-piece power law, else use power law + constant
	nepoch = 2
	zqso = 3.
	resolution = ss.obs_setup.resolution
	nphottxt = '_nphot%.2e'%nphot if const_nphot else '_koa%d'%kid
	if not const_nphot: nphottxt += '' if ss.keck_args.fitcont_deg is None else '_contdeg%d'%ss.keck_args.fitcont_deg # continuum fit degree
	nphottxt += '' if fixres=='res' else '_fixdlam' if fixres=='resele' else '_unknownFixres'

	# variables
	#Rs = np.r_[2e3, 5e3, 1e4, 2.5e4, 5e4, 1e5]
	Rs = 10**np.arange(3, 5.01, 0.1)
	Rs = np.array([x.round(1-int(np.log10(x))) for x in Rs])
	#zqsos = np.r_[2., 2.5, 3., 3.5, 4., 4.5, 5.]
	zqsos = np.arange(2., 5.01, 0.1)
	variables = [Rs, zqsos]
	varnames = ['Spectral resolution / 1000', '$z_\mathrm{QSO}$']
	varnames_dvdtosig = ['Spectral resolution / 1000', '$z$']
	varshort = ['R', 'zqso']
	varshort_dvdtosig = ['R', 'zmid']
	# derived parameters
	dts = np.linspace(0., ss.obs_setup.period, nepoch)

	# compute sigma
	sigmas = []; sigma_errs = []
	nphots = []; nphot_errs = []
	if ivar==1: fluxtrue_zqso_sigs = np.array([]).reshape(10, 0, 3) # (10, npix * n_zqso, 3)
	for var in variables[ivar]:
		if ivar==0: resolution = var
		elif ivar==1: zqso = var
		sigma_perspec = []
		nphot_perspec = []
		if ivar==1: fluxtrue_zqso_sigs_perzqso = []
		for ispec in range(10):
			# spectra template
			ss.genspec_args.zqso = zqso
			ss.genspec_args.ispec = ispec
			ss.obs_setup.resolution = resolution
			lam, flux_temps = ss.multi_epoch_templates(dts, 'genspec', ss.genspec_args, ss.keck_args, ss.obs_setup, fixres=fixres)
			# add noise
			if const_nphot: fluxes, errs = ss.add_noise_const(flux_temps, nphot)
			else:
				fluxes, errs, nphot_perpix, flux_true = ss.add_noise(item, lam, flux_temps, ss.obs_setup, exptime_perobs, zqso=zqso, 
				             verbose=False, substituting=False, fitcont_deg=ss.keck_args.fitcont_deg, return_nphot=True, return_fluxtrue=True)
				flux_true_mean = np.mean(flux_true) # flux_true: (npix,)
				nphot = np.mean(np.median(nphot_perpix, axis=-1)) # median over pixels
			nphot_perspec.append(nphot)
			# compute sigma
			dvdti, sig2_dvdti = ls.liske_dvdti(lam, fluxes, errs, dts) # in cm/s/yr; (npix,); have nan values
			dvdt_est = ls.wa_1d(dvdti, sig2_dvdti) # scalar
			sigma_dvdt = ls.overall_sigma(sig2_dvdti) # scalar
			sigma_perspec.append(sigma_dvdt)
			# fluxtrue_zqso_sig
			if ivar==1: 
				fluxtrue_zqso_sig_perspec = np.array([flux_true, [zqso]*flux_true.size, np.sqrt(sig2_dvdti)]).T # (npix, 3)
				fluxtrue_zqso_sigs_perzqso.append(fluxtrue_zqso_sig_perspec)
		sigmas.append(np.mean(sigma_perspec))
		sigma_errs.append(np.std(sigma_perspec, ddof=1))
		nphots.append(np.mean(nphot_perspec))
		nphot_errs.append(np.std(nphot_perspec, ddof=1) or 1e-20)
		# fluxtrue_zqso_sig
		if ivar==1: 
			fluxtrue_zqso_sigs_perzqso = np.array(fluxtrue_zqso_sigs_perzqso) # (10, npix, 3)
			fluxtrue_zqso_sigs = np.concatenate([fluxtrue_zqso_sigs, fluxtrue_zqso_sigs_perzqso], axis=1)
	sigmas = np.array(sigmas); sigma_errs = np.array(sigma_errs)
	# fluxtrue_zqso_sig
	if ivar==1: 
		'''
		fluxtrues = fluxtrue_zqso_sigs[:, :, 0] # (10, npix * n_zqsos) # debug
		same_fluxtrue = np.all(np.min(fluxtrues, axis=0) == np.max(fluxtrues, axis=0)) #  if flux_true same for all ispec # debug
		print('fluxtrue_zqso_sigs:', fluxtrue_zqso_sigs.shape) # result: (10, 790035, 3) # debug
		print('flux_true same for all ispec?', same_fluxtrue) # result: True # debug
		'''
		fluxtrue_zqso_sigs_mean = np.nanmean(fluxtrue_zqso_sigs[:, :, [-1]], axis=0) # (npix * n_zqsos, 1)
		fluxtrue_zqso_sigs_err = np.nanstd(fluxtrue_zqso_sigs[:, :, [-1]], axis=0, ddof=1) # (npix * n_zqsos, 1)
		fluxtrue_zqso_sigs = np.concatenate([fluxtrue_zqso_sigs[0, :, :-1], fluxtrue_zqso_sigs_mean, fluxtrue_zqso_sigs_err], axis=1) # (npix * n_zqsos, 4); flux_true, zqso, sigs, sigs_err
		fluxtrue_zqso_sigs = fluxtrue_zqso_sigs[~np.any(np.isnan(fluxtrue_zqso_sigs), axis=1)] # remove rows with nan values
		#print('fluxtrue_zqso_sigs:', fluxtrue_zqso_sigs.shape) # result: (747501, 4) # debug

	# fit sigmadvdt vs zqso
	if ivar == 0: result = fit_sigma_R(Rs, sigmas, sigma_errs, sep=sigma_as_R_sep)
	if ivar == 1:
		result = fit_sigma_zqso(zqsos, sigmas, sigma_errs)
		result_nphot = fit_nphot_zqso(zqsos, nphots, nphot_errs)
		# fluxtrue_zqso_sig
		result_ftrue = fit_sigma_ftrue_zqso(*fluxtrue_zqso_sigs.T)
	paras = result.params
	print(paras)
	if ivar == 1:
		# print nphot ~ zqso result
		paras_nphot = result_nphot.params
		print('nphot vs zqso:', paras_nphot)
		print('nphot = %.4e +/- %.2e * (1 + zqso)^2'%(paras_nphot['const1'].value, paras_nphot['const1'].stderr))
		print('S/N = %.4e +/- %.2e * (1 + zqso)'%(np.sqrt(paras_nphot['const1'].value), paras_nphot['const1'].stderr/2./np.sqrt(paras_nphot['const1'].value)))

		# print sigma ~ zqso result
		# sigma = [const1 * (1 + zqso_l)**power1, const2 * (1 + zqso_r)**power2]
		#       = [const1 * (1 + sep)**power1 * ((1 + zqso_l)/(1 + sep))**power1, ...]
		#       = [factor * ((1 + zqso_l)/(1 + sep))**power]
		#       = [factor * (F/F_mean)**(-0.5) * ((1 + zqso_l)/(1 + sep))**power]
		#       = [factor * (F/F_mean)**(-0.5) * (Fref/Fref)**(-0.5) * ((1 + zqso_l)/(1 + sep))**power]
		#       = [factor * (Fref/F_mean)**(-0.5) * (F/Fref)**(-0.5) * ((1 + zqso_l)/(1 + sep))**power]
		factor = paras['const1'].value * (1. + paras['sep'].value)**paras['power1'].value
		sepperr_over_sepp_2 = (paras['power1'].value / (1. + paras['sep'].value) * paras['sep'].stderr)**2.\
		                    + (np.log(1. + paras['sep'].value) * paras['power1'].stderr)**2. # (sigma((1+sep)^power1)/((1+sep)^power1))^2
		factor_err = factor * np.sqrt((paras['const1'].stderr / paras['const1'].value)**2. + sepperr_over_sepp_2)
		flux_ref = 100. # 10^-17 erg ...
		factor *= (flux_ref/flux_true_mean)**(-0.5)
		factor_err *= (flux_ref/flux_true_mean)**(-0.5)
		print(f'sigma_vdot = ({factor:.2e} +/- {factor_err:.2e}) * (F_mean/{flux_ref} * 10^-17 erg ...)^(-0.5) * ((1 + zqso)/(1 + zbreak))^power')

		# print sigma_i ~ (flux_true, zqso) result
		# sigma = [const1 * flux_trues^-0.5 * (1 + zqso)^power1, ...]
		#       = [const1 * flux_ref^-0.5 * (1 + sep)**power1 * (flux_trues/flux_ref)^-0.5  * ((1 + zqso_l)/(1 + sep))**power1, ...]
		paras_ftrue = result_ftrue.params
		print('sigma vs (flux_true, zqso):', paras_ftrue)
		flux_ref = 100. # 10^-17 erg ...
		#factor = paras_ftrue['const1'].value * flux_ref**(-0.5) * (1. + paras_ftrue['sep'].value)**paras_ftrue['power1'].value
		#sepperr_over_sepp_2 = (paras_ftrue['power1'].value / (1. + paras_ftrue['sep'].value) * paras_ftrue['sep'].stderr)**2.\
		#                    + (np.log(1. + paras_ftrue['sep'].value) * paras_ftrue['power1'].stderr)**2. # (sigma((1+sep)^power1)/((1+sep)^power1))^2
		#factor_err = factor * np.sqrt((paras_ftrue['const1'].stderr / paras_ftrue['const1'].value)**2. + sepperr_over_sepp_2)
		#print(f'sigma_vdoti = ({factor:.2e} +/- {factor_err:.2e}) * (F/{flux_ref} * 10^-17 erg ...)^(-0.5) * ((1 + zqso)/(1 + zbreak))^power')
	if ivar == 0 and not sigma_as_R_sep: # for R, convert to sigma_vdot = C * (R / 50k)^power + const
		ref = 5e4 # reference R
		C_50k = paras['const1'].value * ref**paras['power1'].value
		C_50k_err = C_50k * np.sqrt((np.log(ref) * paras['power1'].stderr)**2. + (paras['const1'].stderr / paras['const1'].value)**2.)
		print('sigma_vdot = (%.2e +/- %.2e) (R / %d)^(%.2f +/- %.2f) + (%.5f +/- %.5f)'%(C_50k, C_50k_err, ref, paras['power1'].value, paras['power1'].stderr, paras['const2'].value, paras['const2'].stderr))
	print('Reduced chi:', result.redchi)
	# generate fitted curve
	if ivar == 0:
		variable_rg = np.logspace(3., 5., 1000) # R
		sigma_rg = sigma_as_R(variable_rg, paras, sep=sigma_as_R_sep)
	if ivar == 1:
		variable_rg = np.linspace(2., 5., 1000) # zqso
		sigma_rg = sigma_as_zqso_sep(variable_rg, paras)
		nphot_rg = nphot_as_zqso(variable_rg, paras_nphot)
		ftrue_rg = np.linspace(flux_true.min(), flux_true.max(), 1000)
		sigma_ftrue_rg = sigma_as_ftrue_zqso((ftrue_rg, np.r_[3.]), paras_ftrue) # fix zqso=3, plot sigma_i ~ flux_true
		sigma_zqso_rg = sigma_as_ftrue_zqso((np.r_[flux_true_mean], variable_rg), paras_ftrue) # fix flux_true to mean, plot sigma_i ~ zqso
		# sigma vs zqso for Liske
		paras_liske = {'sep': 4., 'power1': -1.7, 'power2': -0.9} # Liske eq 14
		z_equal = 3.9 # set sigma_mine(z_equal) == sigma_liske(z_qual)
		paras_liske['const1'] = paras['const1'] * (1. + paras['sep'])**(paras['power1'] - paras['power2']) * (1. + z_equal)**(paras['power2'] - paras_liske['power1'])
		sigma_liske = sigma_as_zqso_sep(variable_rg, paras_liske)

	# compute dvdt/sigma_dvdt
	if ivar == 0:
		zqsos = zqso + np.zeros(Rs.shape)
		zqso_rg = zqso + np.zeros(variable_rg.shape)
	if ivar == 1:
		zqso_rg = variable_rg
	zmids = (zqsos + su.z_lyb(zqsos)) / 2.
	z_errs = zqsos - zmids
	dvdts = np.abs(cs.dvdt(zmids))
	dvdtosigs = dvdts / sigmas
	dvdtosig_errs = np.abs(dvdtosigs * sigma_errs / sigmas)
	# generate shaded region for dvdtosig vs z
	zlyb_rg = su.z_lyb(zqso_rg)
	zmid_rg = (zqso_rg + zlyb_rg) / 2.
	dvdt_rg = np.abs(cs.dvdt(zmid_rg))
	dvdtosig_rg = dvdt_rg / sigma_rg
	## divide region into two
	imin = np.argmin(dvdtosig_rg)
	left = np.arange(len(dvdtosig_rg)) <= imin
	right = np.arange(len(dvdtosig_rg)) >= imin

	# plot
	plot = 1
	if plot:
		matplotlib.use('Agg')

		# plot dvdt/sigma_dvdt vs R/zmid
		fig, ax = plt.subplots()
		if ivar == 1: # dvdt/sigma_dvdt vs zmid
			#ax.errorbar(zmids, dvdtosigs, xerr=z_errs, yerr=dvdtosig_errs, fmt='ko', capsize=2)
			ax.errorbar(zmids, dvdtosigs, yerr=dvdtosig_errs, fmt='ko', capsize=2)
			ax.plot(zmid_rg, dvdtosig_rg, 'b') # fitted line
			ax.fill_betweenx(y=dvdtosig_rg, x1=zlyb_rg, x2=variable_rg, where=left, color='b', alpha=0.5, lw=0)
			ax.fill_betweenx(y=dvdtosig_rg, x1=zlyb_rg, x2=variable_rg, where=right, color='b', alpha=0.5, lw=0)
			# add labels for lya and lyb
			ax.plot([zlyb_rg[-1], zlyb_rg[-1]], [dvdtosig_rg[-1], dvdtosig_rg[-1]+0.5], 'k', lw=1) # lyb indicator
			ax.plot([variable_rg[-1], variable_rg[-1]], [dvdtosig_rg[-1], dvdtosig_rg[-1]+0.5], 'k', lw=1) # lya indicator
			ax.text(zlyb_rg[-1], dvdtosig_rg[-1]+0.5, 'Ly$\\beta$', ha='center', va='bottom')
			ax.text(variable_rg[-1], dvdtosig_rg[-1]+0.5, 'Ly$\\alpha$', ha='center', va='bottom')
			#ax.set_ylim([-1, 17])
		if ivar == 0: # dvdt/sigma_dvdt vs R
			#ax.errorbar(Rs, dvdtosigs, yerr=dvdtosig_errs, fmt='ko', capsize=2)
			ax.plot(Rs, dvdtosigs, 'ko') # no errorbar
			ax.plot(variable_rg, dvdtosig_rg, 'b') # fitted line
			# create another y axis showing vdot/sigma percentage of R=100k
			add_pct = True
			if add_pct:
				ax2 = ax.twinx()
				pct = dvdtosigs / np.max(dvdtosigs)
				print('Rs:            %s\nvdot/sigma percent: %s'%(Rs, pct))
				pct_rg = dvdtosig_rg / np.max(dvdtosigs)
				line = ax2.plot(Rs, pct, 'ko')[0] # no errorbar
				line_rg = ax2.plot(variable_rg, pct_rg, 'b')[0] # fitted line
				line.set_visible(False)
				line_rg.set_visible(False)
		# fitted line
		ax.set_xlabel(varnames_dvdtosig[ivar])
		ax.set_ylabel('$|\dot{v}|/\sigma_\dot{v}$')
		#ax.tick_params(which='major', length=6)
		#ax.tick_params(which='minor', length=3)
		# set axis ticks
		if ivar == 1: # for dvdt/sigma_dvdt vs zmid
			ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
			ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
			ax.yaxis.set_major_locator(plt.MultipleLocator(2))
			ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
		if ivar == 0: # for dvdt/sigma_dvdt vs R
			#ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
			ax.xaxis.set_minor_locator(plt.MultipleLocator(10000))
			ax.set_xticklabels(['%d'%(i/1e3) for i in ax.get_xticks()])
			#ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
			ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
			if add_pct:
				ax2.set_ylabel('$\\frac{|\dot{v}|/\sigma_\dot{v}}{|\dot{v}|/\sigma_\dot{v}(R=100\mathrm{k})}$')
				ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
				ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
		fig.tight_layout()
		tosave = path + 'plots/dvdtosig_%s%s.pdf'%(varshort_dvdtosig[ivar], nphottxt)
		fig.savefig(tosave); print('Saved: %s'%tosave)
		plt.close(fig)

		# plot sigma_dvdt vs R/zqso
		fig, ax = plt.subplots()
		ax.errorbar(variables[ivar], sigmas, yerr=sigma_errs, fmt='ko', capsize=2)
		# fitted line
		ax.plot(variable_rg, sigma_rg, 'b', label='This work')
		#if ivar==1: # overplot liske
		#	ax.plot(variable_rg, sigma_liske, 'r--', label='Liske et al. 2008') # overplot liske
		#	ax.legend() # uncomment if overplot liske
		title = ','.join(['%s:%.2f'%(key, paras[key].value) for key in paras.keys()])
		title += ',rchi:%.2f'%(result.redchi)
		#ax.set_title(title)
		# wrap up
		ax.set_xlabel(varnames[ivar])
		ax.set_ylabel('$\sigma_\dot{v}$ (cm s$^{-1}$ yr$^{-1}$)')
		ax.set_xscale('log')
		ax.set_yscale('log')
		#ax.tick_params(which='major', length=6)
		#ax.tick_params(which='minor', length=3)
		# don't use scientific notation
		ax.xaxis.set_major_formatter(plt.ScalarFormatter())
		ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
		ax.yaxis.set_major_formatter(plt.ScalarFormatter())
		ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
		if ivar == 1: # for sigma_dvdt vs zqso
			ax.xaxis.set_major_locator(plt.MultipleLocator(1))
			ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
		if ivar == 0: # for sigma_dvdt vs R
			ax.set_xticklabels(['%d'%(i/1e3) for i in ax.get_xticks()])
			ax.yaxis.set_ticklabels([], minor=True)
		ax.xaxis.set_ticklabels([], minor=True)
		fig.tight_layout()
		tosave = path + 'plots/sigmadvdt_%s%s.pdf'%(varshort[ivar], nphottxt)
		fig.savefig(tosave); print('Saved: %s'%tosave)
		#plt.show()

		# plot nphot vs zqso
		if ivar == 1:
			# plot nphot vs zqso
			fig, ax = plt.subplots()
			ax.errorbar(variables[ivar], nphots, yerr=nphot_errs, fmt='ko', capsize=2)
			# fitted line
			ax.plot(variable_rg, nphot_rg, 'b', label='This work')
			ax.set_xlabel(varnames[ivar])
			ax.set_ylabel('$N_\mathrm{photon}$')
			fig.tight_layout()
			tosave = path + 'plots/nphot_%s%s.pdf'%(varshort[ivar], nphottxt)
			fig.savefig(tosave); print('Saved: %s'%tosave)

			# plot sigma_i vs zqso/flux_true
			fig, axes = plt.subplots(1, 2)
			# --- plot sigma_i vs zqso
			# fluxtrue_zqso_sigs: (npix, 4), flux_true, zqso, sigs, sigs_err)
			axes[0].errorbar(*fluxtrue_zqso_sigs.T[[1, 2]], yerr=fluxtrue_zqso_sigs[:, 3], fmt='ko', capsize=2)
			# fitted line
			axes[0].plot(variable_rg, sigma_zqso_rg, 'b', label='This work')
			axes[0].set_xlabel(varnames[ivar])
			axes[0].set_ylabel('$\sigma_{\dot{v}, i}$ (cm s$^{-1}$ yr$^{-1}$)')
			# --- plot sigma_i vs flux_true
			axes[1].errorbar(*fluxtrue_zqso_sigs.T[[0, 2]], yerr=fluxtrue_zqso_sigs[:, 3], fmt='ko', capsize=2)
			# fitted line
			axes[1].plot(ftrue_rg, sigma_ftrue_rg, 'b', label='This work')
			axes[1].set_xlabel('Continuum flux ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)')
			axes[1].set_ylabel('$\sigma_{\dot{v}, i}$ (cm s$^{-1}$ yr$^{-1}$)')
			fig.tight_layout()
			tosave = path + 'plots/sigmadvdti_ftrue_%s%s.png'%(varshort[ivar], nphottxt)
			fig.savefig(tosave); print('Saved: %s'%tosave)
	return paras

if __name__ == '__main__':
	const_nphot = False # if True, constant continuum, else real quasar continuum; default should be False
	ivar = 1 # 0 for R, 1 for zqso, modify here and the for loop below
	fixres = 'res' # 'res' or 'resele', default should be 'res'
	plot_paradist = False # whether plot distribution of parameters of fitting sigma vs R/zqso
	if plot_paradist: load = False

	# combine all fitted paras to para_all
	varshort = ['R', 'zqso']
	if not plot_paradist:
		nphot = 1e8 if const_nphot else None
		kid = 104297
		paras = explore_setup(ivar=ivar, const_nphot=const_nphot, nphot=nphot, kid=kid, fixres=fixres)
	else:
		contdegtxt = '' if ss.keck_args.fitcont_deg is None else '_contdeg%d'%ss.keck_args.fitcont_deg
		para_all_tosave = path + 'paras/paradist_sigmadvdt_%s_constcont%d%s.pkl'%(varshort[ivar], const_nphot, contdegtxt)
		if load: para_all = pkload(para_all_tosave, verbose=True)
		else:
			nphot = None; kid = None
			if const_nphot:
				nphots = [1e8]
				variables = nphots
			else:
				koajobids = rvs.kid_withspec()
				variables = koajobids
			para_all = {} # {parname: [val1, val2, ...]} para list for all var
			for var in variables: # var is nphot or kid
				if const_nphot:
					nphot = var
					print('Nphot: %e'%nphot)
				else:
					kid = var
					print('KOAjobID:', kid)
				paras = explore_setup(ivar=ivar, const_nphot=const_nphot, nphot=nphot, kid=kid)
				for key in paras.keys():
					val = paras[key].value
					try: para_all[key].append(val)
					except KeyError: para_all[key] = [val]
			pkdump(para_all, para_all_tosave)

		# plot hist of para_all
		npara = len(para_all)
		fig, axes = plt.subplots(*subplot_shape(npara))
		axes = axes.flatten()
		for ipara, key in enumerate(para_all.keys()):
			axes[ipara].hist(para_all[key], density=False, bins=10, histtype='step')
			axes[ipara].set_title(key)
		fig.tight_layout()

		hist_tosave = path + 'plots/paradist_sigmadvdt_%s_constcont%d%s.pdf'%(varshort[ivar], const_nphot, contdegtxt)
		fig.savefig(hist_tosave); print('Saved: %s'%hist_tosave)
		plt.show()

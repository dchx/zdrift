from utils import *
import simulation_setup as ss
import liske_sigma as ls
import read_elqs as re

def item2sig(item, dts, linelistmode, exptime_perobs, genspec_args, keck_args, obs_setup):
	# get templates
	lam, flux_temps = ss.multi_epoch_templates(dts, linelistmode, genspec_args, keck_args, obs_setup)
	# add noise
	#item_cont = df_all.loc[104297.] # only use continuum flux
	#item_cont = item_cont.append(pd.Series([item_cont['M1450'], item_cont['z']], index=['M1450_origin', 'z_origin']))
	item_cont = item
	fluxes, errs = ss.add_noise(item_cont, lam, flux_temps, obs_setup, exptime_perobs, zqso=item['z'], verbose=False, substituting=False, fitcont_deg=keck_args.fitcont_deg)
	# compute sigma
	dvdti, sig2_dvdti = ls.liske_dvdti(lam, fluxes, errs, dts) # in cm/s/yr
	dvdt_est = ls.wa_1d(dvdti, sig2_dvdti) # scalar
	sigma_dvdt = ls.overall_sigma(sig2_dvdti) # scalar
	return dvdt_est, sigma_dvdt

def kid_withspec(exclude_weak=True):
	toglob = path + 'paras/voigtforest_bestparray_*_deg10_fitaddline*.pzip'
	fs = glob.glob(toglob)
	koajobids = [int(f.split('bestparray_')[1].split('_smooth')[0]) for f in fs]
	if exclude_weak:
		koajobids.remove(12850) # spectrum too low S/N, drop
		#koajobids.remove(9075) # can remove this line latter
		pass
	return koajobids

def sigma_realll_vs_simll():
	# parameters
	nepoch = 2
	exptime_perobs = 26000. * 3600. # 50 h per week, 20 years, 2 epochs, 1 qso

	# derived parameters
	dts = np.linspace(0., ss.obs_setup.period, nepoch) # time points for each epoch, in years

	# get koajobids
	koajobids = kid_withspec()
	print('koajobids:', koajobids, len(koajobids))

	sigma_reals = []; sigma_sims = []; sigma_sim_errs = []; zqsos = []
	for koajobid in koajobids:
		print('koajobid', koajobid)
		item = df_all.loc[koajobid]
		#item = item.append(pd.Series([item['M1450'], item['z']], index=['M1450_origin', 'z_origin']))
		item = item.append(pd.Series([item['z']], index=['z_origin']))

		ss.genspec_args.zqso = item['z'] # spectra's own z
		zqsos.append(item['z'])
		ss.keck_args.item = item
		args = ss.genspec_args, ss.keck_args, ss.obs_setup

		# real line list
		dvdt_est, sigma_dvdt = item2sig(item, dts, 'keck', exptime_perobs, ss.genspec_args, ss.keck_args, ss.obs_setup)
		sigma_reals.append(sigma_dvdt)

		# simulated line list
		sigma_pervar = []
		for ispec in range(10):
			ss.genspec_args.ispec = ispec
			dvdt_est, sigma_dvdt = item2sig(item, dts, 'genspec', exptime_perobs, ss.genspec_args, ss.keck_args, ss.obs_setup)
			sigma_pervar.append(sigma_dvdt)
		sigma_sims.append(np.mean(sigma_pervar))
		sigma_sim_errs.append(np.std(sigma_pervar, ddof=1))
	return sigma_reals, sigma_sims, sigma_sim_errs, zqsos, koajobids

def plot_realll_vs_simll():
	sigma_reals, sigma_sims, sigma_sim_errs, zqsos, koajobids = sigma_realll_vs_simll()
	# compute squared error
	ndata = len(sigma_reals)
	print('ndata:', ndata) # debug
	sq_err = (np.array(sigma_reals) - np.array(sigma_sims))**2. # squared error
	sq_err_rel = sq_err / np.array(sigma_reals)**2. # relative error
	mse = np.sum(sq_err) / ndata # scalar
	mse_rel = np.sum(sq_err_rel) / ndata # scalar
	rms = np.sqrt(mse) # scalar
	rms_rel = np.sqrt(mse_rel) # scalar
	print('RMS: %.6f, RMS_rela: %.4f '%(rms, rms_rel * 100.) + '%')
	chi2 = sq_err / np.array(sigma_sim_errs)**2.
	mchi2 = np.sum(chi2) / ndata # scalar

	contdegtxt = '' if ss.keck_args.fitcont_deg is None else '_contdeg%d'%ss.keck_args.fitcont_deg
	minlwtxt = '' if ss.keck_args.min_lw==0 else '_minlw%skms'%ss.keck_args.min_lw
	
	# one-to-one plot
	fig, ax = plt.subplots(figsize=(5, 5))
	ax.errorbar(sigma_reals, sigma_sims, yerr=sigma_sim_errs, fmt='ob', capsize=3)
	axis = ax.axis()
	ax.plot([0, 100], [0, 100], 'k')
	ax.axis(axis)
	ax.errorbar(sigma_reals, sigma_sims, yerr=sigma_sim_errs, fmt='ob', capsize=3)
	ax.set_aspect('equal')
	ax.set_xlabel(r'$\sigma_\dot{v}$ (real line list; cm s$^{-1}$ yr$^{-1}$)')
	ax.set_ylabel(r'$\sigma_\dot{v}$ (synthetic line list; cm s$^{-1}$ yr$^{-1}$)')
	fig.tight_layout()
	tosave = path + 'plots/sigma_realll_vs_simll%s.pdf'%(contdegtxt + minlwtxt)
	fig.savefig(tosave); print('Saved: %s'%tosave)
	#plt.show()
	plt.close(fig)

	# sigma_sim/sigma_real vs zqso
	ratio = np.array(sigma_sims) / sigma_reals
	ratio_err = np.array(sigma_sim_errs) / sigma_reals
	fig, ax = plt.subplots(figsize=(5, 5))
	ax.axhline(1, color='k')
	ax.errorbar(zqsos, ratio, yerr=ratio_err, fmt='ob', capsize=3)
	addtext = False
	addtexttxt = '_withtext' if addtext else ''
	if addtext:
		for i in range(len(zqsos)): ax.text(zqsos[i], ratio[i], str(koajobids[i]))
	ax.set_xlabel(r'$z_\mathrm{QSO}$')
	ax.set_ylabel(r'$\\frac{\sigma_\dot{v}\ (\mathrm{synthetic\ line\ list})}{\sigma_\dot{v}\ (\mathrm{real\ line\ list})}$')
	fig.tight_layout()
	tosave = path + 'plots/sigma_realllsimll_vs_zqso%s.pdf'%(contdegtxt + minlwtxt + addtexttxt)
	fig.savefig(tosave); print('Saved: %s'%tosave)
	#plt.show()
	plt.close(fig)

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
	plot_realll_vs_simll()

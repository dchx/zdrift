from utils import *
import lmfit as lf
import simulation_setup as ss
from matplotlib.lines import Line2D
'''
fit a relation for vds vs aperture and time
'''

def vds_as_aperture_period(aperture, period, paras):
	vds = paras['a'] * aperture * (period)**(1.5)
	return vds

def residual(paras, aperture, period, vds, vds_err=1.):
	'''
	vds = a * D * t^(3/2)
	'''
	# prediction
	vds_pred = vds_as_aperture_period(aperture, period, paras)
	# residual
	vds_err = np.atleast_1d(vds_err)
	if np.all(vds_err==0): vds_err = np.ones(vds_err.shape)
	res = (vds - vds_pred) / vds_err
	return res

def fit_vds_aperture_period(aperture, period, vds, vds_err=1.):
	'''
	fit vdot/sigma vs aperture and period
	  vds = a * D * t^(3/2)
	'''
	# fit
	paras = lf.Parameters()
	paras.add('a', value=1e-3, min=0.)
	result = lf.minimize(residual, paras, args=(aperture, period, vds, vds_err))
	return result

def aperture_vs_period(paras, ax=None, tosave=None, color='b', legend=True, plot_5sig=True, **kwargs):
	snrs = [3, 5] # the wanted vdot/sigma values
	if not plot_5sig: snrs = [3]
	linestyles = ['-', '--']
	period = np.linspace(10, 40, 100) # year
	if type(ax)==type(None):
		gen_fig = True
		fig, ax = plt.subplots(figsize=(5,4))
	else: gen_fig = False
	for isnr, snr in enumerate(snrs):
		aperture = snr / paras['a'] * period**(-1.5)
		ax.plot(period, aperture, color + linestyles[isnr], label='%d$\\sigma$ detection'%snr, **kwargs)
	ax.set_xlabel('Time (year)') 
	ax.set_ylabel('Aperture (m)')
	ax.axis([10, 40, 10, 50])
	if legend: ax.legend()
	ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
	ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
	if gen_fig:
		fig.tight_layout()
		if type(tosave)!=type(None):
			fig.savefig(tosave); print('Saved: %s'%tosave)
		plt.show()

def get_setup_require(sigma, factor, aperture=None, period=None, efficiency=None, fid_aperture=20., fid_period=20., fid_efficiency=0.25):
	'''
	Given sigma, compute required apertur/period
	sigma = factor * (D / fid_aperture) (t / fid_period)^(3/2) (e / fid_efficiency)^1/2
	'''
	Dt32e12 = sigma / factor # (D / fid_aperture) (t / fid_period)^(3/2) (e / fid_efficiency)^1/2
	if aperture is not None: D = aperture / fid_aperture
	if period is not None: t32 = (period / fid_period)**(3./2.)
	if efficiency is not None: e12 = (efficiency / fid_efficiency)**(1./2.)
	if period is None:
		assert aperture is not None and efficiency is not None
		t32 = Dt32e12 / D / e12
		t = t32**(2./3.)
		period = t * fid_period
		return period
	if aperture is None:
		assert period is not None and efficiency is not None
		D = Dt32e12 / t32 / e12
		aperture = D * fid_aperture
		return aperture
	if efficiency is None:
		assert aperture is not None and period is not None
		e12 = Dt32e12 / D / t32
		e = e12**(2.)
		efficiency = e * fid_efficiency
		return efficiency

def main(default_aperture=15., default_period=20., figure=9, selection='vds'):
	'''
	figure - 9 or 11
	selection - 'm14' or 'vds'
	'''
	if selection=='vds':
		if figure==9:
			nqsos_3sig = [1, 'plan', 5] # paper figure 9; int or 'plan'
			nqsos_5sig = ['plan'] # paper figure 9
		elif figure==11:
			nqsos_3sig = ['plan_add1MagBrighter', 'plan_add0.5MagBrighter', 'plan_add0MagBrighter', 'plan'] # paper figure 11; int or 'plan'
			nqsos_5sig = []
		#nqsos_3sig = [1, 2, 'plan', 3, 5, 10] # int or 'plan'
		# change added mag
		#nqsos_3sig = ['plan_add2MagBrighter', 'plan_add1MagBrighter', 'plan_add0.5MagBrighter', 'plan_add0MagBrighter', 'plan', 5, 10] # int or 'plan'
		#nqsos_3sig = ['plan_add2MagBrighter_top10.5MagBrighter', 'plan_add1MagBrighter_top10.5MagBrighter', 'plan_add0.5MagBrighter_top10.5MagBrighter', 'plan_add0.0MagBrighter_top10.5MagBrighter', 'plan_top10.5MagBrighter', 'plan'] # int or 'plan'
		# change top1 mag
		#nqsos_3sig = ['plan_top12MagBrighter', 'plan_top11MagBrighter', 'plan_top10.5MagBrighter', 'plan', 5]
		#nqsos_3sig = ['plan_add0.0MagBrighter_top12MagBrighter', 'plan_add0.0MagBrighter_top11MagBrighter', 'plan_add0.0MagBrighter_top10.5MagBrighter', 'plan', 5]
		#nqsos_5sig = ['plan_add1MagBrighter', 'plan_add0.5MagBrighter', 'plan_add0MagBrighter', 'plan'] # int or 'plan'
		#nqsos = ['plan', '3mimicplan', 2, '3concentrateMimic', '3testConcentrateTime', 3] # int or 'plan'
	elif selection=='m14':
		nqsos_3sig = [5]
		nqsos_5sig = [5]
	colors_nqso = ['r', 'b', 'g', 'c', 'm', 'y']
	varnames = ['aperture', 'period']
	# plot setup
	fig, axes = plt.subplots(1, 2, figsize=(8, 4))
	fig35, ax35 = plt.subplots(figsize=(5,4))
	title = ''
	contdegtxt = '' if ss.keck_args.fitcont_deg is None else '_contdeg%d'%ss.keck_args.fitcont_deg
	lines = []; legends = [] # for plot 35 sigma
	factors = {}
	for inqso, nqso in enumerate(nqsos_3sig):
		# initialize data
		data = np.array([[]]*4) # aperture, period, vds, vds_err
		fitted_product = [] # [(aperture_fit, period_fit), ...]
		for ivar, varname in enumerate(varnames):
			# load product
			f = ss.vds_filename(varname, nqso, selection, contdegtxt)
			vds_product = pkload(f) # (2(3), ndata) variable, vds(, vds_err)
			if len(vds_product)==2:
				axes[ivar].plot(*vds_product, 'ok') # plot data points; variable, vds
				vds_product = np.vstack([vds_product, np.zeros(vds_product.shape[1])]) # (3, ndata) variable, vds, vds_err
			else: axes[ivar].errorbar(*vds_product, 'ok', capsize=3) # plot data points; variable, vds, vds_err
			# add default values
			if varname=='aperture': default = default_period
			elif varname=='period': default = default_aperture
			if np.isscalar(default): default = np.array([default]*vds_product.shape[1]) # (ndata,) make default a vetor
			vds_product = np.insert(vds_product, 1-ivar, default, axis=0) # (4, ndata) aperture, period, vds, vds_err
			# append to data
			data = np.hstack([data, vds_product]) # (4, ndatas)
			# variable grid for fitted curve
			variable = vds_product[ivar]
			n_grid = 100 # number of data points in the grid
			var_fit = np.linspace(min(variable), max(variable), n_grid)
			default_fit = np.zeros(n_grid) + default[0]
			if varname=='aperture': fitted_product.append((var_fit, default_fit)) # (aperture_fit, period_default)
			elif varname=='period': fitted_product.append((default_fit, var_fit)) # (aperture_default, period_fit)
		# fit
		result = fit_vds_aperture_period(*data)
		print(f'\nnqso {nqso}:', result.params)
		print('Reduced chi:', result.redchi)
		# fit result
		factor = result.params['a'].value
		factors[nqso] = factor
		fid_aperture = 20. # m, fiducial aperture
		fid_period = 20. # yr, fiducial period
		factor = factor * fid_aperture * fid_period**(3./2.)
		# print information
		print(f'vdot/sigma = {factor:.5f} (D / {fid_aperture} m) (t / {fid_period} yr)^(3/2)')
		if selection=='vds':
			if figure==9: Ds = [42., 25.]
			elif figure==11: Ds = [20.]
			for D in Ds:
				t_3sig = get_setup_require(3, factor, aperture=D, period=None, efficiency=0.25, fid_aperture=fid_aperture, fid_period=fid_period)
				t_4sig = get_setup_require(4, factor, aperture=D, period=None, efficiency=0.25, fid_aperture=fid_aperture, fid_period=fid_period)
				t_5sig = get_setup_require(5, factor, aperture=D, period=None, efficiency=0.25, fid_aperture=fid_aperture, fid_period=fid_period)
				print(f'For {D} m, 3sigma: {t_3sig:.4f} yr; 4sigma: {t_4sig:.4f} yr, 5sigma: {t_5sig:.4f} yr')
			if figure==9:
				D = 25.
				efficiency = 0.35
				t_3sig = get_setup_require(3, factor, aperture=D, period=None, efficiency=efficiency, fid_aperture=fid_aperture, fid_period=fid_period)
				print(f'For {D} m, e = {efficiency}, 3sigma: {t_3sig:.4f} yr')

		# plot fit
		#title += '$|\dot{v}|/\sigma_\dot{v} = $%.3e$\ D\ t^{3/2}$\n'%result.params['a'].value
		title += '$|\dot{v}|/\sigma_\dot{v} = $%.5f$\ (D/$%d m$)\ (t/$%d yr$)^{3/2}$\n'%(factor, fid_aperture, fid_period)
		label_nqso = ('Uniform (%d QSOs)'%nqso if nqso > 1 else 'Idealized (%d QSO)'%nqso) if np.isreal(nqso) else nqso.replace('_add', ' Case ').replace('MagBrighter', '').title()
		if label_nqso=='Plan' and figure==9: label_nqso = 'Plan (5 QSOs)'
		for ivar, varname in enumerate(varnames):
			vds_fit = vds_as_aperture_period(*fitted_product[ivar], result.params)
			var_fit = fitted_product[ivar][ivar]
			#if ivar==1: print(var_fit, vds_fit)
			axes[ivar].plot(var_fit, vds_fit, colors_nqso[inqso], label=label_nqso) # plot fitted
			# wrapup plot
			if varname=='aperture': axes[ivar].set_xlabel('Aperture (m)') 
			elif varname=='period': axes[ivar].set_xlabel('Time (year)') 
		# plot 35 sigma
		linewidth = 1.5 if np.isreal(nqso) else 3 # highlight plan
		linewidth = 1.5 # don't highlight
		kwargs = {'lw': linewidth}
		if nqso in nqsos_5sig: plot_5sig = True
		else: plot_5sig = False
		aperture_vs_period(result.params, ax=ax35, color=colors_nqso[inqso], legend=False, plot_5sig=plot_5sig, **kwargs)
		lines.append(Line2D([0], [0], linestyle='-', color=colors_nqso[inqso], **kwargs))
		legends.append(label_nqso)
	# information about factors
	if selection=='vds':
		nqso_fid = 'plan' if figure==11 else 5 if figure==9 else None
		factor_fid = factors[nqso_fid]
		snr_change = {k: round((v/factor_fid - 1)*100., 3) for k, v in factors.items()}
		time_change = {k: round(((v/factor_fid)**(-2./3.) - 1)*100., 3) for k, v in factors.items()}
		aperture_change = {k: round((factor_fid/v - 1)*100., 3) for k, v in factors.items()}
		print(f'\nS/N change from {nqso_fid} by {snr_change} percent')
		print(f'required time change from {nqso_fid} by {time_change} percent')
		print(f'required aperture change from {nqso_fid} by {aperture_change} percent')
	# plot fit
	axes[0].set_ylabel('$|\dot{v}|/\sigma_\dot{v}$') # (vdot/sigma)
	axes[1].legend()
	fig.suptitle(title)
	fig.tight_layout()
	selection_txt = '' if selection=='vds' else '_m14'
	fig_tosave = path + 'plots/vds_aperture_period_fit%s.pdf'%(selection_txt + contdegtxt)
	fig.savefig(fig_tosave); print('Saved: %s'%fig_tosave)

	# plot aperture vs period
	# custom legend
	for lstyle in ['-', '--']: lines.append(Line2D([0], [0], linestyle=lstyle, color='k'))
	if len(nqsos_5sig)!=0 and len(nqsos_3sig)!=0:
		#legends = legends + ['$|\dot{v}|/\sigma_\dot{v} = %d$'%snr for snr in [3, 5]]
		legends = legends + ['$%d \sigma$ detection'%snr for snr in [3, 5]]
	if len(nqsos_3sig)!=0 and len(nqsos_5sig)==0: ax35.axis([10, 30, 10, 50])
	ax35.legend(lines, legends)
	fig35.tight_layout()
	fig35_tosave = path + 'plots/aperture_vs_period_35sigma%s.pdf'%(selection_txt + contdegtxt)
	fig35.savefig(fig35_tosave); print('Saved: %s'%fig35_tosave)
	#plt.show()

if __name__=='__main__':
	plot_alpha_vs_addedhours = False
	if plot_alpha_vs_addedhours:
		hours = [0., 0.18, 5.18, 10.18, 30.18, 50.18, 80.18, 100.18, 200.18, 300.18]
		alpha = [6.03244, 4.13933, 4.28199, 4.42044, 4.93378, 5.39885, 6.02957, 6.41569, 8.07392, 9.44539]
		fig, ax = plt.subplots()
		ax.plot(hours, alpha, '-o')
		ax.set_xlabel('Jan-Feb time for added QSO (hour)')
		ax.set_ylabel('$\\alpha$')
	else:
		for figure in [9, 11]:
			main(figure=figure, selection='vds')
		main(figure=figure, selection='m14')

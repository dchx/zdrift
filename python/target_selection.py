from utils import *
import spec_utils as su
import cosmology as cs
import read_elqs as re
import read_ps as rp
import m1450

labels = ['$|\dot{v}|/\sigma_\dot{v}$ selection', '$M_{1450}$ selection', 'Pan-STARRS']

def plot_subz():
	'''
	help decide how to do spectra substitution
	'''
	tsub = pd.read_csv(path + 'data/substitution.csv')
	ys = [[tsub.zsub_all, tsub.zsub_all], [tsub.zsub_m14, tsub.zsub_vds]]
	titles = ['Redshift first', 'Uniqueness first']

	for iplot in range(2):
		fig, ax = plt.subplots()
		ax.plot(tsub.z_m1450, ys[iplot][0], 'xr', label=labels[1], ms=10, mew=2)
		ax.plot(tsub.z_vds, ys[iplot][1], '+b', label=labels[0], ms=10, mew=2)
		axis = ax.axis()
		ax.plot([0., 6.], [0., 6.], 'k')
		ax.set_xlabel('$z_\mathrm{origin}$')
		ax.set_ylabel('$z_\mathrm{substitute}$')
		ax.set_aspect('equal')
		ax.axis(axis)
		ax.legend()
		ax.set_title(titles[iplot])
		fig.tight_layout()
		plt.show()

if __name__ == '__main__':
	matplotlib.rc('font',size=10) # global font size
	period = 20. # yr
	exptime_per_epoch = 44. * 3600. # in seconds, per qso, 50 h per week for 10 qsos
	nepoch = 121 # for 20 years
	D = 15. # m, diameter
	eff = 0.25 # efficiency

	# plot every qso as one point
	colors = ['r', 'b', 'g']
	lines = ['-', ':', '--']
	fills = ['left', 'right', 'full']
	#fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	fig, ax = plt.subplots(figsize=(5, 5))
	axes = [None, ax]
	#for ind, top10 in enumerate([re.top10vds_nosub, re.top10m14_nosub, df_elqs_ps]):
	for ind, top10 in enumerate([rp.top10vds_N_nosub, rp.top10m14_N_nosub]):
		print('sample')
		for i,item in enumerate(top10.iloc):
			if np.isnan(item['rmag']): continue
			# dvdt/sigma
			zqso = item['z']
			zlyb = su.z_lyb(zqso)
			zmid = (zqso + zlyb) / 2.
			zerr = (zqso - zlyb) / 2.
			dvdt_theory, sigma_dvdt = re.dvdt_over_sigma(zqso, item['rmag'], exptime_per_epoch*nepoch, D, eff, nepoch, period, False)
			vds = np.abs(dvdt_theory)/sigma_dvdt # vdot / sigma
			print('vds:', vds)
			# m1450
			m14 = item['M1450']
			f1450 = m1450.m14502sdss(m14, zqso)
			'''
			if ind==1: # for m1450 selection, shift a bit
				#zmid += 0.01
				dvdt_theory -= 0.01 * sigma_dvdt
				f1450 -= 1
			'''
			label = labels[ind] if i==0 else None
			# plot f1450
			'''
			eb0 = axes[0].errorbar(zmid, f1450, xerr=zerr, fmt=',', marker=None, color=colors[ind], capsize=3, elinewidth=2, label=label)
			eb0[-1][0].set_linestyle(lines[ind])
			'''
			# plot dvdt/sigma
			#axes[1].errorbar(zmid, dvdt_theory, xerr=zerr, yerr=sigma_dvdt, fmt='o', color=colors[ind])
			eb1 = axes[1].errorbar(zmid, vds, xerr=zerr, fmt=',', marker=None, color=colors[ind], capsize=3, elinewidth=2, label=label)
			eb1[-1][0].set_linestyle(lines[ind])
			axes[1].plot(zqso, vds, 'o', c=colors[ind], fillstyle=fills[ind])
	'''
	# f1450
	# plot cosmology
	zlim = ax[0].get_xlim()
	zgrid = np.linspace(*zlim, 1000)
	dvdt_grid = cs.dvdt(zgrid)
	axes[0].plot(zgrid, dvdt_grid, 'k')
	axes[0].axhline(0., color='k')
	#axes[0].set_xlim(zlim)
	axes[0].set_xlabel('$z$')
	axes[0].set_ylabel('$f_{1450}$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
	'''
	# dvdt/sigma
	axes[1].legend()
	#axes[1].axhline(0., color='k')
	#axes[1].set_xlim(zlim)
	axes[1].set_ylim([0., axes[1].get_ylim()[1]])
	axes[1].set_xlabel('$z$')
	axes[1].set_ylabel('$|\dot{v}|/\sigma_\dot{v}$')
	fig.tight_layout()
	tosave = path + 'plots/target_selection.pdf'
	fig.savefig(tosave); print('Saved: %s'%tosave)
	plt.show()

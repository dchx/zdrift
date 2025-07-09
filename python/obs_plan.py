from utils import *
import astroplan as ap
#ap.download_IERS_A()
import read_ps as rp
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord

def change_topn_mag(topn, ind, mag_brighter, ind_method='loc'):
	topn = copy.deepcopy(topn)
	magkeys = [] # column names to change mag
	if 'M1450_origin' in topn.columns: magkeys.append('M1450_origin')
	elif 'M1450' in topn.columns: magkeys.append('M1450')
	for magkey in ['rmag', 'imag']:
		if magkey in topn.columns: magkeys.append(magkey)
	# do mag change
	for magkey in magkeys:
		if ind_method=='loc': topn.at[ind, magkey] = topn.at[-1, magkey] - mag_brighter
		elif ind_method=='iloc': topn.iloc[ind, topn.columns.get_loc(magkey)] = topn.iloc[ind, topn.columns.get_loc(magkey)] - mag_brighter
	return topn

def topn_addnew(topn, addfake=None, mode='all', changeTop1=None):
	'''
	addfake - if None, use selected targets
	          else number, how many magnitude brighter to add a fake qso
	          #if 'top1', add another top1 source but opposite season
	          #if 'brighter', add another source half a mag brighter but opposite season
	mode - if 'ra' change RA of new qso
	       if 'mag' change magnitude of new qso
	       if 'all' or 'both' change RA and magnitude of new qso
	changeTop1 - how much brighter to change top1
	'''
	topn = copy.deepcopy(topn)
	if mode=='both': mode=='all'
	if (addfake is not None): # insert new qso to topn
		topn = pd.concat([pd.DataFrame(topn.iloc[0]).T.set_index(pd.Index([-1])), topn]) # new qso use top1 as template, index is -1
		topn.at[-1, 'Name'] = 'New Discovered'
		if mode=='ra' or mode=='all':
			topn.at[-1, 'RA'] = (topn.at[-1, 'RA'] + 180) % 360 # to the opposite season
		if (mode=='mag' or mode=='all') and (addfake=='brighter' or np.isreal(addfake)):
			if np.isreal(addfake): mag_brighter = addfake
			else: mag_brighter = 0.5 # make it half mag brighter
			topn = change_topn_mag(topn, -1, mag_brighter, 'loc')
	if (changeTop1 is not None) and (changeTop1!=0) and (mode=='mag' or mode=='all'):
		if addfake is None: top1_iloc = 0
		else: top1_iloc = 1
		topn = change_topn_mag(topn, top1_iloc, changeTop1, 'iloc')
	return topn

def main(site='lapalma', year=2021, nqso=5, hemisphere='N', tosave=None, addfake=None):
	'''
	addfake - if None, use selected targets
	          else number, how many magnitude brighter to add a fake qso
	          #if 'top1', add another top1 source but opposite season
	          #if 'brighter', add another source half a mag brighter but opposite season
	'''
	# parameters
	altitude_threshold = 30. * u.deg # altitude requirement of the targets
	nmonth_perobs = 2 # how many months per observation
	
	# location
	observer = ap.Observer.at_site(site)

	# time
	cur_time = Time('%d-01-01'%year) # 01/01
	cur_time = observer.twilight_evening_astronomical(cur_time, 'next') # go to next astronomical twilight evening
	end_time = Time('%d-01-01'%(year + 1)) # next year 01/01
	end_time = observer.twilight_evening_astronomical(end_time, 'next') # go to next astronomical twilight evening
	# -- per observation
	months_start = np.arange(1, 13, nmonth_perobs)
	start_time_perobs = Time(['%d-%02d-01'%(year, month) for month in months_start])
	start_time_perobs = observer.twilight_evening_astronomical(start_time_perobs, 'next') # go to next astronomical twilight evening
	nobs = len(start_time_perobs) # 12 / nmonth_perobs
	
	# targets
	topn = eval('rp.top10vds_%s_nosub[:nqso]'%hemisphere)
	topn = topn_addnew(topn, addfake, mode='all')
	coords = SkyCoord(topn.RA, topn.DEC, unit='deg')
	targets = [ap.FixedTarget(name=topn.Name.iloc[i], coord=coords[i]) for i in range(len(coords))] # should be sorted by vdot/sigma
	exptimes = TimeDelta(np.zeros([nobs, len(targets) + 1])) # (nobs, ntargets+1) initialize exptime for each target, while the last column is total night time without target

	# run
	iobs = 0
	next_morning = observer.twilight_morning_astronomical(cur_time, 'next') # the next astronomical twilight morning
	while cur_time < end_time: # while before next year
		print(cur_time.iso)
		iobs = np.sum(cur_time >= start_time_perobs) - 1 # index of observation
		targets_up = observer.target_is_up(cur_time, targets, horizon=altitude_threshold).tolist() # [ntargets] whether each target is up
		if any(targets_up): # some targets is up at this time, accumulate observing time
			itar = targets_up.index(True) # index of first up target
			this_target_set_time = observer.target_set_time(cur_time, targets[itar], 'next', horizon=altitude_threshold) # when this target set
			if itar==0:
				next_time = Time([this_target_set_time, next_morning]).min()
			else:
				better_target_rise_times = observer.target_rise_time(cur_time, targets[:itar], 'next', horizon=altitude_threshold) # when better target rise
				next_time = Time([this_target_set_time, better_target_rise_times, next_morning]).min()
			exptimes[iobs, itar] += (next_time - cur_time) # accumulate observing time
		else: # no target up at this time, accumulate empty time
			rise_times = observer.target_rise_time(cur_time, targets, 'next', horizon=altitude_threshold) # when all targets rise
			next_time = Time([rise_times, next_morning]).min()
			exptimes[iobs, -1] += (next_time - cur_time) # accumulate empty night time
		cur_time = next_time + 10 * u.s # go to next time

		if cur_time >= (next_morning): # if morning comes
			cur_time = observer.twilight_evening_astronomical(cur_time, 'next') # go to next astronomical twilight evening
			next_morning = observer.twilight_morning_astronomical(cur_time, 'next') # the next astronomical twilight morning

	if type(tosave)!=type(None): pkdump(exptimes, tosave)
	return exptimes

def plot_exptimes(exptimes, tosave=None):
	exptimes = np.atleast_2d(exptimes.to(u.hour).value) # (nobs, ntargets+1) array
	exptime_pertarget = np.sum(exptimes, axis=0)
	print('Exptime per observation per target (hours):\n', exptimes)
	print('Total exptime per target (hours):\n', exptime_pertarget)
	print('Total exptime: %.2f hours'%np.sum(exptimes))

	# plotting labels
	xticklabels = (np.arange(exptimes.shape[1] - 1) + 1).tolist() + ['None'] # equivlent to below
	xticklabels = ['%d'%(i + 1) for i in range(len(exptime_pertarget) - 1)] + ['None'] # equivlent to above
	xlabel = 'QSO'
	yticklabels = ['Jan$-$Feb', 'Mar$-$Apr', 'May$-$Jun', 'Jul$-$Aug', 'Sep$-$Oct', 'Nov$-$Dec']

	# exptime per observation
	fig, ax = plt.subplots()
	img = ax.imshow(exptimes)
	cbar = fig.colorbar(img)
	ax.set_xticks(np.arange(exptimes.shape[1]))
	ax.set_yticks(np.arange(exptimes.shape[0]))
	ax.set_xticklabels(xticklabels)
	#ax.set_yticklabels(np.arange(exptimes.shape[0]) + 1)
	ax.set_yticklabels(yticklabels)
	ax.set_xlabel(xlabel)
	ax.set_ylabel('Months')
	cbar.set_label('Exposure time (hour)')
	fig.tight_layout()
	if type(tosave)!=type(None):
		matrix_tosave = tosave + '_matrix.pdf'
		fig.savefig(matrix_tosave); print('Saved: %s'%matrix_tosave)

	# total exptime
	fig, ax = plt.subplots()
	ax.bar(xticklabels, exptime_pertarget)
	ax.set_xlabel(xlabel)
	ax.set_ylabel('Total exposure time (hour)')
	fig.tight_layout()
	if type(tosave)!=type(None):
		tot_tosave = tosave + '_total.pdf'
		fig.savefig(tot_tosave); print('Saved: %s'%tot_tosave)
	#plt.show()

def exptime_tosave(nqso=5, site='lapalma', year=2021, hemisphere='N', addfake=None):
	addfaketxt = '' if addfake is None else '_addtop1'
	base = 'exptimes_top%d_%s_%s_%d%s'%(nqso, hemisphere, site, year, addfaketxt)
	para_tosave = path + 'paras/' + base + '.pkl'
	plot_tosave = path + 'plots/' + base
	return para_tosave, plot_tosave

if __name__ == '__main__':
	# parameters
	hemisphere = 'N'
	year = 2021
	nqso = 5
	addfake = 'top1' # None or number (how many mag brighter) # 'top1' (add new target at top1's brightness) or 'brighter' (add new target 0.5mag brighter than top1)
	load = False

	# define site
	if hemisphere=='N': site = 'lapalma' # Canary Islands
	elif hemisphere=='S': site = 'paranal' # Paranal Observatory, southern

	for nqso, addfake in np.vstack(np.array(np.meshgrid([5, 10], [None, 'top1'])).T):
		print('nqso = %s, addfake = %s'%(nqso, addfake))
		para_tosave, plot_tosave = exptime_tosave(nqso, site, year, hemisphere, addfake=addfake)
		#para_tosave, plot_tosave = None, None # don't save
		if load: exptimes = pkload(para_tosave)
		else: exptimes = main(site, year, nqso, hemisphere, para_tosave, addfake=addfake)
		plot_exptimes(exptimes, tosave=plot_tosave)

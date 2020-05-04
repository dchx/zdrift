'''
line-wise sigma estimation
Given two spectra with different noise added, fit voigt profiles to
the 1st spectrum and derive a template, then correlate the template
with the 2nd spectrum to estimate sigma
'''
from __future__ import print_function
from utils import *
from cosmology import dz2dv
import generate_spec as gs
import voigtforest as vf
from continuum_fit import get_keck_spec
from read_koa import smoothwidth,get_res
import read_koa as rk

def keck_intrinsic_lw(parray, R_keck, smoothwidth=None, dlam=None):
	'''
	return intrinsic line width adjusting for fitting smooth and R_keck
	smoothwidth - (pixel) =2sigma for gaussian, if None, no smooth
	dlam - (AA/pixel) for keck observed lam grid, only used if smoothwidth!=None
	'''
	lam0 = parray[0]
	paras = parray[2:] # line width parameters to be adjusted, shape: (npara, npoints)
	fwhm_keck = lam0 / R_keck
	if smoothwidth!=None:
		fwhm_smooth = smoothwidth/2.*2.355 * dlam #AA
		fwhm_toadj = np.sqrt(fwhm_smooth**2. + fwhm_keck**2.)
	else: fwhm_toadj = fwhm_keck
	least_fv_intrin = 0./3e5*lam0 # make intrinsic fwhm to be at least 5 km/s
	if len(paras)==2: fv_obs = vf.fv(*paras) # should paras == [fl, fg]
	elif len(paras)==1: fv_obs = paras[0] # paras == [b]
	fv_intrin_square = fv_obs**2. - fwhm_toadj**2.
	fv_intrin = np.array([least_fv_intrin[i] if fv_intrin_square[i]<least_fv_intrin[i]**2.\
	                      else np.sqrt(fv_intrin_square[i]) for i in range(len(fv_obs))])
	if len(paras)==2:
		factor = fv_intrin / fv_obs
		paras_adj = paras * factor
		# make fg not zero
		fg_min = 1e-20
		paras_adj[1] = np.array([pval if pval > fg_min else fg_min for pval in paras_adj[1]])
	elif len(paras)==1: paras_adj = fv_intrin
	parray[2:] = paras_adj
	return parray

def keck_template(ind, shiftmode, dz, fitcont_dist, fitcont_deg, fitcont_mode, vfaddline, CSL_cut, fix, res_fixres=2e4, dlam_fixresele=0.0125, smooth=True, verbose=True):
	'''
	modified mk_qso_spec3_dz
	from keck data to template
	smooth - whether smooth during read_koa
	'''
	saveid = saveid_func(ind)
	fitconttxt = '_dist%d'%fitcont_dist + ('_deg%d'%fitcont_deg if fitcont_mode=='poly' else '_cont%s'%fitcont_mode)
	restxt = '_dlam%.3e'%dlam_fixresele if fix=='resele' else '_R%.1e'%res_fixres if fix == 'res' else ''
	smoothtxt = '_smooth%dpix'%rk.smoothwidth if smooth else '_nosmooth'
	vfaddlinetxt = '_fitaddline' if vfaddline else ''
	csltxt = '_CSLcut%.1f'%CSL_cut

	templam_file = path + 'data/keck_gen_spec/keck_lam_%s.pickle'%(saveid + restxt)
	tempflux_file = path + 'data/keck_gen_spec/keck_flux_%s_%s%.3e.pickle'%(saveid + fitconttxt + restxt + smoothtxt, shiftmode, dz)
	if os.path.exists(tempflux_file): # assuming templam_file exists too
		lam = pkload(templam_file, verbose=verbose)
		flux_temp = pkload(tempflux_file, verbose=verbose)
	else:
		# get keck data and fit continuum
		klam, kflux, knoise = get_keck_spec(ind, fitcont_dist, fitcont_deg, fitcont_mode, plot_rest_frame=False, smooth=smooth) # form spec in observed frame

		# set lam grid (not from data), including save lam
		lam, res_ele = gs.lam_grid(np.min(klam), np.max(klam), fix, res=res_fixres, dlam_fixresele=dlam_fixresele, tosave=templam_file)

		# fit voigtforest
		keckvf_file = path + 'paras/voigtforest_bestp_%s.pzip'%(saveid + smoothtxt + fitconttxt + vfaddlinetxt + csltxt) # assume vf.para_set=='fLfG'
		vfresults = vf.fit_forest(klam, kflux, knoise, tosave=keckvf_file, addline=vfaddline, CSLcut=CSL_cut, verbose=verbose)
		vfparray = vf.results2parray(vfresults)

		# adjust parameters by Keck resolution, smooth resolution
		smoothwidth = rk.smoothwidth if smooth else None
		vfparray = keck_intrinsic_lw(vfparray, R_keck=rk.get_res(matched['KOAjobID'][ind]), smoothwidth=smoothwidth, dlam=np.diff(klam).mean())

		# parray2flux, with tosave (save flux)
		flux_temp = gs.parray2flux_dz(vfparray, lam, shiftmode, dz, matched['z'][ind], voigt_is_tau=vf.voigt_is_tau, v1d=vf.v1d)

		# convolve with resolution element
		flux_temp = rk.flux_smooth(flux_temp, res_ele)
		
		# save flux
		pkdump(flux_temp, tempflux_file, verbose=True)
	return lam, flux_temp

def get_template(linelistmode, fixres, res_fixres, dlam_fixresele, shiftmode, dz, genspec_args, keck_args, verbose=True):
	if linelistmode == 'genspec':
		lam, flux_temp = gs.generate_spec(genspec_args.zqso, res=res_fixres, dlam=dlam_fixresele, fix=fixres, dz=dz, rest_frame=False,\
		                 shiftmode=shiftmode, ispec=genspec_args.ispec, verbose=verbose)
	elif linelistmode == 'keck':
		lam, flux_temp = keck_template(keck_args.ind, shiftmode, dz, keck_args.fitcont_dist, keck_args.fitcont_deg, keck_args.fitcont_mode,\
		                 keck_args.vfaddline, keck_args.CSL_cut, fixres, res_fixres, dlam_fixresele, smooth=keck_args.smooth, verbose=verbose)
	return lam, flux_temp
	
def zero_runs(a): # from https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
	'''
	returns n*2 array with chunck bound indexs
	'''
	# Create an array that is 1 where a is 0, and pad each end with an extra 0.
	iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
	absdiff = np.abs(np.diff(iszero))
	# Runs start and end where absdiff is 1.
	ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
	return ranges

def cut_lam_inds(lam, chunklims):
	'''
	return same format as zero_runs()
	chunklims: list of lam values that split lam into chunks
	'''
	lam = np.array(lam)
	ranges = []
	for i in range(len(chunklims)-1): # loop through chunks
		crit = (chunklims[i] <= lam) * (lam <= chunklims[i+1])
		subranges = zero_runs(1 - crit)
		for r in subranges: ranges.append(r)
	ranges = np.array(ranges)
	return ranges

def separate_lines(lam, flux):
	'''
	Chunking by gradient
	Separate flux with individual lines, assuming flux is analytical, with only Voigt profiles
	OUTPUT
	    chunklims - in lam units, list of cutting points
	                if cut lam to not continous, chunklims = [[left1,right1], [left2,right2], ...]
	'''
	chunklims=[]
	if flux[0]>flux[1]: chunklims.append(lam[0])
	for i in range(1,len(lam)-1): # loop through non-edge points
		if (flux[i-1]< flux[i] and flux[i]> flux[i+1])\
		or (flux[i-1]==flux[i] and flux[i]> flux[i+1])\
		or (flux[i-1]< flux[i] and flux[i]==flux[i+1]): chunklims.append(lam[i]) # find where lines crossover
	if flux[-2]<flux[-1]: chunklims.append(lam[-1])
	chunklims = np.array(chunklims)
	return chunklims

def chunk_spec(lam, flux, minlinesize=3, chunk_flux_threshold='gradchk'):
	'''
	divide template spectrum (not added noise) into chunks
	---
	chunk_flux_threshold - if 'gradchk' use gradient method, else
	                       (float) threshold to cut spectrum into chunks 
	chunk_flux_threshold = 0.8, 0.975
	---
	return - chk_boundinds shape(nlines, 2)
	'''
	# ----- divide spec into chunks by flux gradient, 1 line per chk
	if chunk_flux_threshold=='gradchk':
		chunklims = separate_lines(lam, flux)
		chk_boundinds = cut_lam_inds(lam, chunklims)
	# ----- divide spec into chunks by 97.5 percent of flux
	elif type(chunk_flux_threshold)!=str:
		tmp1=np.zeros(len(flux))
		fmax=np.max(flux)
		tmp1[flux > chunk_flux_threshold * fmax] = flux[flux > chunk_flux_threshold * fmax]
		chk_boundinds = zero_runs(tmp1)
	else: return None # e.g. when chunk_flux_threshold=='vfregchk'
	# ----- filter by minlinesize
	linesize = chk_boundinds[:,1] - chk_boundinds[:,0]
	jlines = [k for k in range(len(linesize)) if linesize[k] >= minlinesize] # index of chunks of qualified chunks (len>=minlinesize)
	chk_boundinds = chk_boundinds[jlines]
	return chk_boundinds

def dvfits2dv(dvfits_masked):
	'''
	Estimate overall dv from dvfits array (can be masked)
	dvfits - size(nchunk, ntrial)
	'''
	dvfits_pertrial = np.ma.median(dvfits_masked, axis=0)
	dvmean = np.ma.mean(dvfits_pertrial)
	dvstd = np.ma.std(dvfits_pertrial, ddof=1)
	return dvmean, dvstd

def dvfits2sigma(dvfits, relaerr_all=None, relaerr_thrshld=0.75, dvstd_thrshld=20., bs_time=1000, bootstrap=True):
	'''
	Estimate sigma from dvfits array
	-------
	dvfits - size(nchunk, ntrial)
	relaerr_thrshld - (default 0.75) maximum relative error
	dvstd_thrshld - (default 20, cm/s) maximum std in dv over ntrials
	bs_time - (default 1000) number of bootstrapping resamples
	          if bs_time==1, do not do bootstrapping
	'''
	dvfits = np.ma.masked_invalid(dvfits) # mask nan and inf values
	plot = 0
	if plot:
		plt.figure()
		#plt.hist(dvfits.flatten(), bins=50)
		plt.hist(dvfits.flatten(), bins=np.arange(-1, 1.1, 0.1))
		plt.xlabel('estimated dv (cm/s)')

	# filter dvfits
	# mask dv result data by relaerr_thrshld
	if np.any(relaerr_all): dvfits_masked = np.ma.masked_where(relaerr_all > relaerr_thrshld, dvfits)
	else: dvfits_masked = dvfits
	print('Drop large relative error dv, estimated dv: %.2f +/- %.2e cm/s'%dvfits2dv(dvfits_masked))
	
	# compute std and dropout high std chunks
	vstd_perchk = np.ma.std(dvfits_masked, axis=1, ddof=1) # size:nchk, std over ntrials
	print('vstd_perchk median:', np.ma.median(vstd_perchk))
	ichk_tokeep = vstd_perchk < dvstd_thrshld
	dvfits_masked = dvfits_masked[ichk_tokeep]
	print('Drop high std chunks, estimated dv: %.2f +/- %.2e cm/s'%dvfits2dv(dvfits_masked))
	
	# bootstrapping to compute sigma with error
	sigvs = [] # size(bs_time), sigma_vs of each bootstrapping sample
	for i in range(bs_time):
		if bs_time==1: inds = np.arange(dvfits.shape[1])
		else: inds = np.random.randint(0,dvfits.shape[1],dvfits.shape[1]) # dvfit.shape[1] == ntrial
		vstd_perchk = np.ma.std(dvfits_masked[:,inds], axis=1, ddof=1) # size(nchunk), std over ntrials
		vstd_perchk = vstd_perchk[vstd_perchk > 0] # filter not to divide by zero
		sigvs.append(np.sqrt(1./np.ma.sum(1./(vstd_perchk**2.))))
	sigv = np.mean(sigvs)
	sigv_err = np.std(sigvs, ddof=1)
	print('sigma: %.2e +/- %.2e cm/s'%(sigv, sigv_err))
	return sigv

if __name__ == '__main__':
	# ---------- set up parameters ----------
	linelistmode = 'keck' # 'genspec' or 'keck'
	### applying for differnt linelistmodes
	class genspec_args:
		zqso = 3.
		ispec = None
	class keck_args:
		ind = 7 # index on keck quasar list
		fitcont_dist = 100
		fitcont_deg = 10
		fitcont_mode = 'poly' # poly, linear or cubic
		vfaddline = False # voigtforest for keck data
		CSL_cut = 1.5 # voigtforest for keck data
		smooth = True # whether smooth during read_koa
	
	### applying to all linelistmodes
	fixres = 'resele' # 'res' or 'resele', whether to fix R or resolution element size (vs lam)
	dlam_fixresele = 0.0125 # AA per pixel, useful only if fixres=='resele'
	res_fixres = 2e4 # R, useful only if fexres=='res'
	nphot2 = 1.69e8 # npoht for epoch 2 for add_shot_noise
	nphot1 = 1**2. * nphot2 # nphot for epoch 1 for add_shot_noise
	ntemplate = 1 # number of templates to generate; if 1: only template 0 from line list, if 2: another template from voigtfit to epoch 1 spectra
	ntrial = 100
	chunk_flux_threshold = 'vfregchk' # 'gradchk' for chunking by gradient, 'vfregchk' for chunking by voigtforest regions, 0.8, 0.975
	###### voigtforest fitting for epoch 1 spec
	CSL_cut = 1.5 # for region detection in voigt fitting, default 1.5
	vfaddline = True # whether to try add lines when fitting voigtforest
	###### shift parameters for epoch 2 and test spectra
	shiftmode = 'dv' # 'dz' / 'dv' / 'dl' (+ 'lw' for adjusting line width too)
	if 'dz' in shiftmode:
		real_dx = 0. # real dz for the second epoch
		delta_dx = 5e-11 # 2e-10 dz testing step
		testrange = 5e-8 # 5e-8 determines the dz test range [real_dx +/- test_range]
	elif 'dv' in shiftmode:
		real_dx = 21.3 # (cm/s) real dv for the second epoch
		delta_dx = 0.4  # 0.2 0.4 (cm/s) dv testing step
		testrange = 390. # 40 200 (cm/s) determines the dv test range [real_dx +/- test_range]
	###### settings not shown in runid
	minlinesize = 3 # pixel, min size for each chunk
	######### dvfits to sigma parameters
	relaerr_thrshld = 0.75 #0.75
	dvstd_thrshld = +np.inf # (cm/s)
	bs_time = 1000
	# ---------- set up parameters end ----------
	
	# derived parameters
	dxmin = real_dx - testrange
	dxmax = real_dx + testrange + delta_dx # add delta_dx for range setup
	dxtests = np.arange(dxmin, dxmax, delta_dx)
	if linelistmode == 'genspec':
		runid = 'zqso%.1f'%genspec_args.zqso
		runid += '_spec%d'%genspec_args.ispec if genspec_args.ispec != None else '' # ispectxt
		runid += '_LiskeDist' if gs.LiskeDist else '' # liskedisttxt
		zqso = genspec_args.zqso
	elif linelistmode == 'keck':
		zqso = matched['z'][keck_args.ind]
		saveid = saveid_func(keck_args.ind)
		runid = saveid
	if ntemplate==1 and chunk_flux_threshold=='vfregchk':
		print("Cannot do vfregchk: not generating 2nd template.\n    Switching to gradchk.")
		chunk_flux_threshold = 'gradchk'
	
	# runid used in dvfits_file
	runid = linelistmode + '_' + runid
	runid += '_vf%s'%vf.para_set # voigtforest use blgNHI (generate_spec.voigt1d) or fLfG (astropy.Voigt1D)
	runid += '_dlam%.3e'%dlam_fixresele if fixres=='resele' else '_R%.1e'%res_fixres if fixres == 'res' else '' # restxt
	runid += '_Nphot%.2e'%nphot2 # nphot for epoch 2
	runid += '_epoch1Nphot%.2e'%nphot1 if nphot1!=nphot2 else '' # nphot for epoch 1
	runid += '_CSLcut%.1f'%CSL_cut # csltxt
	runid += '_vfaddline' if vfaddline else '_vfnoaddline'
	runid += '_shift%s'%shiftmode
	runid += '_1template' if ntemplate==1 else '' if ntemplate==2 else ''
	specid = runid # used in vf_tosave and testspec_file
	runid += '_addnoise' if (ntrial == 1) else '_addnoise%dmean'%ntrial # addnoisetxt
	runid += '_%s'%chunk_flux_threshold if type(chunk_flux_threshold)==str else '_chk%.3fflux'%chunk_flux_threshold # chunking method
	runid += '_real%s%.3e'%(shiftmode, real_dx)
	runid += '_testrange%.1ed%.1e'%(testrange, delta_dx) # testrangetxt
	
	# ---------- start run ----------
	# get dvfits
	dvfits_file = path + 'paras/dvfits_%s.pickle'%(runid)
	if os.path.exists(dvfits_file):
		dvfits = pkload(dvfits_file)
	else:
		print('Running',runid)
		# ----- get original spectral templates for two epochs -> lam, flux1_temp, flux2_temp -----
		# epoch 1 template
		lam, flux1_temp = get_template(linelistmode, fixres, res_fixres, dlam_fixresele, shiftmode, 0., genspec_args, keck_args)
		# epoch 2 template
		if real_dx == 0: flux2_temp = flux1_temp
		else: _, flux2_temp = get_template(linelistmode, fixres, res_fixres, dlam_fixresele, shiftmode, real_dx, genspec_args, keck_args)
		# chunk epoch 1 template
		chk_boundinds = chunk_spec(lam, flux1_temp, minlinesize, chunk_flux_threshold) # chunk by epoch 1 template

		# ----- get voigtforest parameters for 2nd template -----
		if ntemplate==2:
			vf_tosave = path + 'paras/voigtforest_bestp_simed_%s.pzip'%(specid)
			# get epoch 1 spectrum by adding noise to original template
			flux1, flux1_err = add_shot_noise(flux1_temp, nphot1, return_error=True)
			# fit voigtforest to epoch 1 spectrum, get result parray, chunk by voigtforest regions
			voigtresult, reg_boundinds = vf.fit_forest(lam, flux1, flux1_err, tosave=vf_tosave, addline=vfaddline, CSLcut=CSL_cut, chkind=True)
			vfparray = vf.results2parray(voigtresult)
			# chunk by voigtforest regions
			if chunk_flux_threshold=='vfregchk': chk_boundinds = reg_boundinds
			# testing voigtforest fit
			plot = 0
			if plot:
				plt.figure()
				fluxtest_temp = gs.parray2flux(vfparray, lam, voigt_is_tau=vf.voigt_is_tau, v1d=vf.v1d)
				plt.plot(lam, flux1_temp, 'k', lw=0.5) # plot epoch 1 template
				plt.plot(lam, fluxtest_temp, 'r', lw=0.5) # plot voigtfit of epoch 1 spectrum
				plt.show()

		# ----- do correlation, find best dz -----
		dvfits = np.zeros([len(chk_boundinds),ntrial]) # the best fit redshift for each of 100 trials for each chunk
		t0 = time.time()
		ntrial = 100
		for i in range(ntrial): # try 100 times
			t1 = time.time()
			# add noise to epoch 2 original template
			flux2 = add_shot_noise(flux2_temp, nphot2, return_error=False)
			# test different dz
			corrs = np.zeros([len(chk_boundinds),len(dxtests)])
			for idz in range(len(dxtests)): # try each test dz
				# form 2nd template spectrum
				testspec_file = path + 'data/test_spec/testspec_%s_%s%.3e.pickle'%(specid, shiftmode, dxtests[idz])
				if ntemplate==1: # use original template
					_, fluxtest_temp = get_template(linelistmode, fixres, res_fixres, dlam_fixresele, shiftmode, dxtests[idz], genspec_args, keck_args, verbose=False)
					fluxtest_temp = add_shot_noise(fluxtest_temp, nphot1)
				elif ntemplate==2: # shift vfparray to form 2nd template spectrum
					fluxtest_temp = gs.parray2flux_dz(vfparray, lam, shiftmode, dxtests[idz], zqso, voigt_is_tau=vf.voigt_is_tau, v1d=vf.v1d, tosave=testspec_file)
		
				# correlation between 2nd template spectrum (fluxtest_temp) and epoch 2 spectrum (flux2)
				for ichk in range(len(chk_boundinds)): # loop through chunks
					il = chk_boundinds[ichk, 0]
					ir = chk_boundinds[ichk, 1]
					corrs[ichk, idz] = np.corrcoef(fluxtest_temp[il:ir], flux2[il:ir])[0,1]
			# testing
			#print('%d chunks'%len(chk_boundinds))
			#for ichk in range(len(chk_boundinds)):
			#	dxfitind = np.argmax(corrs[ichk,:])
			#	dxfit = dxtests[dxfitind]
			#	testspec_file = path + 'data/test_spec/testspec_%s_%s%.3e.pickle'%(specid, shiftmode, dxfit)
			#	fluxfit = gs.parray2flux_dz(vfparray, lam, shiftmode, dxfit, zqso, voigt_is_tau=vf.voigt_is_tau, v1d=vf.v1d, tosave=testspec_file)
			#	testspec_file = path + 'data/test_spec/testspec_%s_%s%.3e.pickle'%(specid, shiftmode, real_dx)
			#	fluxreal = gs.parray2flux_dz(vfparray, lam, shiftmode, real_dx, zqso, voigt_is_tau=vf.voigt_is_tau, v1d=vf.v1d, tosave=testspec_file)
			#	il = chk_boundinds[ichk, 0]
			#	ir = chk_boundinds[ichk, 1]
			#	diffreal = (flux2[il:ir] - fluxreal[il:ir])
			#	difffit =  (flux2[il:ir] - fluxfit[il:ir])
			#	corrreal = np.corrcoef(fluxreal[il:ir], flux2[il:ir])[0,1]
			#	corrfit = np.corrcoef(fluxfit[il:ir], flux2[il:ir])[0,1]
			#	#corrreal = corrs[ichk, np.argmin(np.abs(dxtests-real_dx))]
			#	#corrfit = corrs[ichk, dxfitind]
			#	'''
			#	plt.plot(lam[il:ir], flux2[il:ir] - fluxreal[il:ir], 'k-x', lw=0.5)
			#	plt.plot(lam[il:ir], flux2[il:ir] - fluxfit[il:ir], 'r--+', lw=0.5)
			#	plt.plot(lam[il:ir], diffreal, 'k-x', lw=0.5)
			#	plt.plot(lam[il:ir], difffit, 'r--+', lw=0.5)
			#	'''
			#	for idz in range(len(dxtests)):
			#		testspec_file = path + 'data/test_spec/testspec_%s_%s%.3e.pickle'%(specid, shiftmode, dxtests[idz])
			#		fluxtest_temp = gs.parray2flux_dz(vfparray, lam, shiftmode, dxtests[idz], zqso, voigt_is_tau=vf.voigt_is_tau, v1d=vf.v1d, tosave=testspec_file)
			#		plt.plot(lam[il:ir], fluxtest_temp[il:ir], 'r--', lw=0.5)
			#	plt.plot(lam[il:ir], flux2[il:ir], 'k-', lw=0.5)
			#	#plt.plot(lam[il:ir], fluxreal[il:ir], 'r--+', lw=0.5)
			#		
			#	#print('chunk',ichk,'real dx',real_dx,'corr',corrreal,'fit dx',dxfit,'corr',corrfit, 'real-fit', corrreal-corrfit)
			#	#print('chunk',ichk,'dxfit',dxfit,'diffreal-difffit',np.mean(diffreal - difffit))
			#	'''
			#	plt.plot(dxtests, corrs[ichk,:], 'b-x')
			#	plt.axvline(dxfit)
			#	'''
			#	plt.title('ichk:%d dxfit: %.3f'%(ichk,dxtests[np.argmax(corrs[ichk,:])]))
			#	plt.show()
			# ^testing
			dvfits[:, i] = dxtests[np.argmax(corrs, axis=1)] # size:nchk, dz with max corrs for each chunk
			if 'dz' in shiftmode: dvfits[:, i] = dz2dv(dvfits[:, i], zqso)
			print('Trial %02d, cost %.2f min, total %.2f min, dvfit median: %.2f cm/s'%(i, (time.time()-t1)/60., (time.time()-t0)/60., np.median(dvfits[:, i])))
		# save dvfits
		pkdump(dvfits, dvfits_file)
	
	# ----- calculate sigma -----
	if 'dz' in shiftmode: real_dx = dz2dv(real_dx, zqso) # convert real_dx to real_dv
	print('real dv: %.2f cm/s'%real_dx)
	relaerr_all = np.abs((real_dx - dvfits) / testrange)
	dvfits2sigma(dvfits, relaerr_all, relaerr_thrshld, dvstd_thrshld, bs_time)

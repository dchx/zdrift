from utils import *
from itertools import groupby
from operator import itemgetter
from scipy.signal import gaussian

#smoothwidth=5 # before 10/28/2019: smoothwidth=50; 10/28: 10; 10/29: 5
smoothwidth=2 # (pixel) =2sigma for gaussian

def flux_smooth(flux,width):
	# smooth flux with 2-sigma width (in pixels)
	w = np.ones(int(round(width))) # flat kernel
	w = gaussian(width*4., width/2.) # 4-sigma range
	return np.convolve(w/w.sum(),flux,mode='same')

def lamflux_from_table(table,z_plot_rest_frame=0.,lya_tocut=None,smooth=True):
	# return lam, flux, flux_err, disp from koa data table, after spectrum prior filtering 
	table=table[np.argsort(table[:,4]),:] # sort based on wavelength
	lam=table[:,4]
	flux=table[:,5]
	flux_err=table[:,6]
	arc_lamp = table[:,10]
	disp=table[:,12]
	exptime=table[:,13]
	out=[lam,flux,flux_err,disp,exptime,arc_lamp]
	# filter spectrum with prior
	## flux error >= 0
	crit=out[2]>=0#flux_err>=0
	for i in range(len(out)): out[i]=out[i][crit]
	## lambda >=0
	crit=out[0]>=0#lam>=0
	for i in range(len(out)): out[i]=out[i][crit]
	# if in rest frame
	out[0]=rest_frame(out[0],z_plot_rest_frame) #lam
	# cut lyaforest
	if lya_tocut!=None:
		crit=out[0]<lya_tocut
		for i in range(len(out)): out[i]=out[i][crit]
	# smoothing
	if smooth and len(out[0])!=0:
		out[1]=flux_smooth(out[1],smoothwidth) #flux
		crit=(np.arange(len(out[0]))>=smoothwidth/2.)*(np.arange(len(out[0]))<len(out[0])-smoothwidth/2.) # cut off edge pixels
		for i in range(len(out)): out[i]=out[i][crit]
	return tuple(out)

def sort_lamlim(specs):
	'''
	sort specs by lam_min of each spec
	'''
	lamlims=np.array([[np.min(spec[0]),np.max(spec[0])] for spec in specs]) #[[lam_min,lam_max],[lam_min,lam_max],...]
	argsortLamMin=np.argsort(lamlims.T[0],axis=0) # sort by lam_min
	sortedSpecs=np.array(specs)[argsortLamMin] # sorted by lam_min
	sortedLamlims=lamlims[argsortLamMin]
	return sortedSpecs,sortedLamlims

def cut_repeat_wave(specs):
	'''
	cut repeated wavelengths in different scales
	INPUT
	    specs: [(lam,flux,flux_err, ...),(lam,flux,flux_err, ...), ... ]
	OUTPUT
	    cuttedSpecs: [(lam,flux,flux_err, ...),(lam,flux,flux_err, ...), ... ]
	'''
	sortedSpecs,sortedLamlims=sort_lamlim(specs)
	# rearange sortedLamlims, cut overlaps
	for ileft in range(len(sortedLamlims)-1):
		for iright in ileft+np.arange(1,4): # [1,2,3]
			if iright<len(sortedLamlims) and sortedLamlims[ileft][1]>=sortedLamlims[iright][0]: # make the cut
				leftedge=sortedLamlims[iright][0]
				sortedLamlims[iright][0]=sortedLamlims[ileft][1]
				sortedLamlims[ileft][1]=leftedge
	# rearange sortedSpecs, cut over sortedLamlims
	cuttedSpecs=[]
	for ispec in range(len(sortedSpecs)):
		cutbool=(sortedSpecs[ispec][0]>=sortedLamlims[ispec][0])*(sortedSpecs[ispec][0]<=sortedLamlims[ispec][1]) #cutting criterita
		cuttedSpec=np.vstack(sortedSpecs[ispec]).T[cutbool]
		if len(cuttedSpec)!=0: cuttedSpecs.append(tuple(cuttedSpec.T))
	return cuttedSpecs

def cut_wave_by_snr(specs):
	'''
	cut wavelength by flux/flux_err (SNR) and make lam connectable.
	1.first find the highest SNR of each spec, if not higher than anyone, drop it.
	2.then go through left and right of highest point, until meet higher from other spec, cut.
	3.recollect all cutted spec, stitch nearby ones.
	INPUT
	    specs: [(lam,flux,flux_err, ...),(lam,flux,flux_err, ...), ... ]
	OUTPUT
	    cuttedSpecs: [(lam,flux,flux_err, ...),(lam,flux,flux_err, ...), ... ]
	'''
	sortedSpecs,sortedLamlims=sort_lamlim(specs)
	startsHere=[0] # new unconnected spec starts here
	for ileft in range(len(sortedSpecs)-1): # excluding the last one
		foundConnected=False
		for iright in ileft+np.arange(1,4): # ileft+[1,2,3]
			if iright<len(sortedSpecs) and sortedLamlims[ileft][1]>=sortedLamlims[iright][0]:
				foundConnected=True
				# get overlap specs
				lamOverlapLim=[sortedLamlims[iright][0],min([sortedLamlims[ileft][1],sortedLamlims[iright][1]])]
				overlapLeftbool=(sortedSpecs[ileft][0]>=lamOverlapLim[0])*(sortedSpecs[ileft][0]<=lamOverlapLim[1])
				overlapLeft=np.vstack(sortedSpecs[ileft]).T[overlapLeftbool].T
				overlapRightbool=(sortedSpecs[iright][0]>=lamOverlapLim[0])*(sortedSpecs[iright][0]<=lamOverlapLim[1])
				overlapRight=np.vstack(sortedSpecs[iright]).T[overlapRightbool].T
				# interpert overlapRight SNR to overlapRight lam
				overlapRight_intpSNR=np.interp(overlapLeft[0],overlapRight[0],overlapRight[1]/overlapRight[2]) 
				# index of higher SNR
				leftHigher=(overlapLeft[1]/overlapLeft[2])>overlapRight_intpSNR
				if leftHigher.prod(): midlam=overlapLeft[0][-1] # if all left higher than right, keep all left
				elif (~leftHigher).prod(): midlam=overlapLeft[0][0] # if all right higher than left, keep all right
				elif (not leftHigher[0]) or leftHigher[-1]: midlam=overlapLeft[0][-1] #if Not left seg left SNR higher and right seg right SNR higher: keep all left
				else:
					edgelam=[] #[leftedge, rightedge]
					for isleft in [True,False]:
						if isleft: ind=np.where(leftHigher)[0]
						else: ind=np.where(~leftHigher)[0]
						indgroups=[]
						#for k,g in groupby(enumerate(ind),lambda(i,x):(i-x)): indgroups.append(map(itemgetter(1),g))
						for k,g in groupby(enumerate(ind),lambda x: x[0]-x[1]): indgroups.append(list(map(itemgetter(1),g)))
						indgroups=[item for item in indgroups if len(item)>=10] # filter len(continus inds)>=10
						if len(indgroups)==0:
							if isleft: edgeind=0
							else: edgeind=1
						else:
							if isleft: edgeind=indgroups[-1][-1]
							else: edgeind=indgroups[0][0]
						edgelam.append(overlapLeft[0][edgeind])
					midlam=np.mean(edgelam) # lam to cut left and right
				sortedSpecs[ileft]=tuple(np.vstack(sortedSpecs[ileft]).T[sortedSpecs[ileft][0]<=midlam].T)
				sortedSpecs[iright]=tuple(np.vstack(sortedSpecs[iright]).T[sortedSpecs[iright][0]>=midlam].T)
		if not foundConnected: startsHere.append(ileft+1) # mark that spectrum is not connected here

	#connect specs that are connected
	connecSpecs=[]
	for iend in range(len(startsHere)):
		# get connected specs
		if len(startsHere)==1: subspecs=sortedSpecs
		else:
			if iend==0: subspecs=sortedSpecs[:startsHere[iend+1]]
			elif iend==len(startsHere)-1: subspecs=sortedSpecs[startsHere[iend]:]
			else: subspecs=sortedSpecs[startsHere[iend]:startsHere[iend+1]]
		subspec=tuple([np.hstack([subspecs[ispec][iitem] for ispec in range(len(subspecs))]) for iitem in range(len(subspecs[0]))]) #(array(lam1,lam2,...),array(flux1,flux2,...),...)
		connecSpecs.append(subspec)
	#sortedSpecs=connecSpecs

	sortedSpecs=[spec for spec in sortedSpecs if len(spec[0])>1] # filter 0-length spec
	return sortedSpecs

def appendtable(files, stackchan=0, z_plot_rest_frame=0.,lya_tocut=None, smooth=True):
	# OUTPUT
	#   (lam,flux,flux_err, disp,exptime) if stachan else [(lam,flux,flux_err, disp,exptime),(lam,flux,flux_err, disp,exptime), ... ]
	if stackchan: table=np.zeros(np.loadtxt(files[0],skiprows=1).shape[1])
	else: tables=[]
	for f in files:
		thedata=np.loadtxt(f,skiprows=1)
		# get exptime
		hdrfile=os.path.abspath(os.path.dirname(f)+'/../hdr/'+os.path.basename(f).replace('_flux.tbl','_hdr.txt'))
		hdrfits=f.replace('/tbl/','/binaryfits/').replace('/flux/','/hdr/').replace('_flux.tbl','_hdr.fits')
		exptime=fits.getheader(hdrfits)['EXPTIME'] # exp time value
		exptime=exptime*np.ones(len(thedata)) # exp time vector to match data num of rows
		# add colunm disp (dispersion AA/pixel) to thedata
		arcidsfile=os.path.abspath(os.path.dirname(f)+'/../arcids/'+os.path.basename(f).replace('_flux','_arcids'))
		col_cen,disp_arcids=np.loadtxt(arcidsfile,skiprows=1)[:,[0,11]].T # column center, dispersion (Angstrom/pixel)
		col=thedata[:,0] # colunm used to interperate disp
		disp=np.polyval(np.polyfit(col_cen,disp_arcids,3),col) # fit polynomial to disp-col_cen relation and extrapolate -> disp
		thedata=np.vstack([thedata.T,disp]).T # add disp colunm
		thedata=np.vstack([thedata.T,exptime]).T # add exptime colunm
		if len(thedata[0])!=0.: 
			if stackchan: table=np.vstack((table,thedata)) # stack different wavelength channels
			else: tables.append(thedata)
	if stackchan: 
		table=np.delete(table,0,axis=0)
		return lamflux_from_table(table,z_plot_rest_frame=z_plot_rest_frame,lya_tocut=lya_tocut,smooth=smooth) # (lam,flux,flux_err,disp,exptime)
	else:
		nostack_out=[]
		for stable in tables:
			out=lamflux_from_table(stable,z_plot_rest_frame=z_plot_rest_frame,lya_tocut=lya_tocut,smooth=smooth)
			if len(out[0]>1): #sometimes there's empty tables, e.g.num==56,koajobid=23057
				nostack_out.append(out)
		#return cut_wave_by_snr(nostack_out) # [(lam,flux,flux_err,disp,exptime),(lam,flux,flux_err, disp,exptime), ... ]
		return nostack_out # [(lam,flux,flux_err,disp,exptime),(lam,flux,flux_err, disp,exptime), ... ]

def multiargmax(thelist):
	thelist=np.array(thelist)
	themax=np.max(thelist)
	return np.where(thelist==themax)[0]

def koa_filelist(koajobid,type='tbl'):
	toglob = path+'data/Keck/KOA_%d/HIRES/extracted/tbl/ccd*/flux/*.tbl'%koajobid
	files = np.array(glob.glob(toglob))
	# observation ids
	ids = [] # like 20051028.26873_3_03, for every file
	koaid_ccds = [] # like 20051028.26873_3, for every file
	koaids = [] # like 20051028.26873, unique
	for f in files:
		thisid=os.path.basename(f)[3:22] # like 20051028.26873_3_03
		ids.append(thisid)
		koaid_ccds.append(thisid[:-3]) # like 20051028.26873_3
	koaid_ccds = np.array(koaid_ccds)
	# group by same koaid_ccds
	koaid_ccds_unq = np.unique(koaid_ccds) # unique koaid_ccds
	filegroups = [] # files grouped by same koaid_ccds
	for koaid_ccd in koaid_ccds_unq:
		filegroups.append(files[koaid_ccds==koaid_ccd])
		koaids.append(koaid_ccd[:-2])
	koaids = np.unique(koaids) # array([koaid, koaid, ...]) unique
	if type=='tbl': return files # return file path list array([file, file, ...])
	elif type=='id': return ids # return file id like 20051028.26873_3_03 [id, id, ...]
	elif type=='group': return filegroups # [array([file, file, ...]), array(...), ...], grouped by same koaid_ccds
	elif type=='groupid': return koaid_ccds_unq # array([koaid_ccd, koaid_ccd, ...]) unique
	elif type=='oneobs' or type=='oneobsids': # pick one observation with most ccds and most total file size
		# deal with number of ccds
		numccd = [] # [num, num, ...] number of ccds for each koaid
		koaid_ccds_unq_grouped=[] # [array([koaid_ccd, koaid_ccd, ...]), array(...), ...] grouped by same koaid
		for kid in koaids:
			thiskid_ccds = koaid_ccds_unq[np.array([koaid_ccd.startswith(kid) for koaid_ccd in koaid_ccds_unq])]
			koaid_ccds_unq_grouped.append(thiskid_ccds)
			numccd.append(len(thiskid_ccds))
		ikoaids_maxccd = multiargmax(numccd) # all index of koaids with max number of ccds
		koaid_ccds_unq_maxccd = [] # [[koaid_ccd, koaid_ccd, ...], [...], ...] all koaid_ccds grouped by same koaid with max number of ccds
		for ikid in ikoaids_maxccd: koaid_ccds_unq_maxccd.append(koaid_ccds_unq_grouped[ikid])
		# deal with file sizes
		diffkid_filegroups=[] # [[[file, file, ...], [...], ...], [[...], [...], ...], ...] grouped by koaid, subgrouped by ccd
		diffkid_filesizes=[] # [size, size, ...] for each koaid
		for koaid_ccds_thiskid in koaid_ccds_unq_maxccd: # list of koaid_ccd for one koaid
			thiskid_files=[] # [[file, file, ...], [file, file, ...], ...] grouped by ccd
			thiskid_filesize = 0. # total file size for all ccds
			for koaid_ccd_thiskid_thisccd in koaid_ccds_thiskid:
				thiskid_thisccd_files = files[koaid_ccds==koaid_ccd_thiskid_thisccd]
				thiskid_files.append(thiskid_thisccd_files)
				thiskid_filesize += np.sum([os.stat(thisfile).st_size for thisfile in thiskid_thisccd_files])
			diffkid_filegroups.append(thiskid_files)
			diffkid_filesizes.append(thiskid_filesize)
		imaxfilesize = np.argmax(diffkid_filesizes) # index of (koaids with max ccdnumber) with largest file size
		if type=='oneobs': return diffkid_filegroups[imaxfilesize] # [array(path, path, ...), array(), ...] grouped by ccd
		elif type=='oneobsids':return koaid_ccds_unq_maxccd[imaxfilesize] # [koaid_ccd, koaid_ccd, ...] with same koaid (selected)

def read_koa_jobid(koajobid,stackchan=0,z_plot_rest_frame=0.,lya_tocut=None,smooth=True):
	'''
	input int/str koajobid, return lam,flux,flux_err,disp
	stackchan: whether to stack different wavelength channels
	   0: not stack any channel, only show one observation (koaid) (with most ccds and file size)
	   1: stack all channels
	   2: stack same ccd and koaid 
	   3: stack same ccd and koaid, only show one observation (koaid) (with most ccds and file size)
	'''
	if stackchan==2: out=koa_filelist(koajobid,type='group')
	elif stackchan==3 or stackchan==0: out=koa_filelist(koajobid,type='oneobs')
	else: out=koa_filelist(koajobid)
	if len(out)==0: print(koajobid, ': table file not found'); return
	files_filegroups = out
	if stackchan==2 or stackchan==3:
		filegroups=files_filegroups
		tablegroups=[]
		for filegroup in filegroups:
			tablegroup=appendtable(filegroup, stackchan=stackchan, z_plot_rest_frame=z_plot_rest_frame,lya_tocut=lya_tocut,smooth=smooth)
			if len(tablegroup)!=0: 
				if len(tablegroup[0])!=0: tablegroups.append(tablegroup)
		return tablegroups # [(lam,flux,flux_err,disp,...),(lam,flux,flux_err,disp,...), ... ]
	else: 
		if stackchan==0: files=np.hstack(files_filegroups) # don't group by ccd. just plot them all
		else: files=files_filegroups
		return appendtable(files, stackchan=stackchan, z_plot_rest_frame=z_plot_rest_frame,lya_tocut=lya_tocut,smooth=smooth) # list or tuple

def make_spec_plot(lam,flux,ax):
	lines=ax.plot(lam,flux,lw=0.5)
	return lines

def plot_koa_spec(thedata,ax,stackchan=0):
	if stackchan==1: 
		lam=thedata[0]
		flux=thedata[1]
		lines=make_spec_plot(lam,flux,ax) # return list of one line
	else:
		lines=[]
		for data in thedata:
			lam=data[0]
			flux=data[1]
			lines.append(make_spec_plot(lam,flux,ax)[0])
	return lines

def get_res(koajobid):
	'''
	return the resolution with maximum number of fitsfiles
	and the list of koaids for that resolution
	'''
	koaid = koa_filelist(koajobid,type='oneobsids')[0][:-2]
	fitsfile = path+'data/Keck/KOA_%d/HIRES/raw/sci/HI.%s.fits'%(koajobid,koaid)
	head = fits.getheader(fitsfile)
	res = float(head['SPECRES'])
	return res

def main(stackchan=0,plot_rest_frame=True,plot_lya_forest=True):
	# z_plot_rest_frame: redshift if plot in rest frame, else zero
	
	for i in itest[[0]]:
		z_plot_rest_frame,lya_toplot=rest_fram_pars(matched['z'][i],plot_rest_frame)
		if plot_lya_forest: lya_tocut=lya_toplot
		else: lya_tocut=None
		fig,ax=plt.subplots(figsize=(12,8))
		thedata=read_koa_jobid(matched['KOAjobID'][i],stackchan=stackchan,z_plot_rest_frame=z_plot_rest_frame,lya_tocut=lya_tocut,smooth=False)
		koa_lines=plot_koa_spec(thedata,ax,stackchan=stackchan)
		#tosave=path+'plots/Keck/KOA_'+str(koajobid)+'_spec.pdf'
		#fig.savefig(tosave);print('Saved:',tosave)
		plt.show()
		#plt.close(fig)

if __name__=='__main__': main()

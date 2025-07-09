import time,pickle,gzip,os
import numpy as np
import lmfit as lf
import matplotlib.pyplot as plt
import astropy.constants as c
import vamp
from astropy.modeling.models import Voigt1D
import generate_spec as gs
from utils import *
from spec_utils import *
import spec_utils as su

# settings
voigt_is_tau = 1 # whether to treat sum(Voigt1Ds) as tau and flux=exp(-tau)
para_set = 'blgNHI' # 'blgNHI' (used in generate_spec.voigt1d) or 'fLfG' (used in astropy.Voigt1D)
# derived
if para_set=='blgNHI': voigt_is_tau = True

# spectrum shift transformation
def dvel2dlam(dvel, lam0, unit='km/s'): return lam0 * dvel / c.c.to(unit).value
def dlam2dvel(dlam, lam0, unit='km/s'): return c.c.to(unit).value * dlam / lam0
# line width transformation
def fwhm2b(fwhm): return fwhm / 2. / np.sqrt(np.log(2.))
def b2fwhm(b): return 2.*np.sqrt(np.log(2.)) * b

def fv(fl, fg):
    '''
    compute fwhm for Voigt profile (fv) from fwhms of Lorentz (fl) and Gaussian (fg)
    '''
    return 0.5346*fl+np.sqrt(0.2166*fl**2.+fg**2.)

def lam_space(lam_start, lam_end, R=35800., pix_per_R=2.):
    '''
    Generate a sequence of wavelengths
    '''
    lam = lam_start
    lams = []
    while lam <= lam_end:
        lams.append(lam)
        dlam = lam / R
        pixscale = dlam / pix_per_R 
        lam += pixscale
    return np.array(lams)

typical_lw = 20. # km/s, typical fwhm line width
typical_lw = 50. # km/s, typical fwhm line width

def generate_test_spec(lam_start=1000., lam_end=1010., nlines=10, R=35800., pix_per_R=2., Nphot=1e3, linewidth=typical_lw):
    '''
    linewidth - (km/s) one value
    '''
    lam = lam_space(lam_start, lam_end, R, pix_per_R)
    lam0s = np.random.uniform(min(lam),max(lam),nlines)
    lam0s_mid = np.sort(lam0s)[int(round(nlines/2))]
    lam0s = np.r_[lam0s, lam0s_mid + 5*dvel2dlam(linewidth, lam0s_mid)]
    lam0s = np.r_[lam0s, lam0s_mid - 5*dvel2dlam(linewidth, lam0s_mid)]
    flux_noerr = np.zeros(len(lam))
    for lam0 in lam0s:
        lw_lam = dvel2dlam(linewidth, lam0) # typical linewidth in wavelength
        #lw_lam = np.abs(np.random.normal(lw_lam, np.sqrt(lw_lam))) # make it random
        if para_set=='fLfG': flux_noerr += Voigt1D(x_0=lam0, amplitude_L=1., fwhm_L=lw_lam, fwhm_G=lw_lam)(lam) # positive
        elif para_set=='blgNHI': flux_noerr += gs.voigt1d(lam0, lgNHI=13., b=lw_lam)(lam)
    cont = 1.
    flux_noerr = gs.form_spec(cont, flux_noerr, voigt_is_tau=voigt_is_tau)
    if min(flux_noerr) < 0: flux_noerr = (flux_noerr - min(flux_noerr))/(1. - min(flux_noerr)) # scale to [0,1]
    # add noise
    flux, noise = add_shot_noise(flux_noerr, Nphot, return_error=True)
    return lam, flux, noise

if para_set=='fLfG':
    para_prefix = ['lam0_', 'AL_', 'fL_', 'fG_']
    v1d = Voigt1D
elif para_set=='blgNHI':
    para_prefix = ['lam0_', 'lgNHI_', 'b_']
    v1d = gs.voigt1d
def para_list(p, paraind=0):
    '''
    Convert lf.Parameters() to parameter list (len=4)
    paraind - index of one line's parameter
    '''
    s = str(int(round(paraind)))
    plist = [p[prefix+s].value for prefix in para_prefix]
    return plist

def singlevoigt_paras(p, lam, paraind=0):
    args = para_list(p, paraind)
    return v1d(*args)(lam)

def num_paras(paras): return len([key for key in paras.keys() if para_prefix[0] in key])

def multivoigt_paras(p, lam):
    parray = paras2parray([p])
    return gs.multivoigt_parray(parray, lam, v1d=v1d)

def voigt_residual(p, voigtfunc, lam, flux, noise, continuum=1., *args):
    model = gs.form_spec(continuum, voigtfunc(p, lam, *args), voigt_is_tau=voigt_is_tau)
    res = (flux - model) / noise
    return res

def addaline(paras, il, lam, flux, ipeak, continuum=1., min_lw=0.):
    '''
    min_lw in km/s
    '''
    if type(continuum) != np.ndarray or type(continuum) != list: continuum = continuum * np.ones(len(lam)) # make continuum an array
    lam0 = lam[ipeak]
    typical_lamwidth = dvel2dlam(typical_lw, lam0)
    min_lamwidth = dvel2dlam(min_lw, lam0) # width to be greater than resolution
    max_lw_factor = 5. # 20.
    paras.add(para_prefix[0]+str(il), value=lam0, min=lam0-typical_lamwidth, max=lam0+typical_lamwidth) # lam0
    if para_set=='fLfG':
        if voigt_is_tau: paras.add(para_prefix[1]+str(il), value=-1.5*(np.log((flux[ipeak] if flux[ipeak]>0 else 1e-20)/continuum[ipeak])), min=0., max=2e2) # AL as tau_0
        else: paras.add(para_prefix[1]+str(il), value=-1.5*(flux[ipeak]-continuum[ipeak]), min=0., max=1.5) # AL, positive
        paras.add(para_prefix[2]+str(il), value=0.5*typical_lamwidth, min=min_lamwidth, max=max_lw_factor*typical_lamwidth) # fL
        paras.add(para_prefix[3]+str(il), value=0.5*typical_lamwidth, min=min_lamwidth, max=max_lw_factor*typical_lamwidth) # fG
    elif para_set=='blgNHI':
        paras.add(para_prefix[1]+str(il), value=13., min=12., max=16.) # lgNHI
        paras.add(para_prefix[2]+str(il), value=0.5*typical_lw, min=min_lw, max=max_lw_factor*typical_lw) # b

def aicc(fitresult):
    '''
    Compute AICC: AIC with Correction for small sample size
    '''
    p = fitresult.nvarys # number of parameters
    n = fitresult.ndata
    aicc = fitresult.aic + 2*p*(p+1.)/(n-p-1.)
    return aicc

def fit_region(lam, flux, noise, ipeaks, continuum=1., addline=True, min_lw=0.):
    '''
    Fit absorption lines in a region
    ipeaks - (array) indexes of lam for line peaks
    addline - whether try to add lines after fit
    min_lw - in km/s
    '''
    plot = 0
    nlines = len(ipeaks)
    # initialize parameters
    paras = lf.Parameters()
    for il in range(nlines): addaline(paras, il, lam, flux, ipeaks[il], continuum=continuum, min_lw=min_lw)
    if plot: flux_guess = gs.form_spec(continuum, multivoigt_paras(paras, lam), voigt_is_tau=voigt_is_tau)
    # fit voigt for this region
    if len(flux) <= len(paras): return None # must have ndata > npara for leastsq
    fitresult = lf.minimize(voigt_residual, paras, args=(multivoigt_paras, lam, flux, noise, continuum))
    fitresult.initparams = paras

    # decide whether to add line
    if nlines >= 20: addline = False # don't try to add line if too many
    if addline:
        max_fails = 2 # how many times of fails allowed for adding a line at a time
        fails = 0
        if nlines == 0: il = -1 # no line detected in region
        while fails < max_fails: # add a line at a time
            il += 1
            addaline(paras, il, lam, flux, int(len(lam)/2), continuum=continuum, min_lw=min_lw)
            paras[para_prefix[0]+str(il)].set(value=np.mean(lam), min=min(lam), max=max(lam)) # adjust lam0 to be mean(lam)
            if para_set=='fLfG': paras[para_prefix[1]+str(il)].set(value=(np.mean(flux)-np.mean(continuum))) # adjust AL to be mean(flux-cont)
            if len(flux) - len(paras) <= 1: break # must have ndata - npara > 1 for aicc
            print('\tTrying to add line %d ...'%(il+1)),
            addlineresult = lf.minimize(voigt_residual, paras, args=(multivoigt_paras, lam, flux, noise, continuum))
            addlineresult.initparams = paras
            if aicc(addlineresult) < aicc(fitresult):
                fitresult = addlineresult
                fails = 0
                print('success')
            else:
                fails += 1
                print('fail')

    flux_fit = gs.form_spec(continuum, multivoigt_paras(fitresult.params, lam), voigt_is_tau=voigt_is_tau)
    if plot:
        plt.plot(lam, flux_fit, 'r', lw=0.5)
        plt.plot(lam, flux_guess, 'b', lw=0.5)
        plt.plot(lam, flux, 'k', lw=0.5)
        plt.plot(lam[ipeaks], flux[ipeaks], 'vb')
        plt.show()
    return fitresult

def paras2parray(paralist):
    '''
    Inputs: paralist - list of lf.Parameters
    Outputs: parray - parameter array, dim:[nparas, nlines]
    '''
    parray = []
    for paras in paralist:
        npara = num_paras(paras)
        for il in range(npara): parray.append(para_list(paras, il))
    return np.array(parray).T

def results2parray(results): return paras2parray([result.params for result in results]) # parameters array
def results2initparray(results): return paras2parray([result.initparams for result in results]) # inital parameters array
def pfile2parray(pfile): return results2parray(pkloadgzip(pfile))
def pfile2flux(pfile, lam, continuum=1.): return gs.parray2flux(pfile2parray(pfile), lam, continuum, voigt_is_tau=voigt_is_tau, v1d=v1d)

def fit_forest(lam, flux, noise, continuum=1., tosave=None, addline=True, CSLcut=1.5, plot=False, chkind=False, verbose=True, gapranges=None, min_lw=0.):
    '''
    Spectrum should be normalized to [0,1]
    addline - whether try to add lines after fit
    chkind - whether to return region_indlims from find_regions
    gapranges - [[lamleft, lamright], ...] if there are gaps in the input spec, fill them in with random lines drawn from existing distribution
    min_lw - minimum line width when fitting in km/s
    '''
    # ----------- divide spectra to regions -----------
    tosave_findreg = tosave[:tosave.rfind('.')] + '_findreg' + tosave[tosave.rfind('.'):]
    typical_pixwidths = dvel2dlam(typical_lw, lam) / np.gradient(lam) # typical linewidth in pixels for each lam
    max_pixwidth = np.mean(typical_pixwidths)+3*np.std(typical_pixwidths) # 3 sigma upper bound
    region_lamlims, region_indlims, region_indpks = \
        vamp.find_regions(lam, flux, noise, continuum=0.99*continuum, extend=True, \
        peak_dist=1, N_sigma=CSLcut, max_pixwidth=max_pixwidth, plot=0, tosave=tosave_findreg, verbose=verbose) # [[lam_start, lam_end]], [[i_start, i_end]] (flux[i_start:i_end])
    '''
    if np.sum([len(ipk) for ipk in region_indpks]) < 100:
        print('Number of lines less than 100, pass.')
        return None
    '''
    if plot: # plot spectrum, regions and line peaks
        plt.plot(lam, flux, 'k', lw=0.5)
        plt.plot(lam, noise, 'g', lw=0.5)
        for ireg in range(len(region_lamlims)):
            plt.fill_betweenx([0,continuum],region_lamlims[ireg][0],region_lamlims[ireg][1],alpha=0.3,color='y')
            plt.plot(lam[region_indlims[ireg][0]:region_indlims[ireg][1]][region_indpks[ireg]],\
                     flux[region_indlims[ireg][0]:region_indlims[ireg][1]][region_indpks[ireg]], 'vb')

    # ----------- fit voigts for each region -----------
    if tosave and os.path.exists(tosave):
        results = pkloadgzip(tosave, verbose=verbose)
    else:
        results = []
        t1 = time.time()
        for ireg, [start, end] in enumerate(region_indlims): # loop through each region
            tosave_reg = tosave[:tosave.rfind('.')]+'_reg%d'%ireg+tosave[tosave.rfind('.'):]
            if tosave and os.path.exists(tosave_reg):
                result_reg = pkloadgzip(tosave_reg, verbose=verbose)
            else:
                lam_reg = lam[start:end]
                flux_reg = flux[start:end]
                flux_reg = np.ma.masked_where(flux_reg < 0., flux_reg) # mask negative values
                noise_reg = noise[start:end]
                ipeaks_reg = region_indpks[ireg]
                nlines = len(ipeaks_reg)
                if (nlines == 0) and (not addline): continue
                #         fit continuum
                #ppoly = fit_poly([lam_reg, flux_reg], poly_deg=3)
                #continuum = np.polyval(ppoly, lam_reg)
                #         fit line
                print('Fitting %d lines for region %d/%d ...'%(nlines, ireg+1, len(region_indlims)))
                t2 = time.time()
                result_reg = fit_region(lam_reg, flux_reg, noise_reg, ipeaks_reg, continuum=continuum, addline=addline, min_lw=min_lw)
                print('%.2f minutes. Total %.2f minutes.'%((time.time()-t2)/60., (time.time()-t1)/60.))
                if tosave: pkdumpgzip(result_reg, tosave_reg)
            if result_reg != None: results.append(result_reg)
            plotreg = 0
            if plotreg:
                lam_reg = lam[start:end]
                flux_reg = flux[start:end]
                plt.plot(lam_reg, flux_reg, 'k', lw=0.5)
                flux_fit = gs.parray2flux(results2parray([result_reg]), lam_reg, continuum=continuum, voigt_is_tau=voigt_is_tau, v1d=v1d)
                plt.plot(lam_reg, flux_fit, 'r', lw=0.5)
                plt.show()
        if tosave: pkdumpgzip(results, tosave)

    # ----------- analysis -----------
    if plot: # plot fitted flux
        bestp_tot_arr = results2parray(results)
        '''
        import sse_lya_sims_zlines_27jun2019_steve as sse
        lam0, al, fl, fg = bestp_tot_arr
        al_cut = -1.2
        fg_cut = dvel2dlam(20., lam0) # AA
        fv_intrin, fv_para = sse.intrinsic_fwhm(fl, fg, lam0, sse.smooth)
        fwhm_intrin = dlam2dvel(fv_intrin, lam0)
        fwhm_para = dlam2dvel(fv_para, lam0)
        fwhm_cut = 20. # km/s
        '''
        initp_tot_arr = results2initparray(results)
        flux_fit = gs.parray2flux(bestp_tot_arr, lam, continuum=continuum, voigt_is_tau=voigt_is_tau, v1d=v1d)
        flux_guess = gs.parray2flux(initp_tot_arr, lam, continuum=continuum, voigt_is_tau=voigt_is_tau, v1d=v1d)
        #    plot fitted spec
        fig, ax = plt.subplots(figsize=(12,3))
        ax.axhline(1,color='k') # continuum
        ax.plot(lam, flux, 'k.', ms=3)#, lw=0.5)
        ax.plot(lam, flux_fit, 'r')
        ax.set_ylim([-0.5, 1.5])
        #ax.plot(lam, flux_guess, 'b', lw=0.5)
        #ax.plot(lam, flux - flux_fit - 0.5, 'k', lw=0.5) # residuals
        #ax.axhline(-0.5,color='k',lw=0.5) # residual zero-points
        #    plot linewidth hist
        '''
        plt.figure()
        plt.hist(fwhm_intrin, bins=np.arange(0,max(fwhm_intrin)+1,1), histtype='step', label='intrinsic')
        #plt.hist(fwhm_para[fwhm_para>fwhm_cut], color = 'r', bins=np.arange(0,max(fwhm_para)+1,1), histtype='step')
        plt.hist(fwhm_para, bins=np.arange(0,max(fwhm_para)+1,1), histtype='step', label='observed')
        plt.legend()
        print('median observed line width: %.2f km/s'%np.median(fwhm_para))
        plt.xlabel('line width FWHM (km/s)')
        '''
        ax.set_xlabel(r'$\lambda$ ($\mathrm{\AA}$)')
        ax.set_ylabel('Normalized flux')
        fig.tight_layout()

    if chkind: return results, region_indlims
    else: return results

if __name__ == '__main__':
    ### generate test spectra
    #lam, flux, noise = generate_test_spec(nlines=50)
    ### use keck spectra
    import continuum_fit as cf
    import generate_spec as gs
    import read_elqs as re
    import read_ps as rp
    import realll_vs_simll as rvs
    top10 = re.top10m14

    class keck_args:
        fitcont_dist = 100
        fitcont_deg = 10 # added after referee report
        fitcont_mode = 'poly' # poly, linear or cubic
        vfaddline = True # voigtforest for keck data
        CSL_cut = 1.5 # voigtforest for keck data
        smoothwidth = 15 # whether smooth during read_koa, if ==0, not smooth
        min_lw = 5. # km/s
    smoothtxt = '_smooth%dpix'%keck_args.smoothwidth if keck_args.smoothwidth!=None else '_nosmooth'
    fitconttxt = '_dist%d'%keck_args.fitcont_dist + ('_deg%s'%keck_args.fitcont_deg if keck_args.fitcont_mode=='poly' else '_cont%s'%keck_args.fitcont_mode)
    addlinetxt = '_fitaddline' if keck_args.vfaddline else ''
    csltxt = '_CSLcut%.1f'%keck_args.CSL_cut
    minlwtxt = '' if keck_args.min_lw==0 else '_minlw%skms'%keck_args.min_lw

    # check nlines
    '''
    percentage = np.zeros([len(top10), 2]) # real_nlines/sim_nlines
    nlines = np.zeros([len(top10), 2])
    increase = []
    for ind,item in enumerate(top10.iloc):
        zqso = item['z']
        kid = item['KOAjobID']
        print('%6d'%kid, 'z = %.3f'%zqso, end=', ')
        sim_nlines = gs.nlines(zqso, random=False)
        print('theory nlines = %5d'%sim_nlines, end=', ')
        for vfaddline in [False, True]:
            addlinetxt = '_fitaddline' if vfaddline else ''
            print('addline = %d'%(vfaddline), end=', ')
            pfile = path + 'paras/voigtforest_bestparray_%d_smooth30pix_dist100_deg%s%s_CSLcut1.5%s.pzip'%(kid, keck_args.fitcont_deg, addlinetxt, minlwtxt)
            parray = pkloadgzip(pfile, verbose=False)
            if vfaddline: n_noadd = nl
            nl = parray.shape[1]
            nlines[ind, int(vfaddline)] = nl
            pct = nl/sim_nlines*100.
            percentage[ind, int(vfaddline)] = pct
            print('nlines = %5d (%2d%%)'%(nl, pct), end=', ')
            if vfaddline: 
                inc = (nl - n_noadd)/n_noadd*100.
                increase.append(inc)
                print('(%2d%% increase)'%(inc), end=', ')
        print('')
    print('average nlines: noaddline %.1f, addline %.1f'%tuple(np.mean(nlines, axis=0)))
    print('average percentage: noaddline %2d%%, addline %2d%%'%tuple(np.mean(percentage, axis=0)), '(%2d%% increase)'%np.mean(increase))
    '''

    # plot figure
    rest_frame = True # if plot in rest frame

    kid = 10408 # 36576
    kids = rvs.kid_withspec(exclude_weak=False)
    #item = df_all.loc[12850]
    #for item in top10.iloc:
    #for kid in kids:
    for kid in [104297, 119681, 125810, 12850, 20463, 2315, 29524, 3105, 32118, 54447, 7260, 9075]:
    #if 1:
        #item = rp.top10vds_N[rp.top10vds_N.KOAjobID==kid].iloc[0] # if kid in top10vds_N
        item = df_all[df_all['KOAjobID']==kid].iloc[0] # if kid not in top10vds_N; but cannot set rest_frame=False
        koa_spec, gapranges = cf.get_keck_spec(item, keck_args.fitcont_dist, keck_args.fitcont_deg, keck_args.fitcont_mode, rest_frame=True, smoothwidth=keck_args.smoothwidth, return_gaprange=True) # in rest frame
        lam, kflux, knoise = koa_spec

        # voigtfit
        koajobid = int(item['KOAjobID'])
        vffile = path + f'paras/voigtforest_bestp_{koajobid:d}{smoothtxt}{fitconttxt}{addlinetxt}{csltxt}{minlwtxt}.pzip'
        keckvfparray_file = vffile.replace('bestp', 'bestparray')
        if os.path.exists(keckvfparray_file):
            print(f'Already created: {keckvfparray_file}, skipping')
            continue # if not in a loop then exit via error raising
        results = fit_forest(*koa_spec, tosave=vffile, addline=keck_args.vfaddline, CSLcut=keck_args.CSL_cut, plot=False, min_lw=keck_args.min_lw)
        vfparray = results2parray(results)
        #flux_vf = gs.parray2flux(vfparray, lam, v1d=gs.voigt1d)

        # adjust parameters by Keck resolution resolution, no matter frame
        import sse
        import read_koa as rk
        try: catalog = item['catalog']
        except Exception: catalog = 'elqs'
        vfparray = sse.keck_intrinsic_lw(vfparray, R_keck=rk.get_res(item['KOAjobID'], catalog), smoothwidth=None)

        # fill gaps in voigt parameters, should in rest frame
        vfparray = su.fill_gap(gapranges, vfparray)
        pkdumpgzip(vfparray, keckvfparray_file)
        reg_file_toglob = vffile[:vffile.rfind('.')]+'_reg*'+vffile[vffile.rfind('.'):]
        os.system(f'rm {reg_file_toglob}')
        print(f'Deleted: {reg_file_toglob}')
        continue # no plotting

        # generate_spec
        zqso = item['z']
        blgNHIz_file = path + 'paras/blgNHIz_zqso%.3f.pickle'%(zqso)
        parray_gen = gs.bNHIz_generator(zqso, blgNHIz_file, rest_frame=rest_frame)
        flux_gen = gs.parray2flux(parray_gen, lam, v1d=gs.voigt1d)

        # plot voigtfit and genspec
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={'hspace':0})

        # plot voigtfit
        axes[0].axhline(1, color='k')
        axes[0].plot(lam, kflux, 'k.', ms=3)
        axes[0].plot(lam, flux_vf, 'r')
        axes[0].set_ylim([-0.4, 1.4])
        axes[0].set_ylabel('Real spectrum')

        # plot genspec
        axes[1].axhline(1, color='k')
        axes[1].plot(lam, flux_gen, 'r')
        axes[1].set_ylim([-0.4, 1.4])
        axes[1].set_xlim([1100., 1150.])
        axes[1].set_xlabel(r'Rest frame wavelength ($\mathrm{\AA}$)')
        axes[1].set_ylabel('Simulated spectrum')

        #axes[0].set_xlim([1125., 1127.]) # to zoomin in debugging

        # add x axis showing observed frame wavelength
        axp = axes[0].twiny()
        lamlim_obsframe = su.obs_frame(np.array(axes[0].get_xlim()), zqso)
        axp.set_xlim(lamlim_obsframe)
        axp.set_xlabel(r'Observed frame wavelength ($\mathrm{\AA}$)')

        fig.tight_layout()
        #tosave = path + 'plots/voigtforest2genspec_%d%s_zoom1125-1127.pdf'%(item['KOAjobID'], addlinetxt) # to zoomin in debugging
        tosave = path + 'plots/voigtforest2genspec_%d_deg%s%s%s.pdf'%(item['KOAjobID'], keck_args.fitcont_deg, addlinetxt, minlwtxt)
        fig.savefig(tosave);print('Saved:%s'%tosave)
        #plt.show()

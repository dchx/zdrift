from utils import *
from spec_utils import *
import spec_utils as su
import m1450
import read_sdss as rs
import continuum_fit as cf
#import target_selection as ts
import liske_sigma as ls
import cosmology as cs
from astropy.coordinates import SkyCoord, Angle

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
        
def plot_elqs(koajobidlist=None, keck_catalog_list=None, titles=None, tosave=None, sdss=None, norm_factor=1., M1450=None, zqso=None, df=None):
    '''
    must provide (koajobidlist, keck_catalog_list) or (df)
    df - pandas.DataFrame
    '''
    if type(df)!=type(None):
        if type(df)==pd.DataFrame: df = df.set_index(pd.RangeIndex(len(df)))
        koajobidlist = df.KOAjobID
        if 'catalog' in df.columns: keck_catalog_list = df.catalog
        if 'M1450' in df.columns:
            M1450 = df.M1450
            zqso = df.z
            if type(M1450)!=type(None): f1450 = m1450.m14502sdss(M1450, zqso) # 1450AA flux in 1e-17 erg / (cm2 s AA)
        if 'norm_factor' in df.columns: norm_factor = df.norm_factor
        if 'SDSS' in df.columns: sdss = df.SDSS
    koajobidlist = np.atleast_1d(koajobidlist)
    keck_catalog_list = np.atleast_1d(keck_catalog_list)
    sdss = np.atleast_1d(sdss)
    norm_factor = np.atleast_1d(norm_factor)
    # keck_catalog_list
    if len(keck_catalog_list)==1: keck_catalog_list = np.repeat(keck_catalog_list, len(koajobidlist))
    keck_catalog_list = np.array(['.' if x =='sdss' else x for x in keck_catalog_list]) # convert 'sdss' to '.'
    # norm_factor
    if len(norm_factor)==1: norm_factor = np.repeat(norm_factor, len(koajobidlist))
    # SDSS files
    sdss_path = path + 'data/sdss/'

    matplotlib.rc('font',size=5) # font size
    fig, ax = plt.subplots(*subplot_shape(len(koajobidlist)), figsize=(12, 8))
    ax = np.atleast_1d(ax).flatten()
    for iplot, koajobid in enumerate(koajobidlist):
        if np.isnan(koajobid): continue
        if not np.any(titles): title = 'KOAjobID: %d'%koajobid
        else: title = titles[iplot]
        print(title)
        ax[iplot].set_title(title)
        try: data = cf.trim_koaspec(koajobid, plot_lya_forest=False, keck_catalog=keck_catalog_list[iplot])
        except Exception as e:
            print(type(e).__name__+':', e)
            continue
        data = np.array(data)
        # normalize
        if (not any(sdss) or (type(sdss[iplot])!=str)) and (type(M1450)!=type(None)) and (~np.isnan(M1450.iloc[0])): # normalize by m1450
            data = m1450.norm2f1450(data, f1450[iplot])
        else: # pre-defined factor
            if not np.isnan(norm_factor[iplot]): data[1:3] *= norm_factor[iplot]
        lam = data[0]
        flux = data[1]
        '''
        if np.any(sdss): # only for top11
            # convert to 1e-17 erg/(cm2 s AA)
            if   iplot==0: ax[iplot].plot(lam, flux*3.59, lw=0.5)
            elif iplot==1: ax[iplot].plot(lam, flux*0.15, lw=0.5)
            elif iplot==2: ax[iplot].plot(lam, flux*26., lw=0.5)
            elif iplot==3: ax[iplot].plot(lam, flux*0.3, lw=0.5)
            elif iplot==5: ax[iplot].plot(lam, flux*10., lw=0.5)
            elif iplot==6: ax[iplot].plot(lam, flux*0.12, lw=0.5)
            elif iplot==7: ax[iplot].plot(lam, flux*0.43, lw=0.5)
            elif iplot==8: ax[iplot].plot(lam, flux*0.39, lw=0.5)
            elif iplot==9: ax[iplot].plot(lam, flux*0.6, lw=0.5)
            elif iplot==10: ax[iplot].plot(lam, flux*0.5, lw=0.5)
            else: ax[iplot].plot(lam, flux, lw=0.5)
        else: ax[iplot].plot(lam, flux, lw=0.5)
        '''
        ax[iplot].plot(lam, flux, lw=0.5)
        ax[iplot].axvline(su.lya_wave, c='k')
        ax[iplot].axvline(lyb_wave, c='k')
        ax[iplot].axvline(1450, c='r')
        #ax[iplot].set_xbound([lyb_wave, su.lya_wave])
        # set ylim
        '''
        flux_min = max(-15, ax[iplot].get_ylim()[0])
        flux_max = min(2000, ax[iplot].get_ylim()[1])
        ax[iplot].set_ylim([flux_min, flux_max])
        '''

        if np.any(sdss) and (type(sdss[iplot])==str) and sdss[iplot].endswith('fits'): # plot sdss
            #try: sdss_data = rs.read_sdss_top11koajobid(koajobid, plot_rest_frame=True)
            #except Exception: continue
            sdss_file = glob.glob(sdss_path + f'**/{sdss[iplot]}', recursive=True)[0]
            sdss_data = rs.read_sdss_file(sdss_file, z_plot_rest_frame=zqso[iplot]) # in rest frame
            ax[iplot].plot(sdss_data[0], sdss_data[1], 'k', lw=0.5)
            #ax[iplot].errorbar(sdss_data[0], sdss_data[1], sdss_data[2])#, c='k', lw=0.5)

        # plot 1450AA flux
        if type(M1450)!=type(None) and (not np.isnan(f1450[iplot])):
            ax[iplot].plot(1450., f1450[iplot], 'xr')
            
    if type(sdss)!=type(None): ax[0].set_ylabel('$10^{-17}$ erg / (s cm2 AA)')
    plt.tight_layout()
    if tosave!=None: fig.savefig(tosave); print('Saved: %s'%tosave)
    plt.pause(0.1)

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
            #criteria = criteria_base & (item.z <= d.z) & (d.z <= item.z + 0.1)
            #if item.No == 5: criteria = criteria_base & (d.z >= item.z - 0.2)
            criteria = criteria_base & (item.z - 0.2 <= d.z)
            substitutes = d[criteria]
            print('----------')
            print(substitutes.to_string())

def plot_substitute_old():
    top11 = pd.read_csv(path + 'data/elqs_full_sortM1450_addmore.csv')[:11]
    dfs = {'.': get_matched('.')[0], 'elqs': get_matched('elqs')[0]}
    dic = {2: ['.', 10408], 4: ['elqs', 119681], 5: ['elqs', 43254], 7: ['elqs', 54447], 8: ['elqs', 119681], 9: ['elqs', 2315], 10: ['elqs', 119681], 11: ['.', 12850]} # No: [keck_catalog, koajobid]
    dic = {2: ['.', 10408], 4: ['elqs', 57261], 5: ['elqs', 43254], 6: ['elqs', 20463], 7: ['elqs', 54447], 8: ['.', 29524], 9: ['elqs', 2315], 10: ['elqs', 125810], 11: ['.', 12850]} # No: [keck_catalog, koajobid]
    dic = {2: ['elqs', 125810], 3:['elqs', 36576], 4: ['.', 3105], 5: ['elqs', 43254], 6: ['elqs', 20463], 7: ['elqs', 54447], 8: ['.', 29524], 9: ['elqs', 2315], 10: ['.', 32118], 11: ['.', 10408]} # No: [keck_catalog, koajobid], changing for spectra with small gaps / too noisy
    
    kjlist = [] # koajobid list
    kclist = [] # keck_catalog list
    titles = []
    for item in top11.iloc:
        num = item.No
        if str(item['use'])!='True' and str(item['use'])!='check': # use substitute
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
    tosave = path + 'plots/top11_substitutes.pdf'
    tosave = None
    plot_elqs(kjlist, kclist, titles, tosave, plot_sdss=True)

def dvdt_over_sigma(zqso, rmag, t_tot=1.9e7, diameter=15., efficiency=0.25, nepoch=120, period=20., return_division=True):
    zmid = (zqso + su.z_lyb(zqso)) / 2.
    dvdt = cs.dvdt(zmid)
    snr = ls.snr_empir(rmag, t_tot, diameter, efficiency)
    sigma_dvdt = ls.sigma_liske_empir_multiepoch(zqso, snr, nepoch, nqso=1, lyb2lya=True) / period
    if return_division: return np.abs(dvdt) / sigma_dvdt
    else: return dvdt, sigma_dvdt
df_elqs_ps['dvdtosig'] = dvdt_over_sigma(df_elqs_ps['z'].to_numpy(), df_elqs_ps['rmag'].to_numpy())
df_elqs['dvdtosig'] = dvdt_over_sigma(df_elqs['z'].to_numpy(), df_elqs['rmag'].to_numpy())
southern['dvdtosig'] = dvdt_over_sigma(southern['z'].to_numpy(), southern['rmag'].to_numpy())

def top10_df(substitute=True, selection='m1450'):
    '''
    return the selected top10 m1450 items from the dataframe
    selection - 'm1450' or 'dvdtsigma'
    '''
    # substitutes dict
    dic = {2: ['elqs', 125810], 3:['elqs', 36576], 4: ['.', 3105], 5: ['elqs', 43254], 6: ['elqs', 20463], 7: ['elqs', 54447], 8: ['.', 29524], 9: ['elqs', 2315], 10: ['.', 32118], 11: ['.', 10408]} # elqs_No: [keck_catalog, koajobid], changing for spectra with small gaps / too noisy
    #dic = {2: ['elqs', 125810], 4: ['.', 3105], 5: ['elqs', 43254], 6: ['elqs', 20463], 7: ['elqs', 54447], 8: ['.', 29524], 9: ['elqs', 2315], 10: ['.', 32118], 29: ['.', 10408], 36: ['.', 10408], 23: ['.', 7260], 11: ['.', 7260], 14: ['.', 12850], 33: ['.', 12850]} # elqs_No: [keck_catalog, koajobid], (new substitute), dropping No 3 (using original), changing No 11, adding dvdt/sigma selection
    #dic = {2: ['elqs', 125810], 4: ['.', 3105], 5: ['elqs', 43254], 6: ['elqs', 20463], 7: ['elqs', 54447], 8: ['.', 29524], 9: ['elqs', 2315], 10: ['.', 32118], 29: ['.', 10408], 36: ['.', 10408], 23: ['.', 10408], 11: ['.', 7260], 14: ['.', 7260], 33: ['.', 7260]} # elqs_No: [keck_catalog, koajobid], new substitute dropping 12850
    #dic = {2: ['elqs', 125810], 3:['elqs', 36576], 4: ['.', 3105], 5: ['elqs', 43254], 6: ['elqs', 20463], 7: ['elqs', 54447], 8: ['.', 29524], 9: ['elqs', 2315], 10: ['.', 32118], 29: ['.', 10408], 36: ['.', 10408], 23: ['.', 7260], 11: ['.', 7260], 14: ['.', 12850], 33: ['.', 12850]} # elqs_No: [keck_catalog, koajobid], new substitute not dropping No 3

    # m1450 selection
    elqs_Nos_top10 = list(range(1, 12)) # 1 to 11
    elqs_Nos_top10.remove(5) # remove No 5. [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    top10_nosub_m1450 = df_elqs.set_index(df_elqs.No).loc[elqs_Nos_top10].set_index(pd.RangeIndex(1, 11))
    if selection.lower() in 'm1450':
        top10_nosub = top10_nosub_m1450
    # dvdt/sigma selection
    elif selection.lower() in 'dvdtsigma':
        top10_nosub = df_elqs[df_elqs['DESdeg']>-20].sort_values('dvdtosig', ascending=False)[:10] # northern sample
    if substitute:
        top10_m1450 = pd.DataFrame()
        for num in top10_nosub_m1450['No']:
            M1450_origin = top10_nosub_m1450[top10_nosub_m1450['No']==num]['M1450']
            z_origin = top10_nosub_m1450[top10_nosub_m1450['No']==num]['z']
            M1450_origin.index = ['M1450_origin']
            z_origin.index = ['z_origin']
            if num in dic.keys(): # substituted ones
                toinsert = df_all.loc[dic[num][1]] # substituted ones, Series
            else: # not substituted ones
                toinsert = df_all[df_all.elqs_No==num].iloc[0] # not substituted ones, Series
            #toinsert = pd.Series(toinsert.append(M1450_origin), name=toinsert.name)
            toinsert = pd.Series(pd.concat([toinsert, M1450_origin]), name=toinsert.name)
            #toinsert = pd.Series(toinsert.append(z_origin), name=toinsert.name)
            toinsert = pd.Series(pd.concat([toinsert, z_origin]), name=toinsert.name)
            #top10_m1450 = top10_m1450.append(toinsert)
            top10_m1450 = pd.concat([top10_m1450, pd.DataFrame(toinsert).T])
            top10_m1450.index.name = df_all.index.name
        if selection.lower() in 'm1450':
            top10 = top10_m1450
        elif selection.lower() in 'dvdtsigma':
            # assgin by z
            substuting = top10_m1450.sort_values('z')
            origin = top10_nosub.sort_values('z')
            substuting['z_origin'] = origin['z'].to_numpy()
            substuting['M1450_origin'] = origin['M1450'].to_numpy()
            top10 = substuting
    else: top10 = top10_nosub
    return top10
top10m14 = top10_df(True, 'm1450')
top10m14_nosub = top10_df(False, 'm1450')
top10vds = top10_df(True, 'dvdtsigma')
top10vds_nosub = top10_df(False, 'dvdtsigma')
top10 = top10m14
top10_nosub = top10m14_nosub

def plot_substitute():
    plot_elqs(df=top10.set_index(pd.RangeIndex(len(top10))))

def check_elqs_spec():
    toplot = d[-np.isnan(d['KOAjobID']) & d['extracted']==True]
    plot_elqs(toplot['KOAjobID'])

def check_original_spec():
    ind_no_use = [0, 22, 44, 51, 55, 56, 62, 66, 73, 81, 85, 87, 103, 109, 111, 137, 148, 157, 166, 183, 184, 186, 196, 199, 211, 213, 223, 246, 283, 284, 289, 306, 340, 350, 365, 372]
    ind_check = [53, 60, 67, 69, 84, 92, 93, 101, 104, 125, 127, 145, 150, 156, 161, 164, 168, 172, 185, 202, 204, 216, 282, 286]
    ind_use = [17, 20, 47, 142, 154, 160, 163, 200, 305]
    toplot = d[(d['KOAjobID']!=0) & (d['extracted']==True)]
    plot_elqs(toplot['KOAjobID'])

def norm_top10():
    '''
    normalize top10 to one
    '''
    for ind,item in enumerate(top10.iloc): 
        spec_normed = cf.get_keck_spec(item, normalize=True, rest_frame=False, plot=True) 
        #plt.savefig(path+'plots/norm_%d_koa%d.pdf'%(ind+1,item['KOAjobID'])) 
        plt.show()
        plt.close() 

def substitute_flux_factor(item):
    '''
    from substituted flux to original spec flux
    factor = f1450_origin / (sdss.flux)_substituted
    flux_origin = flux_substituted * factor
    '''
    # origin 1450 flux
    f1450_origin = m1450.m14502sdss(item['M1450_origin'], item['z_origin']) # 1450AA flux in 1e-17 erg / (cm2 s AA)

    havesdss = rs.check_item_havesdss(item) # use SDSS
    if not havesdss:
        f1450_sub = m1450.m14502sdss(item['M1450'], item['z']) # 1450AA flux in 1e-17 erg / (cm2 s AA)
        factor = f1450_origin / f1450_sub
    else:
        # read sdss
        sdssfile = glob.glob(path + 'data/sdss/**/' + item['SDSS'], recursive=True)[0]
        sdss = rs.read_sdss_file_class(sdssfile)
        sdss_spec_rest = np.array([su.rest_frame(sdss.lam, item['z']), sdss.flux]) # in rest frame

        # f1450
        factor = m1450.norm2f1450(sdss_spec_rest, f1450_origin, return_factor=True)
    return factor

def mask_add_item(item, adjust_zqso=None):
    '''
    mask sections of spectra with bad data / don't want to be in continuum fit
    assuming at rest frame
    adjust_zqso - if none, assuming lam not adjusted by lya search, else adjust lam from item['z'] to adjust_zqso
    contmask - continuum fit mask, mask local max points after local_peak_max searching, under smoothwidth==30
    contadd - list of scalar or list; if scalar: wavelength point to add; if list: [lam, flux] of point to add, flux in keck flux (if have SDSS spectra) or norm by M1450 (if don't have SDSS spectra)
    '''
    top10 = top10_df().set_index(pd.RangeIndex(1, 11))
    class args:
        contmask = []
        contadd = []
        voigtmask = []

    if item['KOAjobID']==104297:
        args.contmask = [[1029., 1030.], [1044., 1045.], [1059., 1060.], [1063., 1066.], [1077., 1078.], [1081., 1086.], [1093., 1094.], [1101., 1104.], [1107., 1108.], [1117., 1118.], [1122., 1123.], [1130., 1131.], [1141., 1142.], [1162., 1165.], [1167., 1172.], [1183., 1189.], [1205., 1206.]]

    if item['KOAjobID']==125810:
        args.contmask = [[1043., 1044.], [1059., 1060.], [1066., 1068.], [1072., 1075.], [1086., 1087.], [1089., 1098.], [1103., 1112.], [1118., 1121.], [1133., 1136.], [1148., 1160.], [1165., 1170.], [1173., 1174.], [1177., 1178.], [1185., 1188.], [1195., 1201.]]
        args.contadd = [[1044.47, 104.563]]

    if item['KOAjobID']==36576:
        args.contmask = [[1025., 1026.], [1040., 1042.], [1085., 1086.], [1119., 1120.], [1133., 1134.], [1137., 1142.], [1172., 1177.], [1180., 1190.], [1197., 1120.]]
        args.contadd = [1212.44]

    if item['KOAjobID']==3105:
        args.contmask = [[1025., 1026.], [1028., 1029.], [1035., 1035.5], [1038., 1040.], [1048., 1049.], [1053.5, 1055.], [1058., 1059.], [1059.5, 1060.], [1067., 1068.], [1071, 1072.], [1081., 1086.], [1087., 1088.], [1098., 1103.5], [1105., 1106.], [1110., 1111.], [1112., 1113.], [1115., 1118.], [1119., 1120.], [1122., 1123.], [1133., 1134.], [1135., 1136.], [1147., 1148.], [1149.5, 1150.], [1152., 1154.], [1160., 1164.], [1167.5, 1168.5], [1179., 1184.5], [1185., 1186.], [1187., 1188.], [1188.5, 1195.], [1201., 1203.], [1204., 1207.], [1210., 1212.]]
        args.contadd = [[1098.32, 15.3855], 1213.09, 1215.]

    if item['KOAjobID']==20463:
        args.contmask = [[1025., 1035.], [1042., 1052.], [1064., 1066.], [1062., 1066.], [1069., 1071.], [1076., 1077.], [1109., 1122.], [1160., 1170.], [1188., 1189.], [1192., 1196.], [1202., 1203.], [1208., 1209.]]
        args.contadd = [[1032.38, 365.787], [1027.96, 370.379]]

    if item['KOAjobID']==54447:
        args.contmask = [[1025., 1026.], [1040., 1041.], [1050., 1056.], [1085., 1086.], [1088., 1090.], [1030.5, 1031.5], [1100., 1101.], [1108., 1109.], [1116., 1117.], [1126., 1132.], [1141., 1148.], [1157., 1160.], [1167., 1170.], [1176., 1177.], [1186., 1187.], [1195., 1198.], [1204., 1205.], [1215., 1216.]]
        args.contadd = [[1025.76, 399.118], 1043.17, 1210.89, 1213.26, 1215.21]

    if item['KOAjobID']==29524:
        args.contmask = [[1025., 1026.], [1033., 1034.], [1044., 1045.], [1050., 1051.], [1060., 1063.], [1068., 1069.], [1077., 1079.], [1089., 1091.], [1093., 1097.], [1099., 1100.], [1106., 1127.], [1132., 1133.], [1142., 1143.], [1148., 1151.], [1155., 1158.], [1174., 1175.], [1178., 1180.], [1183., 1190.], [1193., 1195.], [1215., 1216.]]
        args.contadd = [1045.03, 1214.98, 1210.]

    if item['KOAjobID']==2315:
        args.contmask = [[1025., 1026.], [1030., 1035.], [1040., 1041.], [1045., 1048.], [1050., 1052.], [1056., 1057.], [1059., 1060.], [1085., 1087.], [1088., 1089.], [1095., 1096.], [1104., 1105.], [1108., 1111.], [1115., 1118], [1120., 1122.], [1129., 1130.], [1131., 1132.], [1138., 1139.], [1140., 1141.], [1142., 1143.], [1148., 1150.], [1155., 1157.], [1158., 1160.], [1165., 1182.], [1189., 1193.], [1204., 1205.], [1215., 1216.]]

    if item['KOAjobID']==32118:
        args.contmask = [[1035., 1036.], [1046., 1047.], [1063., 1069.], [1085., 1086.], [1094., 1095.], [1104., 1105.], [1118., 1119.], [1122., 1125.], [1110., 1113.], [1130., 1131.], [1142., 1145.], [1153., 1158.], [1166., 1171.], [1182., 1190.], [1198., 1199.], [1202., 1203.]]
        args.contadd = [[1035.31, 142.79], 1209.97, 1212.59]

    if item['KOAjobID']==10408:
        args.contmask = [[1025., 1026.], [1028., 1029.], [1039., 1040.], [1041.5, 1042.5], [1044., 1048.], [1055., 1060.], [1062., 1063.], [1074., 1076.], [1080., 1083.], [1097., 1109.], [1114., 1116.], [1118., 1128.], [1130., 1131.], [1133., 1134.], [1139., 1148.], [1152., 1157.], [1163., 1182.], [1187., 1194.], [1195., 1199.], [1199.5, 1207.], [1210., 1211.], [1214., 1216.]]
        args.contadd = [[1025.81, 81.447], [1097.57, 66.7052]]

    if item['KOAjobID']==7260:
        args.contmask = [[1025., 1027.], [1029., 1030.], [1035., 1036.], [1045., 1046.], [1047., 1048.], [1049.5, 1050.5], [1052., 1053.], [1055.5, 1057.], [1060., 1065.], [1071., 1073.], [1079., 1084.], [1086., 1087.], [1114., 1114.5], [1116., 1118.], [1118.5, 1122.], [1126., 1130.], [1131., 1132.], [1134., 1151.], [1152.5, 1153.5], [1154., 1155.], [1156., 1158.], [1161., 1162.], [1166., 1172.], [1173., 1178.], [1180., 1183.], [1184., 1187.], [1190., 1196.], [1201., 1204.], [1205., 1207.], [1207.5, 1210.], [1212., 1213.], [1215., 1216.]]
        args.contadd = [[1025.76, 71.2429], [1086.63, 49.1428], [1215.66, 112.11]]

    if item['KOAjobID']==12850:
        args.contmask = [[1025., 1035.], [1036., 1037.], [1039., 1040.], [1048., 1053.], [1057., 1058.], [1060., 1061.], [1063., 1064.], [1070., 1071.], [1077., 1081.], [1085., 1087.], [1088., 1095.], [1096., 1098.], [1098., 1099.], [1104., 1108.], [1109., 1112.], [1113.5, 1114.5], [1119., 1121.], [1125., 1128.], [1131., 1133.5], [1134., 1135.], [1136., 1137.], [1139., 1140.], [1141., 1142.], [1143., 1145.], [1147., 1148.], [1152., 1168.], [1170., 1203.], [1204., 1213.], [1215., 1216.]]
        args.contadd = [[1027.27, 32.5539], [1176.72, 18.6022], [1215.53, 37.8522]]

    if item['KOAjobID']==119681: # already adjusted for searchlya
        args.contmask = [[1025., 1026.], [1027., 1028.], [1031., 1033.], [1035., 1036.], [1041., 1046.], [1054., 1057.], [1073., 1075.], [1080., 1081.], [1085., 1086.], [1098., 1100.], [1116., 1117.], [1121., 1122.], [1141., 1143.], [1154., 1160.], [1171., 1175.], [1180., 1181.], [1185., 1186.], [1192., 1193.], [1199., 1200.], [1208., 1209.], [1213., 1214.]]
        args.contadd = [[1027.25, 231.989]]

    if item['KOAjobID']==9075:
        args.contmask = [[1025., 1026.], [1028., 1029.], [1030., 1031.], [1038., 1039.], [1044., 1045], [1046., 1047.], [1051.5, 1052.], [1061., 1062.], [1065., 1066.], [1070., 1072.], [1073., 1078.], [1081., 1082.], [1089., 1097.], [1098., 1100.5], [1102., 1105.], [1108., 1109.], [1110., 1111.], [1112., 1116.], [1119., 1120.], [1123., 1124.], [1127., 1128.], [1129., 1136.], [1137., 1138.], [1140., 1141.], [1145., 1149.], [1151., 1157.], [1159., 1160.], [1185., 1189.], [1193., 1194.], [1197., 1200.], [1209., 1211.], [1215.5, 1216.]]
        args.contadd = [[1185.1, 152.49], [1186.72, 157.194]]

    # voigtfit mask
    if item['KOAjobID']==104297: args.voigtmask = [[]]
    if item['KOAjobID']==125810: args.voigtmask = [[]]
    if item['KOAjobID']==36576: args.voigtmask = [[]]
    if item['KOAjobID']==3105: args.voigtmask = [[]]
    if item['KOAjobID']==20463: args.voigtmask = [[]]
    if item['KOAjobID']==54447: args.voigtmask = [[]]
    if item['KOAjobID']==29524: args.voigtmask = [[]]
    if item['KOAjobID']==2315: args.voigtmask = [[]]
    if item['KOAjobID']==32118: args.voigtmask = [[]]
    if item['KOAjobID']==10408: args.voigtmask = [[]]
    
    # adjust lam by adjust_zqso
    if adjust_zqso!=None and item['KOAjobID']!=119681:
        args.contmask = [su.rest_frame(su.obs_frame(np.array(ls), item['z']), adjust_zqso) for ls in args.contmask]
        args.contadd = [su.rest_frame(su.obs_frame(np.array(ls), item['z']), adjust_zqso) for ls in args.contadd]
        args.voigtmask = [su.rest_frame(su.obs_frame(np.array(ls), item['z']), adjust_zqso) for ls in args.voigtmask]
    return args

if __name__ == '__main__': plot_substitute()

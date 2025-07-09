from utils import *
from astropy.coordinates import SkyCoord, Angle

def match_ps_coord(search_radius=Angle('1 arcsec')):
    '''
    match df_elqs_ps coordinates with df_elqs
    '''
    from read_elqs import df_elqs, df_elqs_ps
    coor_ps = SkyCoord(df_elqs_ps['RA'], df_elqs_ps['DEC'], unit='deg')
    coor_sdss = SkyCoord(df_elqs['RASdeg'], df_elqs['DESdeg'], unit='deg')
    nfound = 0
    for ips, cps in enumerate(coor_ps):
        found = df_elqs[cps.separation(coor_sdss) <= search_radius]
        if len(found) > 0:
            nfound += 1
            df_elqs_ps.at[ips, 'dvdtosig_sdss'] = found['dvdtosig'].to_numpy()
            df_elqs_ps.at[ips, 'rmag_sdss'] = found['rmag'].to_numpy()
            df_elqs_ps.at[ips, 'e_rmag_sdss'] = found['e_rmag'].to_numpy()
            df_elqs_ps.at[ips, 'z_sdss'] = found['z'].to_numpy()
            df_elqs_ps.at[ips, 'M1450_sdss'] = found['M1450'].to_numpy()
            df_elqs.at[found['No'].iloc[0]-1, 'PS'] = ips
            '''
            print(pd.DataFrame(df_elqs_ps.iloc[ips]).T)
            print(found)
            print('----------------------------')
            '''
    # assign KOAjobID, use to crossmatched df_elqs_ps
    df_elqs_matched = df_elqs[~np.isnan(df_elqs.PS)]
    to_assign = ['KOAjobID', 'SDSS', 'use']
    df_elqs_ps.loc[df_elqs_matched['PS'], to_assign] = df_elqs_matched[to_assign].to_numpy()
    df_elqs_ps.loc[df_elqs_matched['PS'], 'Name'] = df_elqs_matched['Name'].to_numpy()
    #print(nfound)

def merge_ps():
    '''
    merge ps-elqs (df_elqs_ps) and elqs (df_elqs) catalog
    if corss-matched, use ps-elqs, except for APM 08279+5255 who use df_elqs's photometry
    '''
    from read_elqs import df_elqs, df_elqs_ps
    match_ps_coord()
    df_elqs = df_elqs.rename(columns={'RASdeg': 'RA', 'DESdeg': 'DEC'})
    cols = ['RA', 'DEC', 'z', 'M1450', 'rmag', 'e_rmag', 'imag', 'KOAjobID', 'SDSS', 'use', 'dvdtosig', 'Name']
    phot = ['M1450', 'rmag', 'e_rmag', 'imag', 'dvdtosig'] # photometry parameters
    keck = ['KOAjobID', 'use'] # keck parameters
    df_merged = pd.concat([df_elqs_ps[cols], df_elqs[np.isnan(df_elqs.PS)][cols]], ignore_index=True)
    # apply keck parameters from df_elqs (crossmatched) to df_merged
    # APM 08279+5255
    apm_merged = df_merged.sort_values('dvdtosig', ascending=False).iloc[0]
    apm_elqs = df_elqs.sort_values('dvdtosig', ascending=False).iloc[0]
    for apm in [apm_merged, apm_elqs]:
        if SkyCoord(apm['RA'], apm['DEC'], unit='deg').separation(SkyCoord(127.923775, 52.754874, unit='deg')) > Angle('1 arcsec'):
            raise Exception('Not found APM 08279+5255')
    for tochange in phot:
        df_merged.loc[apm_merged.name, tochange] = apm_elqs[tochange]
    return df_merged
df_merged = merge_ps()

df_elqs_N = df_elqs[df_elqs.DESdeg >= -20]
df_elqs_ps_N = df_elqs_ps[df_elqs_ps.DEC >= -20]
df_elqs_S = df_elqs[df_elqs.DESdeg <= 20]
df_elqs_ps_S = df_elqs_ps[df_elqs_ps.DEC <= 20]
df_merged_N = df_merged[df_merged.DEC >= -20]
df_merged_S = df_merged[df_merged.DEC <= 20]
top10vds_N_nosub = df_merged_N.sort_values('dvdtosig', ascending=False)[:10]
top10vds_S_nosub = df_merged_S.sort_values('dvdtosig', ascending=False)[:10]
top10m14_N_nosub = df_merged_N.sort_values('M1450')[:10]
top10m14_S_nosub = df_merged_S.sort_values('M1450')[:10]

warnings.filterwarnings("ignore", category=FutureWarning)
def top10_cont_sub_df(top10_nosub):
    '''
    continuum substitution for top10 vds/top5 m14 north, using generate_spec
    '''
    dic = {'APM 08279+5255': 29524, 'PS1 J212540.96-171951.4': 36576, 'SDSS J034151.16+172049.7': 54447, 'PSS J1723+2243': 104297, 'SDSS J161737.78+595020.1': 12850, 'PS1 J052136.92-133938.8': 20463, 'SDSS J164804.84+493326.7': 2315} # {Name: KOAjobID}, if not in dic, use self
    colnames = ['RA', 'DEC', 'z', 'M1450', 'KOAjobID', 'catalog', 'SDSS', 'z_origin', 'M1450_origin', 'Name']
    top10_withsub = pd.DataFrame(columns=colnames)
    #for item in top10_nosub.iloc: # loop through origin targets
    for ind in range(len(top10_nosub)): # loop through origin targets
        item = top10_nosub.iloc[ind]
        name = item['Name']
        if name=='SDSS J095937.11+131215.5': # use sdss continuum
            toappend = pd.DataFrame(item[['RA', 'DEC', 'z', 'M1450', 'KOAjobID', 'SDSS', 'Name']]).T
            toappend.set_index(pd.Index([ind + 1]), inplace=True)
        else:
            if name in dic.keys(): koajobid = dic[name]
            else: koajobid = item['KOAjobID']
            toappend = df_all[['RA', 'DEC', 'z', 'M1450', 'KOAjobID', 'catalog', 'SDSS', 'Name']][df_all['KOAjobID']==koajobid]
            toappend.rename(index={koajobid: ind + 1}, inplace=True) # change index to rank
        toappend['z_origin'] = item['z']
        toappend['M1450_origin'] = item['M1450']
        top10_withsub = pd.concat([top10_withsub, toappend])
    #top10_withsub = top10_withsub.set_index(pd.RangeIndex(1, len(top10_withsub) + 1)) # change index to rank
    return top10_withsub
top10vds_N = top10_cont_sub_df(top10vds_N_nosub)
top10m14_N = top10_cont_sub_df(top10m14_N_nosub)

def plot_mags():
    tocmp = ['dvdtosig', 'rmag', 'e_rmag', 'z', 'M1450']
    labels = [r'$|\dot{v}|/\sigma_\dot{v}$', r'$r$', r'$\sigma_r$', r'$z_\mathrm{QSO}$', r'$M_\mathrm{1450}$']
    for loc in ['_N', '_S']:
        ps = eval('df_elqs_ps' + loc)
        sdss = eval('df_elqs' + loc)
        ps_nomatch = ps[np.isnan(ps.z_sdss)]
        sdss_nomatch = sdss[np.isnan(sdss.PS)]
        for iplot in range(len(tocmp)):
            # SDSS vs Pan-STARRS
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect('equal')
            if tocmp[iplot]=='rmag':
                ax.errorbar(ps[tocmp[iplot]], ps[tocmp[iplot]+'_sdss'], xerr=ps['e_'+tocmp[iplot]], yerr=ps['e_'+tocmp[iplot]+'_sdss'], fmt='k.')
            else: ax.plot(ps[tocmp[iplot]], ps[tocmp[iplot]+'_sdss'], '.k', label='both') # matched
            #ax.plot(ps_nomatch[tocmp[iplot]], ps_nomatch[tocmp[iplot]], '.r', label='Pan-STARRS only') # ps nomatch
            #ax.plot(sdss_nomatch[tocmp[iplot]], sdss_nomatch[tocmp[iplot]], '.b', label='SDSS only') # ps nomatch
            axlim = ax.axis()
            ax.plot([-40, 20], [-40, 20], 'k')
            ax.axis(axlim)
            ax.set_xlabel('Pan-STARRS '+labels[iplot])
            ax.set_ylabel('SDSS '+labels[iplot])
            #ax.legend()
            fig.tight_layout()
            plt.pause(0.1)

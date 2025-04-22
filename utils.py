import sys
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import interpolate
from scipy import stats

def color_codes():
    cols = {}
    cols['unsup'] = [0.46,0, 0.23]
    cols['grating'] = "0.5"
    cols['sup'] = 'g'
    cols['leaf1'] = 'b'
    cols['leaf2'] = 'c'
    cols['leaf3'] = [0.27,0.51,0.71]
    cols['leaf1_swap'] = [0,0.47,0.47]
    cols['circle1'] = 'r'
    cols['circle2'] = 'm'
    return cols

def fmt(ax=None, xtick=None, ytick=None, title=None, xlabel=None, ylabel=None, boxoff=1,  ypad=1, xpad=1, tpad=None,
          axis_off='on', xlm=None, ylm=None, aspect='auto', tcolor='k', ticklen=1.5, tickwid=0.5, y_invert=0):
    ax = plt.gca() if ax==None else ax     
    plt.sca(ax)
    if xtick != None:
        plt.xticks(ticks=xtick[0], labels=xtick[1]) if len(xtick)>1 else plt.xticks(ticks=xtick[0], rotation = xrot)
    if ytick != None:
        plt.yticks(ticks=ytick[0], labels=ytick[1]) if len(ytick)>1 else plt.yticks(ticks=ytick[0])              
    if xlabel!=None:
        plt.xlabel(xlabel, labelpad=xpad)
    if ylabel!=None:
        plt.ylabel(ylabel, labelpad=ypad)
    plt.title(title, color=tcolor, pad=tpad)
    ax.tick_params(axis='both', which='major', length=ticklen, width=tickwid)
    if boxoff == 1:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)       
    elif boxoff == 2:
        for spine in plt.gca().spines.values():
            spine.set_visible(False)   
    elif boxoff==0:
        for spine in plt.gca().spines.values():
            spine.set_visible(True)          
    ax.axis(axis_off)
    plt.xlim(xlm)  
    plt.ylim(ylm)
    ax.set_aspect(aspect)             
    if y_invert:
        ax.invert_yaxis()    

def interp_value(v, vind, tind):
    """v: value; vind: index; tind: target index"""
    Model_ = interpolate.interp1d(vind, v, fill_value='extrapolate')
    return Model_(tind)

def spk_pos_interp(raw_spk=[], accum_pos=[], corridorLen=[], new_shape=[]):
    """raw_spk: neurons x frames
        accum_pos: accumulated position, should be increasement
        corridorLen: length of corrdior
        new_shape: [0]: trials number. [1]: bins for each corridor, if[1]==0 then bins=corridorLen
    """
    if len(new_shape)==2:
        if new_shape[1]==0:
            new_shape[1] = corridorLen
    linPos = np.arange(0, new_shape[0], 1/new_shape[1])
    spk_resh = []
    for s in range(raw_spk.shape[0]): # loop through neurons
        spk_resh.append(np.reshape(interp_value(raw_spk[s, :], accum_pos/corridorLen, linPos), (int(new_shape[0]), int(new_shape[1]))))
    return np.array(spk_resh)  

def get_interpPos_spk(spk, spk_culm_pos, ntrial, n_bins=60, lengths=60, save_path=''):
    interp_spk = np.zeros((spk.shape[0], ntrial, n_bins))
    step_size = 10000 # decrease this if has small RAM
    i=0    
    while i <= spk.shape[0]:
        interp_spk[i:i+step_size,:] = spk_pos_interp(raw_spk = spk[i:i+step_size, :],
                          accum_pos=spk_culm_pos, corridorLen=lengths, new_shape=[ntrial, 0])
        i += step_size
        print(i)  
    if len(save_path)>0:
        np.save(save_path, interp_spk)   
    return interp_spk        

def find_between(x, xmin, xmax):
    '''find x >= xmin and <= xmax'''
    return (x>=xmin) & (x<=xmax)

def get_lick_raster(dat):
    SoundPos = dat['SoundPos']
    LickPos = np.array([dat['LickPos'][dat['LickTrind']==n] for n in range(dat['ntrials'])], dtype=object) 
    RewPos = dat['RewPos']
    isRew = dat['isRew']
    StimTrial = dat['StimTrial']

    UniqW = list(StimTrial.keys())
    LickRaster = {}
    for _, typ in enumerate(['sort_by_cue', 'sort_by_time']):
        LickRaster[typ] = {}
        for _, tw in enumerate(UniqW):
            LickRaster[typ][tw] = {}
            ind = StimTrial[tw]
            issort = np.argsort(SoundPos[ind]) if typ=='sort_by_cue' else np.arange(ind.sum())

            LickRaster[typ][tw]['SoundPos'] = SoundPos[ind][issort]
            LickRaster[typ][tw]['RewPos'] = RewPos[ind][issort]

            temp_lick = LickPos[ind][issort].reshape((1,-1)) if LickPos[issort].ndim<1 else LickPos[ind][issort]
            LPos,LTr,FLPos,FLTr = np.empty(0),np.empty(0), [], []
            for n in range(len(temp_lick)):
                LPos = np.concatenate((LPos, temp_lick[n]))
                LTr = np.concatenate((LTr, n*np.ones(len(temp_lick[n]))))  
                if len(temp_lick[n])>0:
                    FLPos.append(temp_lick[n][0])
                    FLTr.append(n)

            LickRaster[typ][tw]['LickPos'] = LPos
            LickRaster[typ][tw]['LickTr'] = LTr
            LickRaster[typ][tw]['firstLickPos'] = np.array(FLPos)
            LickRaster[typ][tw]['firstLickTr'] = np.array(FLTr)     
            LickRaster[typ][tw]['trN'] = np.arange(ind.sum()) 
            LickRaster[typ][tw]['isRew'] = any(isRew[ind])
    return LickRaster


def lickCount(dats, def_range=[]):
	"""return binary lick response for each trial"""
	befRew, aftRew, inRange = [], [], []
	for n in range(dats['ntrials']):
	    RewTime, SoundTime = dats['RewTime'], dats['SoundTime']
	    cue_del = SoundTime + dats['Reward_Delay_ms']/(1000*3600*24) # convert to datenum 
	    if dats['Reward_Mode']=='Passive':
	        befRew.append(find_between(dats['LickTime'],dats['Trial_start_time'][n],cue_del[n]))
	    elif dats['Reward_Mode']=='Active after cue':
	        befRew.append(find_between(dats['LickTime'],dats['Trial_start_time'][n],dats['Gray_space_time'][n]))
	    aftRew.append(find_between(dats['LickTime'],RewTime[n],dats['Gray_space_time'][n]))
	    if len(def_range)==2:
	        inRange.append(find_between(dats['LickPos'], def_range[0], def_range[1]) & (dats['LickTrind']==n))
	    else:
	        inRange.append([np.nan])

	lickResp = {'befRew':np.array(befRew).sum(1)>0,
	            'aftRew':np.array(aftRew).sum(1)>0,
	            'inRange':np.array(inRange).sum(1)>0}    
	return lickResp

def lick_response(dats, trial_include=[], def_range=[]):
    lickResp= lickCount(dats, def_range=def_range)
    
    Resp = {}
    Resp['tex_name'] = dats['UniqWalls']
    Resp['category'] = []
    Resp['isRew'] = []
    Resp['stim_id'] = dats['stim_id'] if 'stim_id' in dats else None
    ntrials = dats['ntrials']
    trial_include = np.array(trial_include)

    kn = list(lickResp.keys())
    for w,wn in enumerate(Resp['tex_name']):
        Resp[wn] = {}
        for k in range(len(kn)):
            idx = dats['WallName']==wn
            trial_n = np.arange(ntrials)
            if len(trial_include)==2:
                idx = idx & (trial_n>=trial_include[0]) & (trial_n<=trial_include[1])

            Resp[wn][kn[k]] = np.array([np.mean(lickResp[kn[k]][idx]), 
                                        np.std(lickResp[kn[k]][idx], ddof=1)/np.sqrt(idx.sum())])
    return Resp	

def get_mean_lick_response(beh, lick_typ='befRew'):
    """lick_typ: befRew, aftRew or inRange"""
    out = {'mname':list(beh.keys())}
    n = len(out['mname'])
    temp = np.empty((n, 6, 2)) # mice, stimuli, [u, sem];
    for k, ky in enumerate(out['mname']):
        lickResp = lick_response(beh[ky])
        stimuli = np.array(['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf_swap'])
        stim_idx = np.zeros(len(stimuli)).astype(bool)
        for istim in range(6):
            stim = lickResp['tex_name'][lickResp['stim_id'] == istim]
            if len(stim)>0:
                temp[k, istim] = lickResp[stim[0]][lick_typ]
                stim_idx[istim] = True
    out['u_sem'] = temp[:, stim_idx]
    out['stimuli'] = stimuli[stim_idx]
    return out    

def neu_area_ID(iarea):
    area_name = ['V1', 'mHV', 'lHV', 'aHV']
    idx = {}
    for ar in area_name:
        if ar=='V1':
            idx[ar] = iarea==8
        elif ar=='mHV':
            idx[ar] = (iarea==0) | (iarea==1) | (iarea==2) | (iarea==9)
        elif ar=='lHV':
            idx[ar] = (iarea==5) | (iarea==6)
        elif ar=='aHV':
            idx[ar] = (iarea==3) | (iarea==4)     
    return idx

def load_retino(db, root = ''):
    """return: dtrans, areasN, ix, ix_area"""
    dtrans = np.load(os.path.join(root, '%s_%s_trans.npz'%(db['mname'], db['datexp'])), allow_pickle=True)
    areasN = ['All','V1','medial','anterior','lateral']
    ix = neu_area_ID(dtrans['iarea'])    
    out = {'xy_t':dtrans['xy_t'], 'iarea':dtrans['iarea'], 'ar_name':areasN, 'neu_ar_idx':ix}
    return out

def load_spk(db, root=''):
    fn = '%s_%s_%s_neural_data.npy'%(db['mname'],db['datexp'],db['blk'])
    spk_path = os.path.join(root, fn)
    spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']],0)
    return spk   

def load_interp_spk(db, root=''):
    fn = '%s_%s_%s_interpolate_spk.npy'%(db['mname'], db['datexp'], db['blk'])
    spk = np.load(os.path.join(root, fn), allow_pickle=True)    
    return spk

def load_coding_direction(root, fn):
    dat = np.load(os.path.join(root, fn), allow_pickle=1).item() 
    print(dat['notes'])   
    resp1 = dat['proj_2_stim1']
    resp2 = dat['proj_2_stim2']
    resp_diff = np.empty((len(resp1.keys()), 4, 7), dtype=object)
    resp_diff_u = np.empty((len(resp1.keys()), 4, 7, dat['pos_length']), dtype=object)
    resp_diff_corr_u = np.empty((len(resp1.keys()), 4, 7), dtype=object)
    for m, mname in enumerate(resp1.keys()):
        for a, arn in enumerate(resp1[mname].keys()):
            resp_diff[m, a] = np.array(resp1[mname][arn], dtype=object) - np.array(resp2[mname][arn], dtype=object)
            for s in range(7):
                resp_diff_u[m, a, s] = resp_diff[m, a, s].mean(0)
                resp_diff_corr_u[m, a, s] = resp_diff_u[m, a, s, dat['pos_from_prev']:dat['pos_from_prev']+40].mean(-1)    
    out = {'proj_single_trials':resp_diff, 'proj_tr_mean':resp_diff_u.astype(float), 'proj_pos_mean':resp_diff_corr_u.astype(float),
    		'notes':dat['notes'], 'stim_ref':dat['stim_ref'], 'pos_from_prev':dat['pos_from_prev']}
    return out    

def load_example_stimSelNeu(root, fn='TX108_2023_03_22_1_stimSelNeu_sorted.npy'):
    out = np.load(os.path.join(root, 'process_data', fn), allow_pickle=1).item()
    return out    

def dprime(x1, x2):
    """x1,x2: neurons * frames """
    u1, u2 = np.nanmean(x1,1), np.nanmean(x2,1)
    sig1, sig2 = np.nanstd(x1,1), np.nanstd(x2,1)
    return 2 * (u1 - u2) / (sig1 + sig2)      

def get_dist_img(xpos, ypos, sel_idx=None, sig=30, sig_bg=30, xoff=800, yoff=800):
    from scipy import ndimage
    """create an binary image for indicate neurons' position using xpos and ypos
    """
    x, y = xpos+xoff, ypos+yoff
    bg, img = np.zeros((5000,5000)), np.zeros((5000,5000))
    bg[y.astype(int),x.astype(int)] = 1.0
    bg = ndimage.gaussian_filter(bg, sigma=1, truncate=sig_bg)>0 # create a background map
    bg = bg.astype(float)
    bg[bg==False]=np.nan # fill empty space with nan
    
    if any(sel_idx != None):
        img[y[sel_idx].astype(int),x[sel_idx].astype(int)] = 1
    img = ndimage.gaussian_filter(img, sigma=sig)
    img[np.isnan(bg)] = np.nan
    img = np.fliplr(img)
    return img    

# def Get_density_map(dprime, retinotopy, dp_thr=0.3):
#     xoff, yoff =800,800   
#     img_both, img_stim1, img_stim2 = [], [], []
#     nmice = len(dprime)
#     for i in range(nmice):   
#         ixy, arid = retinotopy[i]['xy_t'], retinotopy[i]['iarea']
#         dp = dprime[i]
#         idx_both = abs(dp) >= dp_thr # both 
#         idx_stim1 = dp >= dp_thr # only stim1
#         idx_stim2 = dp <= -dp_thr # only stim2
#         idx_neu = (arid!=-1) & (arid != 7)
#         nneu = idx_neu.sum() # total neurons

#         xpos, ypos = -ixy[idx_neu, 1], ixy[idx_neu, 0]

#         img0 = get_dist_img(xpos, ypos, idx_both[idx_neu], xoff=xoff, yoff=yoff)
#         img_both.append(img0/nneu) # devided by total neurons

#         img1 = get_dist_img(xpos, ypos,idx_stim1[idx_neu], xoff=xoff, yoff=yoff)
#         img_stim1.append(img1/nneu) 

#         img2 = get_dist_img(xpos, ypos, idx_stim2[idx_neu], xoff=xoff, yoff=yoff)
#         img_stim2.append(img2/nneu)     
#     img_both = np.nanmean(np.array(img_both),axis=0) # mean across mice
#     img_stim1 = np.nanmean(np.array(img_stim1),axis=0)
#     img_stim2 = np.nanmean(np.array(img_stim2),axis=0)
    
#     n_down = 10 # down sample ratio
#     imgs = {}
#     imgs['both'] = img_both[::n_down,::n_down]
#     imgs['stim1'] = img_stim1[::n_down,::n_down]
#     imgs['stim2'] = img_stim2[::n_down,::n_down]
#     imgs['n_down'] = n_down    
#     return imgs  

def Get_density_map(dprime, retinotopy, dp_thr=0.3, typ='both', n_down = 10):
    """dp_thr: threshold,
    n_down: down sample ratio"""
    xoff, yoff =800,800   
    img = []
    for i in range(len(dprime)):   
        ixy, arid = retinotopy[i]['xy_t'], retinotopy[i]['iarea']
        dp = dprime[i]
        if typ=='both':
            idx = abs(dp) >= dp_thr
        elif typ=='stim1':
            idx = dp >= dp_thr
        elif typ=='stim2':
            idx = dp <= -dp_thr            
        idx_neu = (arid!=-1) & (arid != 7) # exclude neurons from outside of visual cortex
        nneu = idx_neu.sum() # total neurons
        xpos, ypos = -ixy[idx_neu, 1], ixy[idx_neu, 0]

        img0 = get_dist_img(xpos, ypos, idx[idx_neu], xoff=xoff, yoff=yoff)
        img.append(img0/nneu) # devided by total neurons
    img = np.nanmean(np.array(img),axis=0) # mean across mice
    imgs = {'img': img[::n_down,::n_down], 'n_down':n_down}
    return imgs           

def Get_dprime_selective_neuron(db, Beh, stim_ID=[2, 0], root=''):
    """stim_ID = [2, 0]: 'circle1':0, 'circle2':1, 'leaf1':2, 'leaf2':3'"""
    dp_all, ret_all, mname = [], [], []
    for n, ndb in enumerate(db):
        kn = '%s_%s_%s'%(ndb['mname'], ndb['datexp'], ndb['blk'])
        beh = Beh[kn]
        ret = load_retino(ndb, root=os.path.join(root, 'retinotopy'))
        spk = load_spk(ndb, root=os.path.join(root, 'spk'))
        nfr = spk.shape[1] 
        ntrials, uniqW, WallN, stim_id = beh['ntrials'], beh['UniqWalls'], beh['WallName'], beh['stim_id']
        stim1 = uniqW[stim_id==stim_ID[0]][0]
        stim2 = uniqW[stim_id==stim_ID[1]][0]  
        ft_WallN = beh['ft_WallID'][:nfr]
        isCorridor = beh['ft_CorrSpc'][:nfr]
        VRmove = beh['ft_move'][:nfr]>0
        fr_valid = VRmove & isCorridor # only use activity inside the texture area plus mouse is running (VR moving)
        dp = dprime(spk[:, (ft_WallN==stim1) & fr_valid],
                            spk[:,(ft_WallN==stim2) & fr_valid])
        dp_all.append(dp)
        ret_all.append(ret)    
        print('%d is finished'%(n))
        mname.append(ndb['mname'])
    all_dat = {'dprime':dp_all, 'retinotopy':ret_all, 'mname':mname}
    return all_dat

def Get_selective_neuron_fraction_with_dprime(dprime, retinotopy, dp_thrs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]):
    arN = ['V1', 'mHV', 'lHV', 'aHV']
    frac = {'Notes':"['V1','mHV','lHV','aHV'] * mice * n_thr * [both,stim1,stim2]"}
    frac['dp_thrs'] = dp_thrs
    frac['value'] = np.empty((4, len(dprime), len(dp_thrs), 3))
    for i in range(len(dprime)):
        dp, areaID = dprime[i], retinotopy[i]['neu_ar_idx']
        for a,ar in enumerate(arN):
            ix = areaID[ar]
            for t,thr in enumerate(dp_thrs):
                both = abs(dp[ix])>=thr
                stim1 = dp[ix]>=thr
                stim2 = dp[ix]<=-thr
                frac['value'][a, i, t] =  [both.sum()/ix.sum(), stim1.sum()/ix.sum(), stim2.sum()/ix.sum()]
    return frac

def Get_dprime_rewPred_neuron(db, Beh, stim_dp, root='', load_save_interp_spk=1, interp_spk_path='',
                             dp_thr=0.3):
    frac = {}
    frac['value'] = np.empty((4, len(db)))
    frac['mname'] = []
    frac['dp_thr'] = dp_thrs
    frac['rew_neurons'] = []
    frac['Notes'] = "['V1','mHV','lHV','aHV'] * mice * n_thr"
    for n, ndb in enumerate(db):
        kn = '%s_%s_%s'%(ndb['mname'], ndb['datexp'], ndb['blk'])
        beh = Beh[kn]
        ntrials, uniqW, WallN, stim_id = beh['ntrials'], beh['UniqWalls'], beh['WallName'], beh['stim_id']
        stim = uniqW[stim_id==2][0] # get the name of reward stimulus
        stim_tr = WallN==stim #
        CL = beh['Corridor_Length']
        ft_AcumPos = beh['ft_PosCum']
        VRmove = beh['ft_move']>0
        SoundPos = np.mod(beh['SoundDelPos'], 60) # Sound Pos is highly correlated with reward position
        dp0 = stim_dp['dprime'][n]
        areaID = stim_dp['retinotopy'][n]['neu_ar_idx']
        sel_ind = dp0>=0 # get stimulus selective neurons
        # load interpolated spks
        if load_save_interp_spk:
            print(ndb)
            interp_spk = load_interp_spk(ndb, root=interp_spk_path)
        else:
            spk = load_spk(ndb, root=os.path.join(root, 'spk'))
            nfr = spk.shape[1] 
            interp_spk = get_interpPos_spk(spk[:, VRmove[:nfr]], ft_AcumPos[VRmove][:nfr], 
                                                 ntrials, n_bins=60, lengths=CL)
        print('load interpolated spk..')
        u_spk = interp_spk[:,:,5:40].mean(2) #reward position alway >=5 and <=40  
        dp1 = dprime(u_spk[:, (SoundPos>SoundPos.mean()) & stim_tr],
                          u_spk[:,(SoundPos<=SoundPos.mean()) & stim_tr]) # early vs late cue trials 
        idx1 = (dp1>=dp_thr) & sel_ind
        for a,ar in enumerate(['V1', 'mHV', 'lHV', 'aHV']):
            frac['value'][a, n, t] =  np.sum(idx1 & areaID[ar])/areaID[ar].sum()
        frac['mname'].append(ndb['mname'])
        frac['rew_neurons'].append(idx1)
    return frac    

def Get_coding_direction(db, Beh, stim_ref=[2, 0], prc=5, root='', 
                            load_save_interp_spk=1, interp_spk_path='', n_bef = 10):
    """stim_ref = [2, 0]: 'circle1':0, 'circle2':1, 'leaf1':2, 'leaf2':3, 'leaf3':4, 'leaf1_swap1':5, 'leaf1_swap2':6'
       prc: percentile
       n_bef: append n_bef postion from previous trial
       step1: choose neurons selective to stim_ref with percentile (prc)
       step2: calculate response along coding direction of stim_ref for all stimuli 
     """
    all_stim = ['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']
    spk_proj_1, spk_proj_2 = {}, {}
    for n, ndb in enumerate(db):
        mname, m = ndb['mname'], 1
        while mname in list(spk_proj_1.keys()):  # for a mouse has multiple sessions
            mname = ndb['mname'] + '_%d'%(m)
            m += 1 

        spk_proj_1[mname], spk_proj_2[mname] = {}, {}

        # load retinotopy 
        ret = load_retino(ndb, root=os.path.join(root, 'retinotopy'))
        areas = [ret['neu_ar_idx'][ky] for ky in ret['neu_ar_idx']] # get area index of neurons 
        # load spks
        spk = load_spk(ndb, root=os.path.join(root, 'spk'))
        nneu, nfr = spk.shape         
        # load behavior
        kn = '%s_%s_%s'%(ndb['mname'], ndb['datexp'], ndb['blk'])
        beh = Beh[kn]
        ntrials, uniqW, WallN, stim_id = beh['ntrials'], beh['UniqWalls'], beh['WallName'], beh['stim_id']
        CL = beh['Corridor_Length']
        trInd = np.arange(ntrials)
        ft_trInd = beh['ft_trInd'][:nfr]
        ft_WallN = beh['ft_WallID'][:nfr]
        ft_AcumPos = beh['ft_PosCum'][:nfr]
        VRmove = beh['ft_move'][:nfr]>0
        corr_fr = beh['ft_CorrSpc'][:nfr] & VRmove
        grey_fr = beh['ft_GraySpc'][:nfr] & VRmove    

        if load_save_interp_spk:
            interp_spk = load_interp_spk(ndb, root=interp_spk_path)
            print('load interpolated spk..')
        else:
            interp_spk = get_interpPos_spk(spk[:, VRmove], ft_AcumPos[VRmove], 
                                                 ntrials, n_bins=60, lengths=CL)
        spk_resh = interp_spk.reshape((nneu,-1))

        # use odd trials to get dprime, using the raw frames activity instead of mean activity of each trial
        stim1_fr = (ft_WallN == uniqW[stim_id==stim_ref[0]]) & (ft_trInd % 2 == 0) & corr_fr # odd trials frames
        stim2_fr = (ft_WallN == uniqW[stim_id==stim_ref[1]]) & (ft_trInd % 2 == 0) & corr_fr # odd trials frames
        grey_fr = (ft_trInd % 2==0) & grey_fr # odd trials gray frames
        dp = dprime(spk[:, stim1_fr], spk[:, stim2_fr]) # get dprime 
        # select neurons that response more in the texture area than in the gray space
        corr_neu = (spk[:, stim1_fr].mean(1) > spk[:, grey_fr].mean(1)) | (spk[:, stim2_fr].mean(1) > spk[:, grey_fr].mean(1))

        stim1_tr = WallN==uniqW[stim_id==stim_ref[0]][0] # stim1 trials
        stim2_tr = WallN==uniqW[stim_id==stim_ref[1]][0] # stim2 trials

        corr_spk = interp_spk[:, :, :40].copy() # get activity in the texture area 
        u0 = interp_spk[:,:,42:52].reshape((nneu, -1)).mean(1,keepdims=1) # mean activity in gray space    
        stim1_std = corr_spk[:, stim1_tr].reshape((nneu, -1)).std(1,keepdims=1) # std for stim1
        stim2_std = corr_spk[:, stim2_tr].reshape((nneu, -1)).std(1,keepdims=1) # std for stim2    
        # center to gray space, normalized by std of stim1 and stim2 
        spk_norm = (2 * (spk_resh - u0) / (stim1_std + stim2_std)).reshape(interp_spk.shape)
        # append activity from gray space of the previous trial 
        gray_prev = np.empty((nneu, ntrials, n_bef)) * np.nan
        gray_prev[:, 1:] = spk_norm[:, :-1, 60-n_bef:]
        spk_norm = np.concatenate((gray_prev, spk_norm), axis=2) # concatenate the inter-trial-interval from previous trial    
        print('done normalization..')
        for i, iarn in enumerate(['V1','mHV','lHV','aHV']): # ['V1','mHV','lHV','aHV']
            thrs = np.percentile(dp[corr_neu & areas[i]], [100-prc, prc]) # get dprime value at percentile for corridor neurons and area    
            stim1_idx = (dp >= thrs[0]) & corr_neu & areas[i]
            stim2_idx = (dp <= thrs[1]) & corr_neu & areas[i]    
            spk_proj_1[mname][iarn] = []
            spk_proj_2[mname][iarn] = []

            for j in range(7): #['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']
                if j in stim_ref:
                    # only take even trials for stimuli whhich are used for getting the dprime
                    tr_idx = (WallN==uniqW[stim_id==j]) & (trInd % 2 == 1) 
                else:
                    if np.sum(stim_id==j) > 0: # if stimulus is presented
                        tr_idx = WallN == uniqW[stim_id == j]
                    else:
                        tr_idx = np.zeros(len(WallN)).astype(bool) 
                spk_proj_1[mname][iarn].append(spk_norm[stim1_idx][:, tr_idx].mean(0)) # mean across neurons
                spk_proj_2[mname][iarn].append(spk_norm[stim2_idx][:, tr_idx].mean(0)) # mean across neurons
        print(n)
    all_dat ={}
    all_dat['notes'] = 'coding direction of %s - %s'%(all_stim[stim_ref[0]], all_stim[stim_ref[1]])
    all_dat['proj_2_stim1'] = spk_proj_1
    all_dat['proj_2_stim2'] = spk_proj_2
    all_dat['all_stim'] = np.array(all_stim)
    all_dat['stim_ref'] = stim_ref
    all_dat['pos_length'] = spk_norm.shape[2]
    all_dat['pos_from_prev'] = n_bef    
    return all_dat

def Get_sort_spk(db, Beh, stim_ref=[2, 0], prc=5, root='', load_save_interp_spk=1, interp_spk_path=''):
    """stim_ref = [2, 0]: 'circle1':0, 'circle2':1, 'leaf1':2, 'leaf2':3, 'leaf3':4, 'leaf1_swap1':5, 'leaf1_swap2':6'
       step1: choose neurons selective to stim_ref with percentile (prc), using training trials
       step2: sort neurons based on peak positions inside texture areas, with odd test trials
       step3: get even test trials or trials for other stimuli, with neurons sorted by peak positions from step2
     """
    all_stim = ['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']
    spk_sort = {} # ____________
    for n, ndb in enumerate(db):
        mname, m = ndb['mname'], 1
        while mname in list(spk_sort.keys()):  # for a mouse has multiple sessions
            mname = ndb['mname'] + '_%d'%(m)
            m += 1 
        spk_sort[mname] = {} # ____________
        # load retinotopy 
        ret = load_retino(ndb, root=os.path.join(root, 'retinotopy'))
        areas = [ret['neu_ar_idx'][ky] for ky in ret['neu_ar_idx']] # get area index of neurons 
        # load spks
        spk = load_spk(ndb, root=os.path.join(root, 'spk'))
        nneu, nfr = spk.shape         
        # load behavior
        kn = '%s_%s_%s'%(ndb['mname'], ndb['datexp'], ndb['blk'])
        beh = Beh[kn]
        ntrials, uniqW, WallN, stim_id = beh['ntrials'], beh['UniqWalls'], beh['WallName'], beh['stim_id']
        CL = beh['Corridor_Length']
        trInd = np.arange(ntrials)
        ft_trInd = beh['ft_trInd'][:nfr]
        ft_WallN = beh['ft_WallID'][:nfr]
        ft_AcumPos = beh['ft_PosCum'][:nfr]
        VRmove = beh['ft_move'][:nfr]>0
        corr_fr = beh['ft_CorrSpc'][:nfr] & VRmove
        grey_fr = beh['ft_GraySpc'][:nfr] & VRmove    

        if load_save_interp_spk:
            interp_spk = load_interp_spk(ndb, root=interp_spk_path)
            print('load interpolated spk..')
        else:
            interp_spk = get_interpPos_spk(spk[:, VRmove], ft_AcumPos[VRmove], 
                                                 ntrials, n_bins=60, lengths=CL)
        sizes = interp_spk.shape
        spk_norm = interp_spk.reshape((nneu,-1))
        del interp_spk
        spk_norm = stats.zscore(spk_norm, axis=1).reshape(sizes)
        print('done normalization..')
        # use odd trials as training trials to get dprime, using the raw frames activity instead of mean activity of each trial
        stim1_fr = (ft_WallN == uniqW[stim_id==stim_ref[0]]) & (ft_trInd % 2 == 0) & corr_fr # odd trials frames, 
        stim2_fr = (ft_WallN == uniqW[stim_id==stim_ref[1]]) & (ft_trInd % 2 == 0) & corr_fr # odd trials frames
        dp = dprime(spk[:, stim1_fr], spk[:, stim2_fr]) # get dprime 
        print('get dprime!')
        # select neurons that response more in the texture area than in the gray space
        grey_fr = (ft_trInd % 2==0) & grey_fr # odd trials gray frames
        corr_neu = (spk[:, stim1_fr].mean(1) > spk[:, grey_fr].mean(1)) | (spk[:, stim2_fr].mean(1) > spk[:, grey_fr].mean(1))
        del spk

        for i, iarn in enumerate(['V1','mHV','lHV','aHV']): # ['V1','mHV','lHV','aHV']
            thrs = np.percentile(dp[corr_neu & areas[i]], [100-prc, prc]) # get dprime value at percentile for corridor neurons and area    
            stim1_idx = (dp >= thrs[0]) & corr_neu & areas[i]
            stim2_idx = (dp <= thrs[1]) & corr_neu & areas[i]    
            spk_sort[mname][iarn] = {}
            for r, rstim in enumerate(stim_ref):
                ky_name =  'sorted_by_odd_'+all_stim[rstim]
                spk_sort[mname][iarn][ky_name] = {}

                # index of test trials for reference stimulus
                test_ref_tr = (WallN==uniqW[stim_id==rstim]) & (trInd % 2 == 1)
                if r==0:
                    sel_spk = spk_norm[stim1_idx]
                elif r==1:
                    sel_spk = spk_norm[stim2_idx]
                test_spk = sel_spk[:, test_ref_tr]
                # sort trials based on peak positions of odd test trials
                odd_test = test_spk[:, ::2].mean(1)
                even_test = test_spk[:, 1::2].mean(1)
                sortID, odd_test_mpos = get_neuID_and_sortID(odd_test, max_pos=40)

                spk_sort[mname][iarn][ky_name]['reference'] = odd_test[sortID]
                spk_sort[mname][iarn][ky_name]['reference_maxPos'] = odd_test_mpos
                spk_sort[mname][iarn][ky_name]['target'] = [] 
                spk_sort[mname][iarn][ky_name]['target_maxPos'] = [] 
                for j, jstim in enumerate(all_stim): 
                    if j == rstim:
                        # only take even trials for stimuli whhich are used for getting the dprime
                        spk_sort[mname][iarn][ky_name]['target'].append(even_test[sortID])
                        _, even_test_mpos = get_neuID_and_sortID(even_test, max_pos=40)
                        spk_sort[mname][iarn][ky_name]['target_maxPos'].append(even_test_mpos)
                    else:
                        if np.sum(stim_id==j) > 0: # if stimulus is presented
                            tr_idx2 = WallN == uniqW[stim_id == j]
                        else:
                            tr_idx2 = np.zeros(len(WallN)).astype(bool) 
                        targ_spk = sel_spk[:, tr_idx2].mean(1)
                        spk_sort[mname][iarn][ky_name]['target'].append(targ_spk[sortID])
                        _, targ_mpos = get_neuID_and_sortID(targ_spk, max_pos=40)
                        spk_sort[mname][iarn][ky_name]['target_maxPos'].append(targ_mpos)
        print(n)
    all_dat ={}
    all_dat['notes'] = 'selective neurons of %s and %s'%(all_stim[stim_ref[0]], all_stim[stim_ref[1]])
    all_dat['spk_sort'] = spk_sort
    all_dat['all_stim'] = np.array(all_stim)
    all_dat['stim_ref'] = stim_ref
    all_dat['pos_length'] = spk_norm.shape[2]
    return all_dat    

def get_neuID_and_sortID(spk_in, max_pos=40):
    """spk_in: neu * position"""
    # get position index of peak activity inside corridor
    maxind = np.argsort(spk_in[:, :max_pos], axis=1)[:, -1] 
    sortID = np.argsort(maxind) # sort trials based on the peak positions
    return sortID, maxind   

def get_stimNeu_and_sorted(db, beh, stim_ref=[2, 0], thr=0.3, root='', load_save_interp_spk=1):
    all_stim = ['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']
    spk = load_spk(db, root=os.path.join(root, 'spk'))
    print('done loading spk..')
    neu,nfr = spk.shape
    ntrials, uniqW, WallN, stim_id = beh['ntrials'], beh['UniqWalls'], beh['WallName'], beh['stim_id']
    ft_trInd = beh['ft_trInd'][:nfr]
    ft_WallN = beh['ft_WallID'][:nfr]    
    VRmove = beh['ft_move'][:nfr]>0
    corr_fr = beh['ft_CorrSpc'][:nfr] & VRmove    
    trInd = np.arange(ntrials)
    # use odd trials (training trials) to get dprime, using the raw frames activity 
    stim1_train_fr = (ft_WallN == uniqW[stim_id==stim_ref[0]]) & (ft_trInd%2 == 0) & corr_fr # odd trials frames
    stim2_train_fr = (ft_WallN == uniqW[stim_id==stim_ref[1]]) & (ft_trInd%2 == 0) & corr_fr # odd trials frames
    dp = dprime(spk[:, stim1_train_fr], spk[:, stim2_train_fr]) # get dprime 
    del spk
    if load_save_interp_spk:
        interp_spk = load_interp_spk(db, root=os.path.join(root, 'process_data'))
        print('load interpolated spk..')
    else:
        CL = beh['Corridor_Length']
        ft_AcumPos = beh['ft_PosCum'][:nfr]
        interp_spk = get_interpPos_spk(spk[:, VRmove], ft_AcumPos[VRmove], 
                                             ntrials, n_bins=60, lengths=CL)    
    print('done loading interpolated spk..')
    # take test trials of selective neurons for sorting
    test_tr = trInd%2==1
    stim1_test_tr = (WallN==uniqW[stim_id==stim_ref[0]])[test_tr]
    stim2_test_tr = (WallN==uniqW[stim_id==stim_ref[1]])[test_tr]    
    test_spk = interp_spk[:, test_tr] 
    del interp_spk
    
    gray_u = test_spk[:, :, 40:].reshape((neu,-1)).mean(1,keepdims=1) # get mean of gray space
    norm_spk = test_spk.reshape((neu, -1))
    norm_spk = np.reshape((norm_spk - gray_u) / norm_spk.std(1, keepdims=1), test_spk.shape) # normalize
    
    stim1_neu = norm_spk[dp>=thr] # get spk for stimulus 1 selective neurons
    stim2_neu = norm_spk[dp<=-thr] # get spk for stimulus 2 selective neurons
    
    single_neu, population = {}, {}
    # neurons within single trial activity
    single_neu['stim1Sel_in_stim1']=stim1_neu[300:400, stim1_test_tr]
    single_neu['stim1Sel_in_stim2']=stim1_neu[300:400, stim2_test_tr]
    single_neu['stim2Sel_in_stim1']=stim2_neu[:50, stim1_test_tr]
    single_neu['stim2Sel_in_stim2']=stim2_neu[:50, stim2_test_tr]
    # average across trials
    temp = stim1_neu[:, stim1_test_tr].mean(1)
    stim1_neu_sortID, _ = get_neuID_and_sortID(temp, max_pos=40)
    population['stim1Sel_in_stim1'] = temp
    
    temp = stim1_neu[:, stim2_test_tr].mean(1)   
    population['stim1Sel_in_stim2'] = temp
    
    temp = stim2_neu[:, stim2_test_tr].mean(1)
    stim2_neu_sortID, _ = get_neuID_and_sortID(temp, max_pos=40)     
    population['stim2Sel_in_stim2'] = temp

    temp = stim2_neu[:, stim1_test_tr].mean(1)    
    population['stim2Sel_in_stim1'] = temp
    

    
    all_dat = {'single_neu':single_neu, 'population':population, 'stim1_neu_sortID':stim1_neu_sortID, 'stim2_neu_sortID':stim2_neu_sortID, 'Note':'%s_%s'%(db['mname'], db['datexp'])}
    return all_dat     
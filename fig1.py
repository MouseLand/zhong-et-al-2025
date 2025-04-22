import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import utils


def make_hot_cmap():
    new_hot = cm.get_cmap('magma_r', 256)
    newcolors = new_hot(np.linspace(0, 1, 256))
    noCol = np.array([0, 0, 0, 1])
    return ListedColormap(newcolors[:,:]) 

def lick_raster_plot(ax, lick, show_reward=1, show_firstLick=0, title='', xlm=[0, 60], tcolor='k'):
    plt.sca(ax)
    SoundPos = lick['SoundPos']
    RewPos = lick['RewPos']
    isRew = lick['isRew']
    LickPos = lick['LickPos']
    LickTr = lick['LickTr']
    fLPos = lick['firstLickPos']
    fLkTr = lick['firstLickTr']
    ntrials = SoundPos.shape[0]

    ax.scatter(LickPos, LickTr, marker='o', s=0.5, color='k', alpha=1, linewidth=0)
    ax.scatter(SoundPos, np.arange(ntrials), marker='o', s=2, color='purple', alpha=1, linewidth=0)
    if show_reward & isRew:
        ax.scatter(RewPos, np.arange(ntrials), marker='o', s=2, color='b', alpha=1, linewidth=0)        
    if show_firstLick:
        ax.scatter(fLPos, fLkTr, marker='o', s=1, color='brown', alpha=1)
    ax.axvline(40, lw=0.5, linestyle='--', color='k')
    utils.fmt(ax, xtick=[[0,20,40,60], [0,2,4,6]], ytick=[[0, ntrials]], tcolor=tcolor,
          ylabel='tirals', xlabel='position (m)', title=title, xlm=xlm, ylm=[0, ntrials], y_invert=1, ypad=-10) 

def sup_perf_plot(ax, perf, title='', yn=1, xlm=[-0.3, 1.3]):
    r = perf['u_sem']
    u, sem = r[:, :, 0].mean(0), r[:, :, 0].std(0, ddof=1)/np.sqrt(r.shape[0])
    ax.plot([0, 1], r[:, :, 0].T, 'k-', lw=1, alpha=0.5)
    ax.plot([0, 1], u, 'k-', lw=2)
    ax.errorbar(0, u[0], yerr=sem[0], marker='s', markersize=3, color='r')
    ax.errorbar(1, u[1], yerr=sem[1], marker='s', markersize=3, color='b')
    yln = 'anticipatory licking (%trials)' if yn else ''
    utils.fmt(ax, xtick=[np.arange(len(perf['stimuli'])), perf['stimuli']], ytick=[[0, 0.5, 1], [0, 50, 100]],
          ylabel=yln, title=title, xlm=xlm, ylm=[0, 1])
    xticklabels = ax.get_xticklabels()    
    colors = [utils.color_codes()[stim] for stim in perf['stimuli']]
    for label, color in zip(xticklabels, colors):
        label.set_color(color)

def distribution_map(ax, img, outlines, scal=10, cmp='', vmax=0.6, hlw = 2, alpha=0.4, scalbar=0):
    sz = img.shape[0]
    ax.imshow(np.flipud(img),cmap=cmp,vmax=vmax,extent=[0, sz*scal, 0, sz*scal], rasterized=True)
    temp_outline=[]
    for j in range(10):
        if j!=7:
            temp = outlines[j].copy()
            temp[:,1] = -(-temp[:,1]+800-2500)+2500
            temp[:,0] = temp[:,0]+800
            ax.plot(temp[:,1],temp[:,0],linewidth=0.5,color='k',alpha=alpha)  
            temp_outline.append(temp)
        else:
            temp_outline.append([])
    if scalbar:
        ax.plot([450,1450],[880,880],'k-',lw=1)      
        
    ax.axis('off')
    utils.fmt(ax, y_invert=0, xlm=[200,4500], ylm=[500,4800],axis_off='off', aspect='equal')

def cbar(ax, cmap='gray_r', ticks=[0,1], tickLabel=[], cbarLabel=[], cbarLabelrotation=270, fs_tick=None, fs_label=None, tick_len=1, tick_wid=1, tpad=1, shrink=0.1, orientation='vertical', labelpad=0, outline_color='None'):
    """pos: position of colorbar [x,y,h,w]"""
    cbar = plt.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=ax, ticks=ticks,orientation=orientation,drawedges=False )   
    cbar.ax.tick_params(length=tick_len,width=tick_wid,pad=tpad)
    if any(tickLabel):
        if orientation=='vertical':
            cbar.ax.set_yticklabels(tickLabel)
            for t in cbar.ax.get_yticklabels():
                 t.set_fontsize(fs_tick)  
        elif orientation=='horizontal':
            cbar.ax.set_xticklabels(tickLabel)
            for t in cbar.ax.get_xticklabels():
                 t.set_fontsize(fs_tick)             
    if any(cbarLabel):
        cbar.set_label(cbarLabel,rotation=cbarLabelrotation,fontsize=fs_label, position='bottom', labelpad=labelpad)
    cbar.outline.set_color(outline_color)
    cbar.outline.set_linewidth(0.3) 

def plot_frac(ax, frac1, frac2, col='k', alpha=0.3, mk='s',lw0=1, lw1=2.5, elw=2, fs=None, mks=5,ylm=[-0.001,0.46]):
    frac = np.array([frac1, frac2])
    for i in range(4):
        x = np.array([0, 0.5]) + i
        ax.plot(x, frac[:, i, :], color=col, alpha=alpha, lw=1)
    u, sem = frac.mean(2), frac.std(2, ddof=1)/np.sqrt(frac.shape[2])
    ax.plot([np.arange(4), np.arange(4)+0.5], u, color=col, lw=2)
    ax.errorbar(np.arange(4), u[0, :], yerr=sem[0, :], marker='s', markersize=3, color=col, ls='None')
    ax.errorbar(np.arange(4)+0.5, u[1, :], yerr=sem[1, :], marker='s', markersize=3, color=col, ls='None')  
    utils.fmt(ax, ylm=[0, 0.25], xtick=[np.arange(8)/2, ['before\nlearning', 'after\nlearning', None, None, None, None, None, None]],
             ylabel='% selective neurons', ytick=[[0, 0.1, 0.2], [0, 10, 20]])    

def plot_exampe_stimSel_single_neuron(ax, root='', stim=0, neu_n=27, vmax=3, cols=['r', 'b']):
    out = utils.load_example_stimSelNeu(root)
    if stim==0:
        stim1_tr = out['single_neu']['stim1Sel_in_stim1']
        stim2_tr = out['single_neu']['stim1Sel_in_stim2']
    elif stim==1:
        stim1_tr = out['single_neu']['stim2Sel_in_stim1']
        stim2_tr = out['single_neu']['stim2Sel_in_stim2'] 
    ax[0].imshow(stim1_tr[neu_n], cmap='gray_r', vmin=0, vmax=vmax)
    ax[0].axvline(40, linestyle='--', lw=0.5, color='k')
    ax[0].text(0.8, 0.8, 'stim1', color=cols[0], transform=ax[0].transAxes)
    utils.fmt(ax[0], xlabel='position (m)', ylabel='trials')
    ax[1].imshow(stim2_tr[neu_n], cmap='gray_r', vmin=0, vmax=vmax)
    ax[1].axvline(40, linestyle='--', lw=0.5, color='k')
    ax[1].text(0.8, 0.8, 'stim2', color=cols[1], transform=ax[1].transAxes)
    utils.fmt(ax[1], xlabel='position (m)', xtick=[[0, 20, 40, 60], [0, 2, 4, 6]])     

def plot_exampe_stimSel_population(ax, root='', stim=0,  vmax=1, cols=['r', 'b']):
    out = utils.load_example_stimSelNeu(root)
    if stim==0:
        sortInd = out['stim1_neu_sortID']
        stim1_tr = out['population']['stim1Sel_in_stim1'][sortInd]
        stim2_tr = out['population']['stim1Sel_in_stim2'][sortInd]
    elif stim==1:
        sortInd = out['stim2_neu_sortID']
        stim1_tr = out['population']['stim2Sel_in_stim1'][sortInd]
        stim2_tr = out['population']['stim2Sel_in_stim2'][sortInd]
    ax[0].imshow(stim1_tr, cmap='gray_r', vmin=0, vmax=vmax)
    ax[0].axvline(40, linestyle='--', lw=0.5, color='k')
    utils.fmt(ax[0], xlabel='position (m)', ylabel='neurons', y_invert=1, xtick=[[0, 20, 40, 60], [0, 2, 4, 6]], title='stim1', tcolor=cols[0], ytick=[[]], tpad=0)
    ax[1].imshow(stim2_tr, cmap='gray_r', vmin=0, vmax=vmax)
    ax[1].axvline(40, linestyle='--', lw=0.5, color='k')
    utils.fmt(ax[1], y_invert=1, xtick=[[0, 20, 40, 60], [0, 2, 4, 6]], title='stim2', tcolor=cols[1], ytick=[[]], tpad=0)     

def brain_coordinate(ax, fs=None, lw=1):
    ax.plot([0,0],[-1,1.8],[-0.5,0.5],[0,0],color='k', lw=lw)
    for tex,pos in [('M',[-0.2,0.5]),('L',[1.2,0.5]),('A',[0.5,1.1]),('P',[0.5,-0.1])]:
    	ax.text(pos[0], pos[1], tex, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')
    # text(ax,[('M',[-0.2,0.5]),('L',[1.2,0.5]),('A',[0.5,1.1]),('P',[0.5,-0.1])], fs=fs)
    utils.fmt(ax, xlm=[-0.7,0.7], ylm=[-1.2,1.2], axis_off='off')    
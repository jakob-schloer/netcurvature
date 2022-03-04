"""
Reproducing figures of the paper.
"""
# %%
import copy
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sn
import cartopy.crs as ccrs
from importlib import reload
from climnet.dataset import AnomalyDataset
from climnet.network import clim_networkx
import climnet.plots as cplt
import climnet.utils.spatial_utils as sputils
import utils as ut

PATH = os.path.dirname(os.path.abspath(__file__))
# plt.style.use('paperplot.mplstyle')


def make_argparser():
    """Set parameters for function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--dataset", default=None, type=str,
                        help=("Path to t2m dataset file regridded on equidistant grid."))
    parser.add_argument("-ep", "--epNetwork", default=None, type=str,
                        help=("Filepath of EP network .graphml."))
    parser.add_argument("-cp", "--cpNetwork", default=None, type=str,
                        help=("Filepath of CP network .graphml."))
    parser.add_argument("-normal", "--normalNetwork", default=None, type=str,
                        help=("Filepath of normal network .graphml."))
    return parser


# True to run with ipython
if False:
    class Args():
        """Workaround of argsparser."""

        def __init__(self) -> None:
            self.dataset = PATH + "/../data/t2m_fekete_grid_2.5_1950-2020.nc"
            self.epNetwork = PATH + \
                '/../outputs/t2m_1950-2020_nino_nets/t2m_Nino_EP_fekete_2.5_spearman_twosided_de_0.02_weighted_lb_2_nx.graphml'
            self.cpNetwork = PATH + \
                '/../outputs/t2m_1950-2020_nino_nets/t2m_Nino_CP_fekete_2.5_spearman_twosided_de_0.02_weighted_lb_2_nx.graphml'
            self.normalNetwork = PATH + \
                '/../outputs/t2m_1950-2020_nino_nets/t2m_standard_fekete_2.5_spearman_twosided_de_0.02_weighted_lb_2_nx.graphml'
    args = Args()
else:
    parser = make_argparser()
    args = parser.parse_args()
# %%
# Load dataset corresponding to the networks
ds = AnomalyDataset(load_nc=args.dataset)
# %%
# Load Enso Index
fname = PATH + "/../data/ersst5.nino.mth.91-20.ascii"
nino_indices = ut.get_nino_indices(
    fname, time_range=[ds.ds.time[0].data, ds.ds.time[-1].data], time_roll=0)
enso_dict = ut.get_enso_years(nino_indices, month_range=[12, 2],
                              mean=True, threshold=0.5,
                              min_diff=0.1,
                              drop_volcano_year=False)
# %%
# Load EP, CP, standard network
nino_networks = [
    {'name': 'standard',
     'title': 'Normal',
     'file': args.normalNetwork},
    {'name': 'Nino_EP',
     'title': 'EP',
     'file': args.epNetwork},
    {'name': 'Nino_CP',
     'title': 'CP',
     'file': args.cpNetwork},
]
for net in nino_networks:
    # naming of net files
    season_type = net['name']

    time_period = enso_dict[season_type]
    ds_tmp = copy.deepcopy(ds)
    ds_tmp.use_time_snippets(time_period)

    # Load networks
    net['cnx'] = clim_networkx.Clim_NetworkX(dataset=ds_tmp,
                                             nx_path_file=net['file']
                                             )
    # Normalize curvature
    net['cnx'].normalize_edge_attr(attributes=['formanCurvature'])

    # Get quantiles
    q_vals = [0.1, 0.9]
    net['cnx'].get_node_attr_q(
        edge_attrs=['formanCurvature',
                    'formanCurvature_norm'],
        q_values=q_vals, norm=True
    )

    # Get link length distribution of quantiles
    ll = {}
    for q in [None]+q_vals:
        print(f'Compute link length q={q}')
        link_length = net['cnx'].get_link_length_distribution(
            q=q, var='formanCurvature')
        ll[q] = link_length
    net['ll'] = ll

# %%
##########################################################################################
# Figure 2
##########################################################################################
figsize = (11, 5)
ncols = 3
nrows = 3
num_links = 10
var = 'formanCurvature'
q_vals = [None, 0.9, 0.1]
central_longitude = 0

fig = plt.figure(figsize=(figsize[0]*ncols, figsize[1]*nrows))
gs = fig.add_gridspec(nrows, ncols+1,
                      height_ratios=[20, 20, 20],
                      width_ratios=[15, 15, 15, 1],
                      hspace=.4, wspace=0.1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0],
                      projection=ccrs.EqualEarth(central_longitude=central_longitude))
ax7 = fig.add_subplot(gs[2, 0],
                      projection=ccrs.EqualEarth(central_longitude=central_longitude))
ax5 = fig.add_subplot(gs[1, 1],
                      projection=ccrs.EqualEarth(central_longitude=central_longitude))
ax8 = fig.add_subplot(gs[2, 1],
                      projection=ccrs.EqualEarth(central_longitude=central_longitude))
ax6 = fig.add_subplot(gs[1, 2],
                      projection=ccrs.EqualEarth(central_longitude=central_longitude))
ax9 = fig.add_subplot(gs[2, 2],
                      projection=ccrs.EqualEarth(central_longitude=central_longitude))

axes = np.array([[ax1, ax4, ax7],
                 [ax2, ax5, ax8],
                 [ax3, ax6, ax9]])

cmaps = ['Reds', 'Blues']
mrks = ['o', 'x']
colors = ['r', 'b']
labels = ['All', f'q>{q_vals[1]}', f'q<{q_vals[2]}']
for idx, nino in enumerate(nino_networks):
    axs = axes[idx]
    cnx = nino['cnx']
    season_type = nino['name']
    np_net_files = []
    link_length_0 = nino['ll'][q_vals[0]]
    link_length_pos = nino['ll'][q_vals[1]]
    link_length_neg = nino['ll'][q_vals[2]]

    link_lengths = [
        link_length_0,
        link_length_pos,
        link_length_neg,
    ]
    for i, q in enumerate(q_vals[1:]):
        el = cnx.get_q_edge_list(var, q=q)[0::num_links]
        link_map = cnx.get_el_map(el=el, binary=False)
        cplt.plot_edges(ds, list(el),
                        ax=axs[i+1],
                        significant_mask=True,
                        projection='EqualEarth',
                        central_longitude=central_longitude,
                        plt_grid=True,
                        lw=0.08,
                        alpha=.7,
                        color=colors[i])

        im_nino = cplt.plot_map(ds, link_map,
                                projection='EqualEarth', plt_grid=True,
                                ax=axs[i+1],
                                plot_type='scatter',
                                significant_mask=True,
                                cmap=cmaps[i], levels=2,
                                vmin=0, vmax=3e1,
                                marker='o',
                                title=' ',
                                bar=False,
                                alpha=.5,
                                size=5,
                                fillstyle='none',
                                central_longitude=central_longitude)
        if idx == 0:
            ax_cbar = fig.add_subplot(gs[i+1, 3], )
            cbar = plt.colorbar(im_nino['im'], cax=ax_cbar, orientation='vertical',
                                label=f'# Links to node ({labels[i+1]})',
                                )
    if idx == 0:
        ylabel = 'counts'
        yticks = True
    else:
        ylabel = None
        yticks = False
    cplt.plot_hist(link_lengths,
                   ax=axs[0],
                   density=False,
                   #    nbins=100,
                   bw=150,  # km bin width
                   xlim=(0.1, 2.1e4),
                   ylim=(1e1, 1e5),
                   log=False,
                   ylog=True,
                   bar=False,
                   label_arr=labels,
                   xlabel='link length [km]',
                   ylabel=ylabel,
                   figsize=(9, 5),
                   loc='upper right',
                   color_arr=['k', 'tab:red', 'tab:blue'],
                   yticks=yticks,
                   title=nino['title'],
                   sci=3)

cplt.enumerate_subplots(axes.T, pos_x=-0.1, pos_y=1.07)

# %%
##########################################################################################
# Figure 3
##########################################################################################
m_name = 'norm'
net_measures = {
    'standard': [dict(var='formanCurvature_norm_q0.9',
                      vmin=.5,
                      vmax=.6,
                      vmin_zonal=.42,
                      vmax_zonal=.55,
                      cmap='Reds', label=r"$\tilde{f}_i^+$"),
                 dict(var='formanCurvature_norm_q0.1',
                      vmin=-.25,
                      vmax=-.07,
                      vmin_zonal=None,
                      vmax_zonal=.0,
                      cmap='Blues_r', label=r"$\tilde{f}_i^-$"),
                 ],
    'Nino_EP': [dict(var='formanCurvature_norm_q0.9',
                     vmin=.23,
                     vmax=.37,
                     vmin_zonal=None,
                     vmax_zonal=None,
                     cmap='Reds', label=r"$\tilde{f}_i^+$"),
                dict(var='formanCurvature_norm_q0.1',
                     vmin=-.5,
                     vmax=-.4,
                     vmin_zonal=None,
                     vmax_zonal=None,
                     cmap='Blues_r', label=r"$\tilde{f}_i^-$"),
                ],
    'Nino_CP': [dict(var='formanCurvature_norm_q0.9',
                     vmin=.53,
                     vmax=.63,
                     vmin_zonal=None,
                     vmax_zonal=None,
                     cmap='Reds', label=r"$\tilde{f}_i^+$"),
                dict(var='formanCurvature_norm_q0.1',
                     vmin=-.26,
                     vmax=-.16,
                     vmin_zonal=-.3,
                     vmax_zonal=None,
                     cmap='Blues_r', label=r"$\tilde{f}_i^-$"),
                ],
}


fig = plt.figure(figsize=(len(net_measures)*7, 15))
gs = gridspec.GridSpec(3, len(net_measures),
                       height_ratios=[3, 3, 4],
                       hspace=0.3, wspace=0.3)
axs = []
clrs = ['darkred', 'darkblue']
for j, nino in enumerate(nino_networks):
    net_measure = net_measures[nino['name']]
    axlat1 = fig.add_subplot(gs[-1, j])
    axlat1 = cplt.prepare_axis(axlat1,
                               xlabel=r'zonal median',
                               ylabel='latitude',
                               xlabel_pos='right',
                               )
    divider = make_axes_locatable(axlat1)
    axlat2 = divider.new_horizontal(size="100%", pad=0.2)
    fig.add_axes(axlat2)

    for i, m in enumerate(net_measure):
        axmap = fig.add_subplot(gs[i, j],
                                projection=ccrs.EqualEarth(central_longitude=180))
        cplt.plot_map(ds, nino['cnx'].ds_nx[m['var']],
                      ax=axmap,
                      plot_type='contourf',
                      cmap=m['cmap'], label=f"node curvature {m['label']}",
                      projection='EqualEarth',
                      plt_grid=True,
                      significant_mask=True,
                      levels=8,
                      tick_step=2,
                      vmin=m['vmin'],
                      vmax=m['vmax'],
                      sci=None,
                      round_dec=3,
                      central_longitude=180,
                      orientation='horizontal',
                      pad='20%'
                      )
        axs.append(axmap)

        da = sputils.interp_fib2gaus(
            nino['cnx'].ds_nx[m['var']],
            grid_step=ds.grid_step
        )
        if i == 0:
            da = xr.where(~np.isnan(da), da, da.min())
            axmap.set_title(nino['title'])
        else:
            da = xr.where(~np.isnan(da), da, da.max())
        zonal_mean = sputils.compute_meridional_quantile(da, q=0.5)
        zonal_mean, zonal_std = sputils.compute_meridional_mean(da)
        zonal_low_quantile = sputils.compute_meridional_quantile(da, q=0.25)
        zonal_up_quantile = sputils.compute_meridional_quantile(da, q=0.75)

        if i == 0:
            cplt.prepare_axis(axlat2)
            im = axlat2.plot(zonal_mean, zonal_mean['lat'],
                             color=clrs[i], label=f"{m['label']}")
            axlat2.fill_betweenx(zonal_mean['lat'],
                                 zonal_mean - zonal_std/2,
                                 zonal_mean + zonal_std/2,
                                 color=im[0].get_color(),
                                 alpha=0.5)
            axlat2.set_xlim(m['vmin_zonal'], m['vmax_zonal'])
            cplt.place_legend(axlat2,
                              fontsize=14,
                              loc='upper right',
                              )
        else:
            im = axlat1.plot(zonal_mean, zonal_mean['lat'],
                             color=clrs[i], label=f"{m['label']}")
            axlat1.fill_betweenx(zonal_mean['lat'],
                                 zonal_mean - zonal_std/2,
                                 zonal_mean + zonal_std/2,
                                 color=im[0].get_color(),
                                 alpha=0.5)

            axlat1.set_xlim(m['vmin_zonal'], m['vmax_zonal'])
            cplt.place_legend(axlat1,
                              fontsize=14,
                              loc='upper left',
                              )

            # hide the spines between ax and axlat2
            axlat1.spines['right'].set_visible(False)
            axlat2.spines['left'].set_visible(False)
            axlat1.yaxis.tick_left()
            axlat1.tick_params(labelright=False, right=False)
            axlat2.tick_params(labelleft=False, left=False)

            d = .015  # how big to make the diagonal lines in axes coordinates
            # arguments to pass plot, just so we don't keep repeating them
            kwargs_ax = dict(transform=axlat1.transAxes,
                             color='k', clip_on=False)
            axlat1.plot((1-d, 1+d), (-d, +d), **kwargs_ax)
            # axlat.plot((1-d, 1+d), (1-d, 1+d), **kwargs_ax)

            # switch to the bottom axes
            kwargs_ax.update(transform=axlat2.transAxes)
            axlat2.plot((-d, +d), (-d, +d), **kwargs_ax)
            # axlat2.plot((-d, +d), (1-d, 1+d), **kwargs_ax)

        axlat1.set_yticks([-60, -30, 0, 30, 60])
        axlat1.set_yticklabels([r'$60^\circ$S', r'$30^\circ$S', r'$0^\circ$',
                                r'$30^\circ$N', r'$60^\circ$N'])
    axs.append(axlat1)

cplt.enumerate_subplots(np.array(axs).reshape(
    len(net_measures), 3).T, pos_x=-0.1, pos_y=1.08)

# %%
##########################################################################################
# Fig 4. Combine regions in one plot
##########################################################################################
q = 0.1
attribute = f'formanCurvature'
locations = [
    dict(name='NINO3', lon=[-150, -90], lat=[-5, 5],
         central_lon=180, link_step=4, vmax=1E2),
    dict(name='NINO4', lon=[160, -150], lat=[-5, 5],
         central_lon=180, link_step=1, vmax=1E2),
    dict(name='IO', lon=[60, 100], lat=[-20, 10],
         central_lon=100, link_step=5, vmax=1E2),
    dict(name='Labrador sea', lon=[-100, -40], lat=[50, 80],
         central_lon=-40, link_step=2, vmax=1E2),
]

# create fig.
colors = ['darkmagenta', 'darkgreen']
cmap = ['RdPu', 'Greens']
mrks = ['o', 'x']
binary = False
alpha = 0.5
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[20, 20, 1],
                      hspace=0.1, wspace=0.1)
axs = []

for i, loc in enumerate(locations):
    ax = fig.add_subplot(
        gs[int(i/2), int(i % 2)],
        projection=ccrs.EqualEarth(central_longitude=loc['central_lon'])
    )
    ax.set_global()
    axs.append(ax)

    for j, nino in enumerate(nino_networks[:2]):
        edge_dic = nino['cnx'].get_edges_nodes_for_region(
            lon_range=loc['lon'], lat_range=loc['lat'],
            attribute=attribute, q=q,
            binary=binary)

        if len(edge_dic['el']) == 0:
            continue

        attr = attribute if q is None else f"{attribute}_q{q}"

        # Plot nodes where edges go to
        if binary:
            im = cplt.plot_map(ds, edge_dic['target_map'],
                               ax=ax,
                               label=f'log(# Links)',
                               projection='EqualEarth', plt_grid=True,
                               plot_type='points', significant_mask=True,
                               marker=mrks[j],
                               size=2, color=colors[j],
                               fillstyle='none',
                               bar=False,
                               alpha=alpha,
                               )
        else:
            im = cplt.plot_map(ds, edge_dic['target_map'],
                               ax=ax,
                               label=f'# Teleconnections',
                               projection='EqualEarth', plt_grid=True,
                               plot_type='scatter', significant_mask=True,
                               cmap=cmap[j],
                               vmin=0, vmax=loc['vmax'],
                               bar=False,
                               alpha=alpha,
                               marker=mrks[j], size=10, fillstyle='none',
                               )
            if j == 0:
                im_ep = im
            elif j == 1:
                im_cp = im

        im = cplt.plot_edges(ds, edge_dic['el'][0::loc['link_step']],
                             ax=im['ax'],
                             significant_mask=True,
                             orientation='vertical',
                             projection='EqualEarth',
                             plt_grid=True,
                             lw=0.1,
                             alpha=0.6,
                             color=colors[j],
                             )
    cplt.plot_rectangle(ax=im['ax'],
                        lon_range=loc['lon'],
                        lat_range=loc['lat'],
                        color='k',
                        lw=3)

# Colorbar
ax = fig.add_subplot(gs[-1, 0])
cbar = plt.colorbar(im_ep['im'], cax=ax, orientation='horizontal',
                    shrink=0.8, label='# Teleconnections (EP)')
ax = fig.add_subplot(gs[-1, 1])
cbar = plt.colorbar(im_cp['im'], cax=ax, orientation='horizontal',
                    shrink=0.8, label='# Teleconnections (CP)')

cplt.enumerate_subplots(np.array(axs), fontsize=24)
plt.tight_layout()

# %%
##########################################################################################
# Figure 5
# Linear Regression Coefficient analysis
##########################################################################################

reload(cplt)
reload(ut)

attribute = f'formanCurvature'
locations = [
    # dict(name='Eastern Pacific', lon=[-145, -80], lat=[-10, 10],
    #      central_lon=180, link_step=7, vmax=1E2),
    # dict(name='Central Pacific', lon=[160, -145], lat=[-10, 10],
    #      central_lon=180, link_step=1, vmax=1E2),
    dict(name=rf'$Ni\~no$ 3', lon=[-150, -90], lat=[-5, 5],
         central_lon=180, link_step=4, vmax=1E2),
    dict(name=rf'$Ni\~no$ 4', lon=[160, -150], lat=[-5, 5],
         central_lon=180, link_step=1, vmax=1E2),
    dict(name='Indian Ocean', lon=[60, 100], lat=[-20, 10],
         central_lon=100, link_step=5, vmax=1E2),
    # dict(name='PO', lon=[-170, -120], lat=[40, 70],
    #      central_lon=180, link_step=3, vmax=1E2),
    dict(name='Labrador Sea', lon=[-100, -40], lat=[50, 80],
         central_lon=-40, link_step=2, vmax=1E2),
    # dict(name='Northern Atlantic', lon=[-10, 40], lat=[50, 70],
    #      central_lon=0, link_step=3, vmax=1E2),
]

method = 'rank'
nrows = 4
ncols = 2
im = cplt.create_multi_map_plot_gs(nrows=nrows, ncols=ncols,
                                   central_longitude=180,
                                   figsize=(9, 6),
                                   orientation='horizontal',
                                   )

lr_range = .4
# for i, loc in enumerate(locations):
for i, loc in enumerate(locations):
    for j, nino in enumerate([nino_networks[0],
                              nino_networks[1]]):
        link_dict = nino['cnx'].get_edges_nodes_for_region(
            lon_range=loc['lon'], lat_range=loc['lat'],
        )
        var = 'anomalies'
        sids = link_dict['sids'][:]

        da_lr = ut.get_LR_map(ds=nino['cnx'].ds,
                              var=var,
                              sids=sids,
                              method=method)
        im_comp = cplt.plot_map(ds,
                                da_lr.mean(dim='sids'),
                                ax=im['ax'][i*ncols + j],
                                plot_type='contourf',
                                cmap='RdBu_r',
                                title=f'{loc["name"]} ({nino["name"]})',
                                projection='EqualEarth',
                                plt_grid=True,
                                significant_mask=False,
                                levels=14,
                                tick_step=2,
                                round_dec=2,
                                vmin=-lr_range,
                                vmax=lr_range,
                                bar=False,
                                )

        cplt.plot_rectangle(ax=im['ax'][i*ncols + j],
                            lon_range=loc['lon'],
                            lat_range=loc['lat'],
                            color='k',
                            lw=4)

label = f'{method}-normalized Linear Regression coefficient'
cbar = cplt.make_colorbar_discrete(ax=im['ax'][-1],
                                   im=im_comp['im'],
                                   set_cax=False,
                                   orientation='horizontal',
                                   vmin=-lr_range,
                                   vmax=lr_range,
                                   label=label,
                                   extend='both',
                                   round_dec=2,
                                   )


# %%
#######################################################
# Figures for Supporting Information
#######################################################
net_measures = {
    'degree': dict(vmin=0, vmax=None, cmap='Oranges', label="Deg(v)"),
    'betweenness': dict(vmin=0, vmax=0.003, cmap='Greys', label=f"log(BC(v)) + 1"),
    'clustering': dict(vmin=0.3, vmax=0.7, cmap='Greens', label="Clust. Coeff."),
    'formanCurvature_norm': dict(vmin=None, vmax=None,
                                 cmap='coolwarm', label=f"Forman Curv. norm"),
    # 'formanCurvature': dict(vmin=-4e2, vmax=1E2,
    #                         cmap='coolwarm', label=f"Forman Curv."),
    # 'formanCurvature_rank': dict(vmin=0, vmax=1,
    #                              cmap='coolwarm', label=f"Forman Curv. rank"),
    # 'formanCurvature_q0.9': dict(vmin=0, vmax=10,
    #                              cmap='Reds', label=f"Forman Curv. q>0.9"),
    # 'formanCurvature_q0.8': dict(vmin=0, vmax=10,
    #                              cmap='Reds', label=f"Forman Curv. q>0.8"),
    # 'formanCurvature_q0.2': dict(vmin=-100, vmax=0,
    #                              cmap='Blues_r', label=f"Forman Curv. q<0.2"),
    # 'formanCurvature_q0.1': dict(vmin=-100, vmax=0,
    #                              cmap='Blues_r', label=f"Forman Curv. q<0.1"),
}
for var, m in net_measures.items():
    ncols = 2
    nrows = 1
    im = cplt.create_multi_map_plot(nrows=nrows,
                                    ncols=ncols)
    for j, nino in enumerate(nino_networks[:2]):
        cnx = nino['cnx']
        name = nino['name']
        cplt.plot_map(ds,
                      cnx.ds_nx[var],
                      ax=im['ax'][j],
                      plot_type='contourf',
                      cmap=m['cmap'],
                      label=m['label'],
                      title=f'{name} {var}',
                      projection='EqualEarth',
                      plt_grid=True,
                      significant_mask=True,
                      levels=14,
                      tick_step=2,
                      round_dec=3,
                      vmin=m['vmin'], vmax=m['vmax'],
                      central_longitude=180,
                      )
        plt.tight_layout()
        cplt.enumerate_subplots(im['ax'])
# %%
# Forman Curvature on same scales and hist plots
######################################################
vmin_pos = 0.2
vmax_pos = 0.6
vmin_neg = -0.55
vmax_neg = -0.01

m_name = 'norm'
net_measures = {
    'standard': [dict(var='formanCurvature_norm_q0.9',
                      vmin=vmin_pos,
                      vmax=vmax_pos,
                      vmin_zonal=vmin_pos,
                      vmax_zonal=vmax_pos,
                      cmap='Reds', label=r"$F(v)$ | q>0.9"),
                 dict(var='formanCurvature_norm_q0.1',
                      vmin=vmin_neg,
                      vmax=vmax_neg,
                      vmin_zonal=vmin_neg,
                      vmax_zonal=vmax_neg,
                      cmap='Blues_r', label=r"$F(v)$ | q<0.1"),
                 ],
    'Nino_EP': [dict(var='formanCurvature_norm_q0.9',
                     vmin=vmin_pos,
                     vmax=vmax_pos,
                     vmin_zonal=vmin_pos,
                     vmax_zonal=vmax_pos,
                     cmap='Reds', label=r"$F(v)$ | q>0.9"),
                dict(var='formanCurvature_norm_q0.1',
                     vmin=vmin_neg,
                     vmax=vmax_neg,
                     vmin_zonal=vmin_neg,
                     vmax_zonal=vmax_neg,
                     cmap='Blues_r', label=r"$F(v)$ | q<0.1"),
                ],
    'Nino_CP': [dict(var='formanCurvature_norm_q0.9',
                     vmin=vmin_pos,
                     vmax=vmax_pos,
                     vmin_zonal=vmin_pos,
                     vmax_zonal=vmax_pos,
                     cmap='Reds', label=r"$F(v)$ | q>0.9"),
                dict(var='formanCurvature_norm_q0.1',
                     vmin=vmin_neg,
                     vmax=vmax_neg,
                     vmin_zonal=vmin_neg,
                     vmax_zonal=vmax_neg,
                     cmap='Blues_r', label=r"$F(v)$ | q<0.1"),
                ],
}


fig = plt.figure(figsize=(len(net_measures)*7, 20))
gs = gridspec.GridSpec(4, len(net_measures),
                       height_ratios=[3, 3, 3, 3],
                       hspace=0.35, wspace=0.3)
axs = []
clrs = ['darkred', 'darkblue']
for j, nino in enumerate(nino_networks):
    net_measure = net_measures[nino['name']]
    axlat1 = fig.add_subplot(gs[2, j])
    axlat1 = cplt.prepare_axis(axlat1,
                               xlabel=r'zonal mean $F(v)$',
                               ylabel='latitude',
                               xlabel_pos='right',
                               )
    divider = make_axes_locatable(axlat1)
    axlat2 = divider.new_horizontal(size="100%", pad=0.2)
    fig.add_axes(axlat2)

    for i, m in enumerate(net_measure):
        # Plot curvature map
        axmap = fig.add_subplot(gs[i, j],
                                projection=ccrs.EqualEarth(central_longitude=180))
        cplt.plot_map(ds, nino['cnx'].ds_nx[m['var']],
                      ax=axmap,
                      plot_type='contourf',
                      cmap=m['cmap'], label=m['label'],
                      projection='EqualEarth',
                      plt_grid=True,
                      significant_mask=True,
                      levels=10,
                      tick_step=2,
                      vmin=m['vmin'],
                      vmax=m['vmax'],
                      sci=None,
                      round_dec=3,
                      central_longitude=180,
                      orientation='horizontal',
                      pad='20%'
                      )
        axs.append(axmap)

        # Zonal mean
        da = sputils.interp_fib2gaus(
            nino['cnx'].ds_nx[m['var']],
            grid_step=ds.grid_step
        )
        if i == 0:
            da = xr.where(~np.isnan(da), da, da.min())
            axmap.set_title(nino['title'])
        else:
            da = xr.where(~np.isnan(da), da, da.max())
        zonal_mean = sputils.compute_meridional_quantile(da, q=0.5)
        zonal_mean, zonal_std = sputils.compute_meridional_mean(da)
        zonal_low_quantile = sputils.compute_meridional_quantile(da, q=0.25)
        zonal_up_quantile = sputils.compute_meridional_quantile(da, q=0.75)

        if i == 0:
            cplt.prepare_axis(axlat2)
            im = axlat2.plot(zonal_mean, zonal_mean['lat'],
                             color=clrs[i], label=f"{m['label']}")
            axlat2.fill_betweenx(zonal_mean['lat'],
                                 zonal_mean - zonal_std/2,
                                 zonal_mean + zonal_std/2,
                                 color=im[0].get_color(),
                                 alpha=0.5)
            axlat2.set_xlim(m['vmin_zonal'], m['vmax_zonal'])
            cplt.place_legend(axlat2,
                              fontsize=14,
                              loc='upper right',
                              )
        else:
            im = axlat1.plot(zonal_mean, zonal_mean['lat'],
                             color=clrs[i], label=f"{m['label']}")
            axlat1.fill_betweenx(zonal_mean['lat'],
                                 zonal_mean - zonal_std/2,
                                 zonal_mean + zonal_std/2,
                                 color=im[0].get_color(),
                                 alpha=0.5)

            axlat1.set_xlim(m['vmin_zonal'], m['vmax_zonal'])
            cplt.place_legend(axlat1,
                              fontsize=14,
                              loc='upper left',
                              )

            # hide the spines between ax and axlat2
            axlat1.spines['right'].set_visible(False)
            axlat2.spines['left'].set_visible(False)
            axlat1.yaxis.tick_left()
            axlat1.tick_params(labelright=False, right=False)
            axlat2.tick_params(labelleft=False, left=False)

            d = .015  # how big to make the diagonal lines in axes coordinates
            # arguments to pass plot, just so we don't keep repeating them
            kwargs_ax = dict(transform=axlat1.transAxes,
                             color='k', clip_on=False)
            axlat1.plot((1-d, 1+d), (-d, +d), **kwargs_ax)
            # axlat.plot((1-d, 1+d), (1-d, 1+d), **kwargs_ax)

            # switch to the bottom axes
            kwargs_ax.update(transform=axlat2.transAxes)
            axlat2.plot((-d, +d), (-d, +d), **kwargs_ax)
            # axlat2.plot((-d, +d), (1-d, 1+d), **kwargs_ax)

        axlat1.set_yticks([-60, -30, 0, 30, 60])
        axlat1.set_yticklabels([r'$60^\circ$S', r'$30^\circ$S', r'$0^\circ$',
                                r'$30^\circ$N', r'$60^\circ$N'])
    axs.append(axlat1)

    # histograms
    axhist = fig.add_subplot(gs[3, j])
    sn.histplot(nino['cnx'].ds_nx['formanCurvature_norm'],
                ax=axhist, stat='density')
    # add percentile borders
    axhist.vlines(
        [np.min(nino['cnx'].ds_nx['formanCurvature_norm_q0.9']),
         np.max(nino['cnx'].ds_nx['formanCurvature_norm_q0.1'])],
        ymin=0, ymax=7, color='k'
    )
    axhist.set_xlim([-1, 1])
    axhist.set_ylim([0, 7])
    axhist.set_xlabel(r'$F(v)$')

    axs.append(axhist)

cplt.enumerate_subplots(np.array(axs).reshape(
    len(net_measures), 4).T, pos_x=-0.1, pos_y=1.08)
# %%
plt.show()

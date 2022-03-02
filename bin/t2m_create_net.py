"""Create network of surface temperature data."""
# %%
import os
import argparse
from climnet.dataset import BaseDataset, AnomalyDataset
from climnet.network import net, clim_networkx 
import utils 

PATH = os.path.dirname(os.path.abspath(__file__))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--dataset", default=None, type=str,
                        help=("Path to t2m dataset file."))
    parser.add_argument("-s", "--season", default='full', type=str,
                        help=("Season type, i.e. 'nino', 'nina', 'standard', 'full', "
                              + "'Nino_EP', 'Nino_CP', 'Nina_CP', 'Nina_EP'"))
    parser.add_argument('-lb', type=str2bool, nargs='?', const=True, default=False,
                        help="Apply link bundeling to network")
    parser.add_argument('-c', '--curvature', default='forman', type=str,
                        help=("Computes curvature of network. "
                              +"Choose between forman and ollivier"))
    return parser

# True to run with ipython
if False:
    class Args():
        """Workaround of argsparser."""

        def __init__(self) -> None:
            self.dataset = PATH + "/../data/2m_temperature_monthly_1979_2020.nc"  
            self.season = 'Nino_CP'
            self.lb = False
            self.curvature = 'forman'
    args = Args()
else:
    parser = make_argparser()
    args = parser.parse_args()

# Set paths
dataset_nc = args.dataset
output_dir = PATH + f"/../outputs/t2m_net/"
plot_dir = output_dir + "/plots/"

if not os.path.isdir(output_dir):
    os.makedirs(plot_dir)

# Regridding of data to equidistant grid
print('Create Dataset')
if os.path.exists(dataset_nc + "_fekete"):
    ds = AnomalyDataset(load_nc=dataset_nc + "_fekete", detrend=False)
else:
    ds = AnomalyDataset(data_nc=dataset_nc, var_name='t2m', grid_step=2.5,
                        grid_type='fekete', detrend=True, climatology='dayofyear')
    ds.save(dataset_nc + "_fekete")

# %%
# Select time periods based on the nino indices 
fnino = PATH + "/../data/ersst5.nino.mth.91-20.ascii"
nino_indices = utils.get_nino_indices(
    fnino, time_range=[ds.ds.time[0].data, ds.ds.time[-1].data], time_roll=0)
enso_dict = utils.get_enso_years(nino_indices, month_range=[12, 2],
                               mean=True, threshold=0.5, min_diff=0.1)

if args.season in ["Nino_EP", "Nino_CP", "Nina_CP", "Nina_EP",
                   'nino', 'nina', 'standard']:
    time_period = enso_dict[args.season]
    ds.use_time_snippets(time_period)
else:
    ds = ds

# %%
# Create Correlation Climnet
Net = net.CorrClimNet(ds, corr_method='spearman',
                  density=0.02)
Net.create()

prefix = f"/t2m_{args.season}_net"
# link bundling
if args.lb:
    adjacency_lb = Net.link_bundles(
        num_rand_permutations=2000,
        bw_type='nn',
        nn_points_bw=2,
        link_bundle_folder=prefix,
    )
    print(f'Sparsity after link bundling {Net.get_sparsity(Net.adjacency)}')

# Save network 
netfile = output_dir + prefix + ".npz"
if not os.path.exists(netfile):
    Net.save(netfile)

# %%
from importlib import reload
reload(clim_networkx)
# Convert to graphtool network
cnx = clim_networkx.Clim_NetworkX(dataset=ds,
                                  network=Net,
                                  weighted=True)

# %%
# Compute node and edge attributes
cnx.compute_network_attrs('degree', 'betweenness', 'clustering_coeff')
cnx.compute_curvature(c_type=args.curvature)
# %%
# Compute link length distribution
cnx.compute_link_lengths_edges()
# %%
# save network as networkx file again
cnx.save(savepath=output_dir + prefix + ".graphml")
# %%

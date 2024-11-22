import matplotlib.pyplot as plt
import seaborn as sns

def set_plt_params(remove_grid=False):
    """Set parameters and use seaborn theme to plot."""
    sns.set_theme()
    if remove_grid:
        sns.set_style("whitegrid", {"axes.grid": False})
    params = {
        "font.family": "Serif",
        "font.serif": "Roman", 
        "text.usetex": True,
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "medium", 
        "savefig.dpi": 300, 
        "axes.grid" : False
    }
    plt.rcParams.update(params)
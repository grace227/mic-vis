

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def addColorBar(fig, img, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img, cax = cax, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    return cbar



def plot_xrf_maps(ch_data: np.ndarray, ch_names: list, x_val: np.ndarray, y_val: np.ndarray, elms: list, 
                  ncol: int = 4, nrow: int = None, 
                  figsize: tuple = (10, 10), vmax_th: float = 100, cmap: str = 'inferno', 
                  show_colorbar: bool = True, 
                  add_scalebar: bool = True, show_ticks: bool = True):
    
    """
    Plot XRF maps.
    """
    
    if nrow is None:
        nrow = len(elms) // ncol
    
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    
    for i, (elm, ax_) in enumerate(zip(elms, axs.flatten())):
        plot_array = ch_data[ch_names.index(elm)]
        img = ax_.imshow(plot_array, cmap=cmap, vmax=np.percentile(plot_array, vmax_th))
        ax_.set_title(elm)
        
        
        if show_colorbar:
            addColorBar(fig, img, ax_)
            
        #TODO: add scalebar
        # if add_scalebar:

    for i, ax in enumerate(axs.flat):
        if len(ax.get_images()) == 0:
            print(f"subplot {i} is empty and deleted")
            fig.delaxes(ax)
            
    return fig
            
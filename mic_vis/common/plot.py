

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def addColorBar(fig, img, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img, cax = cax, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    return cbar


# Function to add a horizontal error bar scale bar to plots
def add_scale_bar(ax, scale_length_um, pixel_size_um, 
                  position='bottom-right', color='white', 
                  linewidth=2, fontsize=10, label_offset=5):
    """
    Add a horizontal error bar scale bar to an image plot.
    
    Parameters:
    -----------
    ax : matplotlib axes
        The axes to add the scale bar to
    scale_length_um : float
        Length of scale bar in physical units (e.g., micrometers)
    pixel_size_um : float
        Size of one pixel in physical units (e.g., micrometers per pixel)
    position : str
        Position of scale bar: 'bottom-right', 'bottom-left', 'top-right', 'top-left'
    color : str
        Color of the scale bar and text
    linewidth : float
        Width of the scale bar line
    fontsize : float
        Font size for the label
    label_offset : float
        Offset in pixels for the label text above the scale bar
    """
    # Calculate scale bar length in pixels
    scale_length_pixels = scale_length_um / pixel_size_um
    
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # Determine position based on input
    if position == 'bottom-right':
        x_start = xlim[1] - x_range * 0.15 - scale_length_pixels
        x_end = xlim[1] - x_range * 0.15
        y_pos = ylim[0] + y_range * 0.1
    elif position == 'bottom-left':
        x_start = xlim[0] + x_range * 0.06
        x_end = xlim[0] + x_range * 0.06 + scale_length_pixels
        y_pos = ylim[0] + y_range * 0.1
    elif position == 'top-right':
        x_start = xlim[1] - x_range * 0.15 - scale_length_pixels
        x_end = xlim[1] - x_range * 0.15
        y_pos = ylim[1] - y_range * 0.1
    elif position == 'top-left':
        x_start = xlim[0] + x_range * 0.15
        x_end = xlim[0] + x_range * 0.15 + scale_length_pixels
        y_pos = ylim[1] - y_range * 0.1
    else:
        raise ValueError("position must be 'bottom-right', 'bottom-left', 'top-right', or 'top-left'")
    
    # Calculate center position for error bar
    x_center = (x_start + x_end) / 2
    x_error = scale_length_pixels / 2
    
    # Draw horizontal error bar (scale bar)
    ax.errorbar(x_center, y_pos, xerr=x_error, 
                fmt='-', color=color, linewidth=linewidth, 
                capsize=0, capthick=linewidth)
    
    # # Add vertical lines at ends (optional, makes it look more like a traditional scale bar)
    # ax.plot([x_start, x_start], [y_pos - y_range*0.01, y_pos + y_range*0.01], 
    #         color=color, linewidth=linewidth)
    # ax.plot([x_end, x_end], [y_pos - y_range*0.01, y_pos + y_range*0.01], 
    #         color=color, linewidth=linewidth)
    
    # Add label
    label_text = f'{scale_length_um:.0f} μm'
    ax.text(x_center, y_pos + y_range * (label_offset / 100), label_text,
            ha='center', va='bottom', color=color, fontsize=fontsize,
            weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='black', alpha=0.5, edgecolor='none'))
    
    return ax


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
    
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
    
    for i, (elm, ax_) in enumerate(zip(elms, axs.flatten())):
        plot_array = ch_data[ch_names.index(elm)]
        img = ax_.imshow(plot_array, cmap=cmap, vmax=np.percentile(plot_array, vmax_th))
        ax_.set_title(elm)
        
        if i == 0:
            xticks = np.linspace(0, len(x_val) - 1, 5, dtype=int)
            yticks = np.linspace(0, len(y_val) - 1, 5, dtype=int)
            
            ax_.set_xticks(xticks)
            ax_.set_yticks(yticks)
            ax_.set_xticklabels([np.round(x_val[i], 2) for i in xticks])
            ax_.set_yticklabels([np.round(y_val[i], 2) for i in yticks])
            ax_.tick_params(axis="both", which="major")
        
        
        if show_colorbar:
            addColorBar(fig, img, ax_)
            
        #TODO: add scalebar
        # if add_scalebar:

    for i, ax in enumerate(axs.flat):
        
        if i % ncol == 0:
            # print(f"first column; {i}")
            ax.set_ylabel(r"Y $\mu$m")
        if i >= (nrow*ncol-nrow-1): 
            # print(f"last row; {i}")
            ax.set_xlabel(r"X $\mu$m")
            
        if len(ax.get_images()) == 0:
            print(f"subplot {i} is empty and deleted")
            fig.delaxes(ax)
            
    return fig
            

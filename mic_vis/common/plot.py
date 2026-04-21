

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
                  linewidth=2, fontsize=10, label_offset=5,
                  x_offset_frac=None, y_offset_frac=None):
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
    x_offset_frac : float | None
        Fractional horizontal offset from the axis edge. If None, use the
        built-in default for the selected position.
    y_offset_frac : float | None
        Fractional vertical offset from the axis edge. If None, use the
        built-in default for the selected position.
    """
    # Calculate scale bar length in pixels
    scale_length_pixels = scale_length_um / pixel_size_um
    
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    if x_offset_frac is None:
        x_offset_frac = 0.15
    if y_offset_frac is None:
        y_offset_frac = 0.1
    
    # Determine position based on input
    if position == 'bottom-right':
        x_start = xlim[1] - x_range * x_offset_frac - scale_length_pixels
        x_end = xlim[1] - x_range * x_offset_frac
        y_pos = ylim[0] + y_range * y_offset_frac
    elif position == 'bottom-left':
        left_offset = x_offset_frac if x_offset_frac is not None else 0.06
        x_start = xlim[0] + x_range * left_offset
        x_end = xlim[0] + x_range * left_offset + scale_length_pixels
        y_pos = ylim[0] + y_range * y_offset_frac
    elif position == 'top-right':
        x_start = xlim[1] - x_range * x_offset_frac - scale_length_pixels
        x_end = xlim[1] - x_range * x_offset_frac
        y_pos = ylim[1] - y_range * y_offset_frac
    elif position == 'top-left':
        x_start = xlim[0] + x_range * x_offset_frac
        x_end = xlim[0] + x_range * x_offset_frac + scale_length_pixels
        y_pos = ylim[1] - y_range * y_offset_frac
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


def _nice_scale_length(target):
    if target <= 0:
        return 1.0
    exp = np.floor(np.log10(target))
    base = 10 ** exp
    for m in (1, 2, 5, 10):
        val = m * base
        if target <= val:
            return float(val)
    return float(10 * base)


def plot_xrf_maps(ch_data: np.ndarray, ch_names: list, x_val: np.ndarray, y_val: np.ndarray, elms: list, 
                  ncol: int = 4, nrow: int = None, 
                  figsize: tuple = (10, 10), vmax_th: float = 100, cmap: str = 'inferno', 
                  show_colorbar: bool = True, 
                  add_scalebar: bool = True, show_ticks: bool = True, scale_length_um: int = None):
    
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

        if add_scalebar:
            x_val_arr = np.asarray(x_val)
            dx = np.median(np.diff(x_val_arr))
            pixel_size_um = float(abs(dx)) if dx != 0 else 1.0
            total_um = float(np.max(x_val_arr) - np.min(x_val_arr))
            target = total_um / 5.0 if total_um > 0 else pixel_size_um * 10.0
            if scale_length_um is None:
                scale_length_um = _nice_scale_length(target)
            add_scale_bar(ax_, scale_length_um, pixel_size_um, position='bottom-left')

    
    for i, ax in enumerate(axs.flat):
        if show_ticks:
            ax.set_xlabel(r"X $\mu$m")
            if i % ncol == 0:
                # print(f"first column; {i}")
                ax.set_ylabel(r"Y $\mu$m")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        
    if len(ax.get_images()) == 0:
        print(f"subplot {i} is empty and deleted")
        fig.delaxes(ax)

    plt.tight_layout()
            
    return fig

def plot_xrf_spectrum(int_spec: np.ndarray, energy: np.ndarray, figsize: tuple = (10, 10), 
                      show_ticks: bool = True, title: str = None, ax=None, fig=None):
    """
    Plot XRF spectrum.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(energy, int_spec)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)
    
    if show_ticks:
        ax.tick_params(axis="both", which="major")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    return fig
            

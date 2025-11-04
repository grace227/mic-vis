import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mic_vis.s2idd.mda import get_roi_from_mda
import logging


logger = logging.getLogger(__name__)

def plot_xrf_2d(plotarr, xaxis, yaxis, scan_name, elm_name, cmap, vmax, vmin):
    """
    Plot the XRF data.

    Parameters
    ----------
    plotarr : numpy.ndarray
        The array to plot.
    xaxis : numpy.ndarray
        The x-axis data.
    yaxis : numpy.ndarray
        The y-axis data.
    scan_name : str
        The name of the scan.
    elm_name : str
        The name of the element.
    cmap : str
        The colormap to use.
    vmax : float
        The maximum value of the colorbar.
    vmin : float
        The minimum value of the colorbar.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(plotarr, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(f"{scan_name} {elm_name}")

    # Show only 5 ticks for both x- and y- axes
    xticks = np.linspace(0, len(xaxis) - 1, 5, dtype=int)
    yticks = np.linspace(0, len(yaxis) - 1, 5, dtype=int)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([np.round(xaxis[i], 2) for i in xticks])
    ax.set_yticklabels([np.round(yaxis[i], 2) for i in yticks])
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()
    return fig



def save_xrfdata(
    img_h5_path: str, 
    output_dir: str, 
    cmap: str = "inferno", 
    elms: list[str] = None, 
    vmax_th: float = 99, 
    vmin: float = 0,
    return_image_array: bool = False
) -> str | None:
    """
    Save the XRF data in png format.

    Parameters
    ----------
    img_h5_path : str
        The path to the h5 file.
    output_dir : str
        The path to the output directory.
    cmap : str
        The colormap to use.
    elms : list
        The elements to plot.
    vmax_th : float
        The threshold for the maximum percentile of the colorbar.
    vmin : float
        The minimum value of the colorbar.
    return_image_array : bool
        If True, an numpy array of the image will be returned
        in addition to the path to the saved image.

    Returns
    -------
    str | None
        The path to the saved image.
    """

    
    data = load_xrf(img_h5_path)
    if data:
        data_arr = data["ROI_arr"]
        data_ch = data["ROI_ch"]
        xaxis = data["x_axis"]
        yaxis = data["y_axis"]
        if elms:
            plot_elms = elms
        else:
            plot_elms = data_ch

        for e in plot_elms:
            plotarr = data_arr[data_ch.index(e)]
            vmax = np.nanpercentile(plotarr, vmax_th)
            fig = plot_xrf_2d(plotarr, xaxis, yaxis, data["scan"], e, cmap, vmax, vmin)
            fname = f"{output_dir}/{data['scan']}_{e}.png"
            fig.savefig(fname)
            plt.close(fig)
            logger.info(f"Image saved to {fname}")
            if return_image_array:
                return fname, plotarr
            else:
                return fname
    else:
        logger.error(f"The XRF h5 file {img_h5_path} not found")
        if return_image_array:
            return None, None
        else:
            return None


def load_xrf(img_h5_path, fit_type=["NNLS", "ROI"], fsizelim=1e3) -> dict:
    """
    Load the XRF data from the h5 file.

    Parameters
    ----------
    img_h5_path : str
        The path to the h5 file.
    fit_type : list
        The type of fitting.
    fsizelim : float
        The size limit of the h5 file, only load
        the h5 file larger than this size.

    Returns
    -------
    dict
        The data from the h5 file.
    """
    data = {}
    fsize = os.path.getsize(img_h5_path)
    if fsize > fsizelim:
        with h5py.File(img_h5_path, "r") as f:
            data.update({"scan": os.path.basename(img_h5_path)})
            data.update({"x_axis": f["MAPS/Scan/x_axis"][:]})
            data.update({"y_axis": f["MAPS/Scan/y_axis"][:]})
            for t in fit_type:
                d = f[f"MAPS/XRF_Analyzed/{t}/Counts_Per_Sec"][:]
                d_ch = f[f"MAPS/XRF_Analyzed/{t}/Channel_Names"][:].astype(str).tolist()
                scaler_names = f["MAPS/Scalers/Names"][:].astype(str).tolist()
                scaler_values = f["MAPS/Scalers/Values"][:]

                data.update({f"{t}_arr": d})
                data.update({f"{t}_ch": d_ch})
                data.update({f"{t}_scaler_names": scaler_names})
                data.update({f"{t}_scaler_values": scaler_values})

        return data
    else:
        print(f"The XRF h5 file {img_h5_path} not found")
        return None



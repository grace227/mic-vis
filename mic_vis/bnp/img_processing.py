"""BNP image-processing helpers for ROI extraction from coarse scans."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
# matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from skimage.measure import regionprops


def get_elm_map(file_path: str | Path, elm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an element map and scan axes from a BNP HDF5 file."""

    with h5py.File(file_path, "r") as dat:
        xrf_path = "/MAPS/XRF_roi_plus" if "/MAPS/XRF_roi_plus" in dat else "/MAPS/XRF_roi"
        channel_names = _decode_channel_names(dat["/MAPS/channel_names"][:])
        try:
            ch_idx = channel_names.index(elm)
        except ValueError as exc:
            raise ValueError(f"Invalid element '{elm}'; not found in channel list") from exc
        elmmap = np.asarray(dat[xrf_path][ch_idx, :, :], dtype=float)
        x_pos = np.asarray(dat["/MAPS/x_axis"][:], dtype=float)
        y_pos = np.asarray(dat["/MAPS/y_axis"][:], dtype=float)
    return elmmap, x_pos, y_pos


def get_roi_coordinate_data(
    elmmap: np.ndarray,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    *,
    savefig: bool = False,
    figpath: str | Path | None = None,
    n_cluster: int = 2,
    sel_cluster: int = 1,
) -> tuple[float, float, float, float]:
    """Cluster an element map and return the selected ROI center and size."""

    kmean_map = kmean_analysis(n_cluster, elmmap, random_state=42, plotoption=False)
    region_prop = regionprops(np.array(kmean_map[0] == sel_cluster, dtype="int"))
    if not region_prop:
        raise ValueError("No ROI region found from clustered element map")
    region_bbox = region_prop[0].bbox
    fig = plot_bbox(elmmap, region_bbox, x_pos, y_pos)

    width = x_pos[region_bbox[3] - 1] - x_pos[region_bbox[1]]
    height = y_pos[region_bbox[2] - 1] - y_pos[region_bbox[0]]
    new_x = width / 2 + x_pos[region_bbox[1]]
    new_y = height / 2 + y_pos[region_bbox[0]]

    if savefig and figpath is not None:
        fig.savefig(figpath, dpi=100, transparent=True)
        plt.close(fig)

    return float(new_x), float(new_y), float(width), float(height)


def get_coordinate(
    file_path: str | Path,
    *,
    elm: str,
    mask_elm: str | None = None,
    use_mask: bool = False,
    n_std: float = 2.0,
    n_cluster: int = 2,
    sel_cluster: int = 1,
    figpath: str | Path | None = None,
) -> tuple[float, float]:
    """Return coarse-scan-derived fine-scan coordinates."""

    elmmap, x_pos, y_pos = get_elm_map(file_path, elm)
    mask = np.ones(elmmap.shape, dtype=bool)
    if use_mask and mask_elm:
        maskmap, _, _ = get_elm_map(file_path, mask_elm)
        mask = maskmap < (np.mean(maskmap) + float(n_std) * np.std(maskmap.ravel()))

    masked = elmmap * mask
    x, y, _w, _h = get_roi_coordinate_data(
        masked,
        x_pos,
        y_pos,
        savefig=figpath is not None,
        figpath=figpath,
        n_cluster=n_cluster,
        sel_cluster=sel_cluster,
    )
    return round(x, 2), round(y, 2)


def plot_bbox(elmmap: np.ndarray, box: tuple[int, int, int, int], x_pos: np.ndarray, y_pos: np.ndarray):
    minr, minc, maxr, maxc = box
    x_st = x_pos[minc]
    y_st = y_pos[minr]
    width = x_pos[maxc - 1] - x_pos[minc]
    height = y_pos[maxr - 1] - y_pos[minr]
    rect = mpatches.Rectangle((x_st, y_st), width, height, fill=False, edgecolor="red", linewidth=2)
    fig = plt.figure()
    ax = fig.gca()
    ax.pcolor(x_pos, y_pos, elmmap, cmap="gray", shading="auto")
    ax.add_patch(rect)
    return fig


def _build_overlay_rgba(
    data: np.ndarray,
    *,
    channel_index: int,
    vmax: float | None,
    alpha: float,
) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    rgba = np.zeros(arr.shape + (4,), dtype=float)
    if vmax is None or vmax <= 0:
        return rgba

    scaled = np.clip(arr / vmax, 0, 1)
    rgba[..., channel_index] = scaled
    rgba[..., 3] = alpha * scaled
    return rgba


def plot_registration_overlay(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    vmax: float | None = None,
    percentile: float = 99.0,
    moving_color: str = "blue",
    reference_color: str = "red",
    alpha: float = 0.45,
    show_ticks: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a false-color overlay of moving and reference maps.

    Parameters
    ----------
    moving, reference:
        2D arrays with the same shape.
    fig, ax:
        Optional existing matplotlib figure/axes. If ``ax`` is omitted, a new
        figure and axes are created unless ``fig`` is provided, in which case
        ``fig.gca()`` is used.
    vmax:
        Optional intensity cap used to scale both overlays. If omitted, a joint
        percentile across both arrays is used.
    """

    moving_arr = np.asarray(moving, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    if moving_arr.shape != reference_arr.shape:
        raise ValueError(
            f"moving and reference must have the same shape, got {moving_arr.shape} and {reference_arr.shape}"
        )

    finite_vals = np.concatenate(
        [
            moving_arr[np.isfinite(moving_arr)],
            reference_arr[np.isfinite(reference_arr)],
        ]
    )
    if vmax is None:
        vmax = float(np.nanpercentile(finite_vals, percentile)) if finite_vals.size else None

    color_to_channel = {"red": 0, "green": 1, "blue": 2}
    try:
        moving_channel = color_to_channel[moving_color.lower()]
        reference_channel = color_to_channel[reference_color.lower()]
    except KeyError as exc:
        raise ValueError("moving_color/reference_color must be one of: red, green, blue") from exc

    moving_rgba = _build_overlay_rgba(moving_arr, channel_index=moving_channel, vmax=vmax, alpha=alpha)
    reference_rgba = _build_overlay_rgba(reference_arr, channel_index=reference_channel, vmax=vmax, alpha=alpha)

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.gca()
    elif fig is None:
        fig = ax.figure

    ax.imshow(reference_rgba)
    ax.imshow(moving_rgba)
    if title is not None:
        ax.set_title(title)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, ax


def kmean_analysis(
    n_clusters: int,
    data: np.ndarray,
    *,
    random_state: int = 52,
    plotoption: bool = False,
) -> tuple[np.ndarray, plt.Figure | None]:
    """Run KMeans segmentation on an element map."""

    data = np.array(data, dtype=float, copy=True)
    data[np.isnan(data)] = 1e-5
    data[np.isinf(data)] = 1e-5

    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    km.fit(data.reshape(-1, 1))

    km_label = np.reshape(km.labels_, data.shape)
    srt_index = np.argsort(km.cluster_centers_[:, 0])
    for i, s in enumerate(srt_index):
        km_label[km_label == s] = -(i + 1)
    km_label = np.multiply(-1, km_label) - 1

    fig = None
    if plotoption:
        fig, ax = plt.subplots()
        ax.imshow(km_label, vmin=0, vmax=n_clusters - 1)
    return km_label, fig


def _decode_channel_names(raw_names: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for entry in raw_names:
        if isinstance(entry, bytes):
            decoded.append(entry.decode("utf-8", errors="ignore").strip())
        else:
            decoded.append(str(entry).strip())
    return decoded

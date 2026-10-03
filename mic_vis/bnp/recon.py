"""BNP reconstruction helpers built on ASTRA."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import astra
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def sino_recon(
    sinogram: np.ndarray,
    angles: np.ndarray,
    w_pixel: float,
    algorithm: str,
    n_iter: int | None = None,
    ycenter: float = 0,
    xcenter: float = 0,
    volscale: float = 1,
) -> np.ndarray:
    """Reconstruct a single 2D slice from a sinogram using ASTRA."""

    sinogram = np.asarray(sinogram, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)
    n_angles, n_y = sinogram.shape
    if n_angles != len(angles):
        raise ValueError(
            "Dim of sinogram and angle does not match. "
            "Sinogram should have shape of (angles, voxels)"
        )

    proj_geom = astra.create_proj_geom('parallel', w_pixel, n_y, angles)
    proj_geom = astra.geom_postalignment(proj_geom, [ycenter, xcenter])
    vol_geom = astra.create_vol_geom(int(n_y * volscale), int(n_y * volscale))
    proj_id = astra.create_projector('linear', proj_geom, vol_geom)

    sinogram_id = astra.data2d.link('-sino', proj_geom, sinogram)
    recon_id = astra.data2d.create('-vol', vol_geom)
    algorithm_id = None
    try:
        cfg = astra.astra_dict(algorithm)
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ReconstructionDataId'] = recon_id
        algorithm_id = astra.algorithm.create(cfg)
        astra.algorithm.run(algorithm_id, 1 if n_iter is None else n_iter)
        return astra.data2d.get(recon_id)
    finally:
        if algorithm_id is not None:
            astra.algorithm.delete(algorithm_id)
        astra.data2d.delete([sinogram_id, recon_id])
        astra.projector.delete(proj_id)


# Backward-compatible alias.
sinoRecon = sino_recon


def order_proj_axis(
    proj: np.ndarray,
    angle_axis: int,
    col_axis: int,
    row_axis: int,
    elm_axis: int | None,
) -> np.ndarray:
    """Reorder projections to [elm, col, angle, row]."""

    new_proj = np.asarray(proj)
    ndim = new_proj.ndim

    if elm_axis is None:
        if ndim > 3:
            raise ValueError('Need to specify axis dimension of elemental channel')
        if ndim == 3:
            new_proj = np.expand_dims(new_proj, axis=0)
            ndim = new_proj.ndim
            elm_axis = 0
            angle_axis += 1
            col_axis += 1
            row_axis += 1
    elif elm_axis >= ndim:
        raise ValueError('Elm axis is larger than array dimension')

    source_axes = (elm_axis, col_axis, angle_axis, row_axis)
    if len(set(source_axes)) != 4:
        raise ValueError('elm_axis, col_axis, angle_axis, and row_axis must be distinct')
    if any(ax < 0 or ax >= ndim for ax in source_axes):
        raise ValueError('Axis index is out of bounds for projection array')

    return np.moveaxis(new_proj, source_axes, (0, 1, 2, 3))


# Backward-compatible alias.
orderProjAxis = order_proj_axis


def recon(
    proj: np.ndarray,
    angles: np.ndarray,
    angle_axis: int,
    col_axis: int,
    row_axis: int,
    elm_axis: int | None,
    w_pixel: float,
    algorithm: str,
    n_iter: int | None = None,
    ycenter: float = 0,
    xcenter: float = 0,
    volscale: float = 1,
) -> dict[str, np.ndarray]:
    """Iterate through all sinograms and return the reconstructed volume."""

    ordered = order_proj_axis(
        proj,
        angle_axis=angle_axis,
        col_axis=col_axis,
        row_axis=row_axis,
        elm_axis=elm_axis,
    )
    n_elm, n_sino, _n_angle, n_row = ordered.shape
    recon_vol = np.zeros((n_elm, n_sino, int(n_row * volscale), int(n_row * volscale)), dtype=np.float32)
    for i in range(n_elm):
        for j in range(n_sino):
            sino = np.array(ordered[i, j, :, :], order='C', dtype=np.float32)
            recon_vol[i, j, :, :] = sino_recon(
                sino,
                angles,
                w_pixel,
                algorithm,
                n_iter=n_iter,
                ycenter=ycenter,
                xcenter=xcenter,
                volscale=volscale,
            )
    return {'proj_input': ordered, 'recon': recon_vol}


def sino_recon_2d_cuda(
    sinogram: np.ndarray,
    angles: np.ndarray,
    w_pixel: float,
    algorithm: str = 'SIRT_CUDA',
    n_iter: int | None = 1,
    ycenter: float = 0,
    xcenter: float = 0,
    volscale: float = 1,
    mini_constraint: float | None = None,
) -> np.ndarray:
    """Reconstruct a single 2D slice from a sinogram using ASTRA CUDA."""

    sinogram = np.asarray(sinogram, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)
    n_angles, n_y = sinogram.shape
    if n_angles != len(angles):
        raise ValueError(
            "Dim of sinogram and angle does not match. "
            "Sinogram should have shape of (angles, voxels)"
        )

    proj_geom = astra.create_proj_geom('parallel', w_pixel, n_y, angles)
    proj_geom = astra.geom_postalignment(proj_geom, [ycenter, xcenter])
    vol_geom = astra.create_vol_geom(int(n_y * volscale), int(n_y * volscale))
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

    sinogram_id = astra.data2d.link('-sino', proj_geom, sinogram)
    recon_id = astra.data2d.create('-vol', vol_geom)
    algorithm_id = None
    try:
        cfg = astra.astra_dict(algorithm)
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ReconstructionDataId'] = recon_id
        if mini_constraint is not None:
            cfg['MinConstraint'] = mini_constraint
        algorithm_id = astra.algorithm.create(cfg)
        if n_iter is None:
            astra.algorithm.run(algorithm_id)
        else:
            astra.algorithm.run(algorithm_id, n_iter)
        return astra.data2d.get(recon_id)
    finally:
        if algorithm_id is not None:
            astra.algorithm.delete(algorithm_id)
        astra.data2d.delete([sinogram_id, recon_id])
        astra.projector.delete(proj_id)


# Backward-compatible alias.
sinoRecon_2DCUDA = sino_recon_2d_cuda


def recon_cuda(
    proj: np.ndarray,
    angles: np.ndarray,
    algorithm: str,
    angle_axis: int = 2,
    col_axis: int = 1,
    row_axis: int = 3,
    elm_axis: int | None = 0,
    w_pixel: float = 1,
    n_iter: int | None = 300,
    ycenter: float = 0,
    xcenter: float = 0,
    volscale: float = 1,
    clip_negative: bool = True,
) -> dict[str, np.ndarray]:
    """Iterate through all sinograms and return the reconstructed volume using CUDA."""

    ordered = order_proj_axis(
        proj,
        angle_axis=angle_axis,
        col_axis=col_axis,
        row_axis=row_axis,
        elm_axis=elm_axis,
    )
    n_elm, n_sino, _n_angle, n_row = ordered.shape
    print(f"ordered proj shape:{n_elm=},{n_sino=},{_n_angle=},{n_row=}")
    recon_vol = np.zeros((n_elm, n_sino, int(n_row * volscale), int(n_row * volscale)), dtype=np.float32)
    for i in range(n_elm):
        sino_iterator = tqdm(
            range(n_sino),
            desc=f"recon_cuda elm {i + 1}/{n_elm}",
            leave=False,
        )
        for j in sino_iterator:
            sino = np.array(ordered[i, j, :, :], order='C', dtype=np.float32)
            vslice = sino_recon_2d_cuda(
                sino,
                angles,
                w_pixel,
                algorithm=algorithm,
                n_iter=n_iter,
                ycenter=ycenter,
                xcenter=xcenter,
                volscale=volscale,
            )
            if clip_negative:
                vslice[vslice < 0] = 0
            recon_vol[i, j, :, :] = vslice
    return {'proj_input': ordered, 'recon': recon_vol}


# Backward-compatible alias.
recon_CUDA = recon_cuda


def num_proj_index(thetas: np.ndarray, numproj: int) -> np.ndarray:
    """Return evenly spaced projection indices across 180 degrees."""

    if numproj > len(thetas):
        raise ValueError('numproj is larger than theta size')
    return np.arange(0, 180, 180 / numproj, dtype=int)


# Backward-compatible alias.
numProjIndex = num_proj_index


def plot_repro(
    numprojs: list[Any],
    sel_recon_data: dict[str, dict[str, np.ndarray]],
    proj: np.ndarray,
    proj_axis: int | None = None,
    sino: bool | None = None,
    sino_slice: list[int] | None = None,
    cmax: list[float] | None = None,
    coltitle: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot reprojection images or sinograms for selected reconstructions."""

    labels = ['P', 'Zn', 'Fe']
    cmap = ['inferno', 'viridis', 'cividis']
    tpos = [-30, -40, -40]

    if proj_axis is None:
        proj_axis = 2

    if proj_axis != 0:
        fig, axes = plt.subplots(3, len(numprojs) + 1, figsize=figsize)
        cbar_s = 0.98
        for i in range(3):
            data = proj[0, :, :, i]
            vmax = None if cmax is None else cmax[i]
            m = axes[i, -1].imshow(data, cmap[i], vmin=0, vmax=vmax)
            axes[i, -1].set_xticks([])
            axes[i, -1].set_yticks([])
            fig.colorbar(m, ax=axes[i, -1], shrink=cbar_s)
        axes[0, -1].set_title('Proj')
    else:
        fig, axes = plt.subplots(3, len(numprojs), figsize=figsize)
        cbar_s = 0.98

    for i, key in enumerate(sel_recon_data.keys()):
        for j, label in enumerate(labels):
            if sino:
                if sino_slice is None:
                    raise ValueError('sino_slice must be provided when sino=True')
                data = sel_recon_data[key]['proj_input'][j, sino_slice[j], :, :]
            else:
                data = np.sum(sel_recon_data[key]['recon'][j, :, :, :], axis=proj_axis).T
            vmax = None if cmax is None else cmax[j]
            m = axes[j, i].imshow(data, cmap[j], vmin=0, vmax=vmax)
            axes[j, i].axis('off')
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
            if i == 0:
                axes[j, i].text(tpos[j], 70, label)
            if j == 0 and coltitle is None:
                axes[j, i].set_title(str(sel_recon_data[key]['proj_input'].shape[2]))
            elif j == 0:
                axes[j, i].set_title(coltitle[i])
            if proj_axis == 0 and i == len(numprojs) - 1:
                fig.colorbar(m, ax=axes[j, i], shrink=cbar_s)
    plt.tight_layout()
    return fig, axes


# Backward-compatible alias.
plotRepro = plot_repro


def plot_repro_line(
    numprojs: list[Any],
    sel_recon_data: dict[str, dict[str, np.ndarray]],
    proj: np.ndarray,
    proj_axis: int | None = None,
    ymax: list[float] | None = None,
):
    """Plot reprojection line profiles for selected reconstructions."""

    labels = ['P', 'Zn', 'Fe']
    cmap = ['Reds', 'Greens', 'Blues']
    label_c = ['r', 'g', 'b']
    xlim = [85, 85, 75]
    ymin = [0.6, 0.02, -2e-6]

    fig, axes = plt.subplots(len(labels), 1, figsize=(5, 7), sharex=True)
    if proj_axis is None:
        proj_axis = 2

    for j, label in enumerate(labels):
        cidx = np.linspace(0.9, 0.2, len(sel_recon_data))
        axins2 = inset_axes(axes[j], width='30%', height='40%')
        for i, key in enumerate(sel_recon_data.keys()):
            p = np.sum(sel_recon_data[key]['recon'][j, :, :, :], axis=proj_axis).T
            lineprofile = np.sum(p, axis=1) / p.shape[1]
            x = np.arange(0, len(lineprofile))
            axes[j].plot(x, lineprofile, color=plt.get_cmap(cmap[j])(cidx[i]), label=key)
            axins2.plot(x, lineprofile, color=plt.get_cmap(cmap[j])(cidx[i]))

        p_proj = proj[0, :, :, j]
        pline = np.sum(p_proj, axis=1) / p_proj.shape[1]
        axes[j].plot(np.arange(0, len(pline)), pline, color='k', label='Proj', alpha=0.8)
        axins2.plot(np.arange(0, len(pline)), pline, color='k', label='Proj', alpha=0.8)
        y_min, y_max = axes[j].get_ylim()
        axins2.set_ylim((ymin[j], y_max))
        axins2.set_yticks([])

        if ymax is not None:
            y_max = ymax[j]
            axes[j].set_ylim((y_min, y_max))

        axes[j].text(0, y_max * 0.85, label, color=label_c[j])
        axes[j].set_ylabel('Average Pixel Intensity (a.u.)')
        axins2.set_xlim((50, xlim[j]))
        axes[j].legend(frameon=False, bbox_to_anchor=(1, 0.3, 0.5, 0.5))

    axes[j].set_xlabel('Position (pixel)')
    return fig


# Backward-compatible alias.
plotReproLine = plot_repro_line


def misalign_proj(proj: np.ndarray, misalign_ratio: float, axis: int | None = None) -> np.ndarray:
    """Apply random circular shifts to a subset of projections."""

    n_angle = proj.shape[0]
    n_proj = int(n_angle * misalign_ratio)
    np.random.seed(40)
    nproj = np.random.randint(n_angle, size=n_proj)

    if axis is None:
        axis = 1

    if axis <= 2:
        n_pix = proj.shape[axis]
        shifts = np.random.randint(-n_pix * 0.2, high=n_pix * 0.2, size=(n_proj, 1))
        s_axis = axis - 1
    elif axis == 3:
        pix1, pix2 = proj[0, :, :, 0].shape
        s1 = np.random.randint(-pix1 * 0.2, high=pix1 * 0.2, size=n_proj)
        s2 = np.random.randint(-pix2 * 0.2, high=pix2 * 0.2, size=n_proj)
        shifts = np.vstack((s1, s2)).T
        s_axis = (0, 1)
    else:
        raise ValueError(
            'Value of n_axis is too large. Expecting n_axis '
            '(number of misaligned axes to be less than or equal to 3)'
        )

    nonalign_proj = np.array(proj, copy=True)
    for n_, s_ in zip(nproj, shifts):
        temp = nonalign_proj[n_, :, :, :].copy()
        nonalign_proj[n_, :, :, :] = np.roll(temp, tuple(s_), axis=s_axis)
    return nonalign_proj


def write_tiff(data: np.ndarray, fname: str = 'tmp/data', digit: int | None = None, ext: str = 'tiff') -> None:
    """Write image data to a TIFF file."""

    out = fname
    if digit is not None:
        out = os.path.join(out, str(digit))
    if not str(out).endswith(ext):
        out = '.'.join([out, ext])
    out = os.path.abspath(out)
    dname = os.path.dirname(out)
    if not os.path.exists(dname):
        os.makedirs(dname)
    tifffile.imwrite(out, data)


def make_tomviz_h5(
    fname: str | os.PathLike[str],
    data: np.ndarray,
    scans: np.ndarray,
    theta: np.ndarray,
    xval: np.ndarray,
    yval: np.ndarray,
) -> None:
    """Export data in HDF5 format for visualization in Tomviz."""

    with h5py.File(fname, 'w') as f:
        g1 = f.create_group('exchange')
        g1.create_dataset('data', data=data)
        g1.create_dataset('scanName', data=scans)
        g1.create_dataset('theta', data=theta)
        g1.create_dataset('x_axis', data=xval)
        g1.create_dataset('y_axis', data=yval)


# Backward-compatible alias.
makeTomvizH5 = make_tomviz_h5


def load_aligned_data_from_json(fpath: str | os.PathLike[str]) -> dict[str, Any]:
    """Load aligned elemental data from a JSON export."""

    with open(fpath) as json_file:
        data = json.load(json_file)

    elms = list(data.keys())[3:]
    angles = np.array(data['angles'], dtype=float)
    a, h, w = np.array(data[elms[0]]).shape
    elmdata = np.zeros((len(elms), a, h, w), dtype=float)
    for i, elm in enumerate(elms):
        elmdata[i, ...] = np.array(data[elm])
    elmdata = np.moveaxis(elmdata, 1, 2)
    p_angles = angles * np.pi / 180

    return {
        'elms': elms,
        'elmdata': elmdata,
        'angles_deg': angles,
        'angle_rad': p_angles,
        'stepsize': data['stepsize'],
    }


# Backward-compatible alias.
loadAlignedDataFromJson = load_aligned_data_from_json


__all__ = [
    'loadAlignedDataFromJson',
    'load_aligned_data_from_json',
    'makeTomvizH5',
    'make_tomviz_h5',
    'misalign_proj',
    'numProjIndex',
    'num_proj_index',
    'orderProjAxis',
    'order_proj_axis',
    'plotRepro',
    'plotReproLine',
    'plot_repro',
    'plot_repro_line',
    'recon',
    'recon_CUDA',
    'recon_cuda',
    'sinoRecon',
    'sinoRecon_2DCUDA',
    'sino_recon',
    'sino_recon_2d_cuda',
    'write_tiff',
]

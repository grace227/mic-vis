import matplotlib.pyplot as plt
import numpy as np

from mic_vis.common.plot import add_scale_bar
from mic_vis.s2idd.xrf import _get_element_map, _get_element_quant


def _match_map_shape(map_arr, x_axis, y_axis):
    map_arr = np.asarray(map_arr)
    if map_arr.ndim != 2:
        raise ValueError(f'expected 2D map, got shape {map_arr.shape}')
    if map_arr.shape == (len(y_axis), len(x_axis)):
        return map_arr
    if map_arr.shape == (len(x_axis), len(y_axis)):
        return map_arr.T
    return map_arr


def _safe_normalize(arr, denom, threshold=1.0):
    out = np.zeros_like(arr, dtype=float)
    return np.divide(arr, denom, out=out, where=np.asarray(denom) > threshold)


def _prepare_plot_axes(row, scaler_name):
    x_axis = np.asarray(row['x_axis'])
    y_axis = np.asarray(row['y_axis'])
    scaler = _match_map_shape(np.asarray(row[scaler_name], dtype=float), x_axis, y_axis)
    return x_axis, y_axis, scaler


def _resolve_map_vmax(plot_arr, vmax_map, key, default_vmax_percentile=99):
    vmax = vmax_map.get(key)
    if vmax is None:
        vmax = np.nanpercentile(plot_arr, default_vmax_percentile)
    return vmax if vmax > 0 else None


def _apply_ticks(
    ax,
    x_axis,
    y_axis,
    show_x_tick_labels=True,
    show_y_tick_labels=True,
    show_x_ticks=True,
    show_y_ticks=True,
):
    xticks = np.linspace(0, max(len(x_axis) - 1, 0), min(5, len(x_axis)), dtype=int)
    yticks = np.linspace(0, max(len(y_axis) - 1, 0), min(5, len(y_axis)), dtype=int)
    ax.set_xticks(xticks if show_x_ticks else [])
    ax.set_yticks(yticks if show_y_ticks else [])
    if show_x_tick_labels:
        ax.set_xticklabels([np.round(x_axis[i], 2) for i in xticks])
    else:
        ax.set_xticklabels([])
    if show_y_tick_labels:
        ax.set_yticklabels([np.round(y_axis[i], 2) for i in yticks])
    else:
        ax.set_yticklabels([])


def _plot_map_panel_grid(
    row,
    plot_fit_type,
    elms,
    map_getter,
    scaler_name='US_IC',
    ncol=3,
    figsize=None,
    show_x_tick_labels=None,
    show_y_tick_labels=None,
    vmax_map=None,
    default_vmax_percentile=99,
    cbar_shrink=0.75,
    add_scalebar=False,
    scalebar_position='bottom-left',
    scale_length_um=None,
    scalebar_x_offset_frac=None,
    scalebar_y_offset_frac=None,
    fig=None,
    axs=None,
    panel_offset=0,
    total_panels=None,
    finalize_layout=True,
):
    required_panels = panel_offset + len(elms)
    if total_panels is None:
        total_panels = required_panels
    total_panels = max(total_panels, required_panels)
    ncol = min(ncol, max(1, total_panels))
    nrow = int(np.ceil(total_panels / ncol))
    cmap = plt.cm.inferno.copy()
    cmap.set_bad('black')

    x_axis, y_axis, scaler = _prepare_plot_axes(row, scaler_name)

    if figsize is None:
        x_span = float(np.ptp(x_axis)) if len(x_axis) > 1 else float(len(x_axis) or 1)
        y_span = float(np.ptp(y_axis)) if len(y_axis) > 1 else float(len(y_axis) or 1)
        aspect = x_span / y_span if y_span not in (0, 0.0) else 1.0
        aspect = min(max(aspect, 0.6), 1.8)
        panel_height = 3.0
        panel_width = panel_height * aspect + 0.9
        figsize = (panel_width * ncol, panel_height * nrow + 0.4)

    if vmax_map is None:
        vmax_map = {}

    x_labels_explicit = show_x_tick_labels is not None
    y_labels_explicit = show_y_tick_labels is not None

    if show_x_tick_labels is None:
        show_x_tick_labels = not add_scalebar
    if show_y_tick_labels is None:
        show_y_tick_labels = not add_scalebar

    show_x_ticks = show_x_tick_labels or x_labels_explicit
    show_y_ticks = show_y_tick_labels or y_labels_explicit

    created_axes = False
    if axs is None and fig is not None and hasattr(fig, '_micvis_main_axes'):
        axs = fig._micvis_main_axes

    if axs is None:
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False, sharex=True, sharey=True)
        created_axes = True
    elif fig is None:
        fig = axs.flat[0].figure if hasattr(axs, 'flat') else np.ravel(axs)[0].figure

    axs_array = np.asarray(axs, dtype=object).reshape(-1)
    if len(axs_array) < required_panels:
        raise ValueError(
            f'not enough axes provided: need {required_panels}, got {len(axs_array)}'
        )

    fig._micvis_main_axes = axs_array.reshape(nrow, ncol) if len(axs_array) == nrow * ncol else axs

    target_axes = axs_array[panel_offset:required_panels]
    for ax, el in zip(target_axes, elms):
        ax.set_axis_on()
        plot_arr, cbar_label = map_getter(row, plot_fit_type, el, x_axis, y_axis, scaler)
        vmax = _resolve_map_vmax(
            plot_arr,
            vmax_map,
            el,
            default_vmax_percentile=default_vmax_percentile,
        )
        img = ax.imshow(plot_arr, cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(el)
        _apply_ticks(
            ax,
            x_axis,
            y_axis,
            show_x_tick_labels=show_x_tick_labels,
            show_y_tick_labels=show_y_tick_labels,
            show_x_ticks=show_x_ticks,
            show_y_ticks=show_y_ticks,
        )
        cbar = fig.colorbar(img, ax=ax, shrink=cbar_shrink)
        cbar.set_label(cbar_label)
        if add_scalebar:
            x_axis_arr = np.asarray(x_axis)
            dx = np.median(np.diff(x_axis_arr)) if len(x_axis_arr) > 1 else 1.0
            pixel_size_um = float(abs(dx)) if dx != 0 else 1.0
            total_um = float(np.max(x_axis_arr) - np.min(x_axis_arr)) if len(x_axis_arr) > 1 else pixel_size_um
            target = total_um / 5.0 if total_um > 0 else pixel_size_um * 10.0
            scale_um = scale_length_um
            if scale_um is None:
                for candidate in (1, 2, 5, 10):
                    val = candidate * 10 ** np.floor(np.log10(target)) if target > 0 else 1.0
                    if target <= val:
                        scale_um = float(val)
                        break
                if scale_um is None:
                    scale_um = float(target)
            add_scale_bar(
                ax,
                scale_um,
                pixel_size_um,
                position=scalebar_position,
                x_offset_frac=scalebar_x_offset_frac,
                y_offset_frac=scalebar_y_offset_frac,
            )

    if created_axes:
        for ax in axs_array[required_panels:]:
            ax.axis('off')

    if finalize_layout:
        fig.tight_layout()
    return fig


def _element_map_getter(normalize_by_scaler=False, scaler_name='US_IC', avoid_divide=1.0):
    def _get_map(row, plot_fit_type, el, x_axis, y_axis, scaler):
        arr = _match_map_shape(_get_element_map(row, el, plot_fit_type), x_axis, y_axis)
        if normalize_by_scaler:
            plot_arr = _safe_normalize(arr, scaler, threshold=avoid_divide)
            quant_value = _get_element_quant(row, el, fit_type=plot_fit_type, quant_norm=scaler_name)
            if quant_value not in (None, 0):
                plot_arr = plot_arr / quant_value
                label = r'$\mu g/cm^2$'
            else:
                label = f'cts/sec/{scaler_name}'
        else:
            plot_arr = arr
            label = 'cts/sec'
        return plot_arr, label

    return _get_map


def _ratio_map_getter(
    normalize_by_scaler=False,
    scaler_name='US_IC',
    avoid_ratio=1.0,
    calibration_factor_map=None,
    clip_percentiles=(1, 99),
    denominator_mask_percentile=None,
    masked_value=np.nan,
):
    if calibration_factor_map is None:
        calibration_factor_map = {}

    def _get_map(row, plot_fit_type, el, x_axis, y_axis, scaler):
        if ':' in el:
            el1, el2 = el.split(':', 1)
            arr1 = _match_map_shape(_get_element_map(row, el1, plot_fit_type), x_axis, y_axis)
            arr2 = _match_map_shape(_get_element_map(row, el2, plot_fit_type), x_axis, y_axis)
            if normalize_by_scaler:
                arr1 = _safe_normalize(arr1, scaler, threshold=avoid_ratio)
                arr2 = _safe_normalize(arr2, scaler, threshold=avoid_ratio)
                quant_value_1 = _get_element_quant(row, el1, fit_type=plot_fit_type, quant_norm=scaler_name)
                quant_value_2 = _get_element_quant(row, el2, fit_type=plot_fit_type, quant_norm=scaler_name)
                if quant_value_1 not in (None, 0):
                    arr1 = arr1 / quant_value_1
                if quant_value_2 not in (None, 0):
                    arr2 = arr2 / quant_value_2

            arr1 = np.asarray(arr1, dtype=float).copy()
            arr2 = np.asarray(arr2, dtype=float).copy()
            arr2_raw = arr2.copy()
            arr1_finite = arr1[np.isfinite(arr1)]
            arr2_finite = arr2[np.isfinite(arr2)]

            if arr1_finite.size == 0 or arr2_finite.size == 0:
                plot_arr = np.full_like(arr1, masked_value, dtype=float)
            else:
                valid = (arr2_raw != 0) & np.isfinite(arr2_raw)
                if denominator_mask_percentile is not None:
                    positive_den = arr2_raw[(arr2_raw > 0) & np.isfinite(arr2_raw)]
                    if positive_den.size:
                        den_cutoff = np.nanpercentile(positive_den, denominator_mask_percentile)
                        valid &= arr2_raw > den_cutoff
                    else:
                        valid &= False

                clip_lo, clip_hi = clip_percentiles
                arr1_lo, arr1_hi = np.nanpercentile(arr1_finite, [clip_lo, clip_hi])
                arr2_lo, arr2_hi = np.nanpercentile(arr2_finite, [clip_lo, clip_hi])
                arr1[(arr1 < arr1_lo) | (arr1 > arr1_hi)] = 0
                arr2[(arr2 < arr2_lo) | (arr2 > arr2_hi)] = 0

                valid &= (arr2 != 0) & np.isfinite(arr2)
                plot_arr = np.full_like(arr1, masked_value, dtype=float)
                np.divide(arr1, arr2, out=plot_arr, where=valid)
                finite_plot = plot_arr[np.isfinite(plot_arr)]
                if finite_plot.size:
                    ratio_hi = np.nanpercentile(finite_plot, clip_hi)
                    plot_arr = np.where(np.isfinite(plot_arr), np.clip(plot_arr, 0, ratio_hi), plot_arr)
                if masked_value == 0:
                    plot_arr[~np.isfinite(plot_arr)] = 0

            calibration_factor = calibration_factor_map.get(el, 1)
            plot_arr = calibration_factor * plot_arr
            label = 'ratio'
        else:
            arr = _match_map_shape(_get_element_map(row, el, plot_fit_type), x_axis, y_axis)
            if normalize_by_scaler:
                plot_arr = _safe_normalize(arr, scaler, threshold=avoid_ratio)
                quant_value = _get_element_quant(row, el, fit_type=plot_fit_type, quant_norm=scaler_name)
                if quant_value not in (None, 0):
                    plot_arr = plot_arr / quant_value
                    label = r'$\mu g/cm^2$'
                else:
                    label = f'cts/sec/{scaler_name}'
            else:
                plot_arr = arr
                label = 'cts/sec'
        return plot_arr, label

    return _get_map


def plot_elm_maps(
    row,
    plot_fit_type,
    elms,
    scaler_name='US_IC',
    normalize_by_scaler=False,
    avoid_divide=1.0,
    ncol=3,
    figsize=None,
    show_x_tick_labels=None,
    show_y_tick_labels=None,
    vmax_map=None,
    default_vmax_percentile=99,
    cbar_shrink=0.75,
    add_scalebar=False,
    scalebar_position='bottom-left',
    scale_length_um=None,
    scalebar_x_offset_frac=None,
    scalebar_y_offset_frac=None,
    fig=None,
    axs=None,
    panel_offset=0,
    total_panels=None,
    finalize_layout=True,
):
    return _plot_map_panel_grid(
        row,
        plot_fit_type,
        elms,
        map_getter=_element_map_getter(
            normalize_by_scaler=normalize_by_scaler,
            scaler_name=scaler_name,
            avoid_divide=avoid_divide,
        ),
        scaler_name=scaler_name,
        ncol=ncol,
        figsize=figsize,
        show_x_tick_labels=show_x_tick_labels,
        show_y_tick_labels=show_y_tick_labels,
        vmax_map=vmax_map,
        default_vmax_percentile=default_vmax_percentile,
        cbar_shrink=cbar_shrink,
        add_scalebar=add_scalebar,
        scalebar_position=scalebar_position,
        scale_length_um=scale_length_um,
        scalebar_x_offset_frac=scalebar_x_offset_frac,
        scalebar_y_offset_frac=scalebar_y_offset_frac,
        fig=fig,
        axs=axs,
        panel_offset=panel_offset,
        total_panels=total_panels,
        finalize_layout=finalize_layout,
    )


def plot_elm_ratiomaps(
    row,
    idx,
    plot_fit_type,
    elms,
    scaler_name='US_IC',
    normalize_by_scaler=False,
    avoid_ratio=1.0,
    calibration_factor_map=None,
    clip_percentiles=(1, 99),
    denominator_mask_percentile=None,
    masked_value=np.nan,
    ncol=3,
    figsize=None,
    show_x_tick_labels=None,
    show_y_tick_labels=None,
    vmax_map=None,
    default_vmax_percentile=99,
    cbar_shrink=0.75,
    add_scalebar=False,
    scalebar_position='bottom-left',
    scale_length_um=None,
    scalebar_x_offset_frac=None,
    scalebar_y_offset_frac=None,
    fig=None,
    axs=None,
    panel_offset=0,
    total_panels=None,
    finalize_layout=True,
):
    return _plot_map_panel_grid(
        row,
        plot_fit_type,
        elms,
        map_getter=_ratio_map_getter(
            normalize_by_scaler=normalize_by_scaler,
            scaler_name=scaler_name,
            avoid_ratio=avoid_ratio,
            calibration_factor_map=calibration_factor_map,
            clip_percentiles=clip_percentiles,
            denominator_mask_percentile=denominator_mask_percentile,
            masked_value=masked_value,
        ),
        scaler_name=scaler_name,
        ncol=ncol,
        figsize=figsize,
        show_x_tick_labels=show_x_tick_labels,
        show_y_tick_labels=show_y_tick_labels,
        vmax_map=vmax_map,
        default_vmax_percentile=default_vmax_percentile,
        cbar_shrink=cbar_shrink,
        add_scalebar=add_scalebar,
        scalebar_position=scalebar_position,
        scale_length_um=scale_length_um,
        scalebar_x_offset_frac=scalebar_x_offset_frac,
        scalebar_y_offset_frac=scalebar_y_offset_frac,
        fig=fig,
        axs=axs,
        panel_offset=panel_offset,
        total_panels=total_panels,
        finalize_layout=finalize_layout,
    )


def plot_scaler_map(row, scaler_name='US_IC'):
    x_axis = np.asarray(row['x_axis'])
    y_axis = np.asarray(row['y_axis'])
    scaler = _match_map_shape(np.asarray(row[scaler_name], dtype=float), x_axis, y_axis)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vmax = np.nanpercentile(scaler, 99)
    img = ax.imshow(scaler, cmap='viridis', vmin=0, vmax=vmax if vmax > 0 else None, aspect='equal')
    ax.set_title(scaler_name)
    _apply_ticks(ax, x_axis, y_axis, show_x_tick_labels=True)
    cbar = fig.colorbar(img, ax=ax, shrink=0.8)
    cbar.set_label(scaler_name)
    fig.tight_layout()
    return fig

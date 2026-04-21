import collections
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _decode_array(values):
    arr = np.asarray(values)
    if arr.dtype.kind in {'S', 'O'}:
        return [v.decode('utf-8', errors='ignore') if isinstance(v, bytes) else str(v) for v in arr.tolist()]
    return arr.astype(str).tolist()


def _read_first_existing(h5_obj, paths):
    last_error = None
    for path in paths:
        try:
            return h5_obj[path][:], path
        except Exception as exc:
            last_error = exc
    raise last_error if last_error is not None else KeyError(paths)


def _extract_scan_num(scan_name):
    stem = Path(scan_name).name
    parts = stem.split('_')
    if len(parts) < 2:
        return None
    try:
        return int(parts[-1].split('.')[0])
    except ValueError:
        return None


def load_h5(
    img_dat_path,
    fit_type=('NNLS', 'Fitted', 'ROI'),
    fsizelim=1e6,
    quant_norm='US_FM',
    add_xbic=False,
    xbic_name='DS_IC',
    add_mca_arr=False,
    scan_nums=None,
):
    data = collections.defaultdict(list)
    img_dat_path = Path(img_dat_path)
    fit_types = [fit_type] if isinstance(fit_type, str) else list(fit_type)
    scan_num_filter = None if scan_nums is None else {int(scan_num) for scan_num in scan_nums}

    for fpath in sorted(img_dat_path.iterdir()):
        if not fpath.is_file():
            continue
        scan_num = _extract_scan_num(fpath.name)
        if scan_num_filter is not None and scan_num not in scan_num_filter:
            continue
        if fpath.stat().st_size <= fsizelim:
            continue

        print(f'processing {fpath.name}')
        try:
            with h5py.File(fpath, 'r') as f:
                try:
                    x_axis, _ = _read_first_existing(f, ['MAPS/Scan/x_axis', 'MAPS/x_axis'])
                    y_axis, _ = _read_first_existing(f, ['MAPS/Scan/y_axis', 'MAPS/y_axis'])
                    int_spec, _ = _read_first_existing(
                        f, ['MAPS/int_spec', 'MAPS/Spectra/Integrated_Spectra/Spectra']
                    )
                    energy_calib, _ = _read_first_existing(
                        f, ['MAPS/energy_calib', 'MAPS/Spectra/Energy_Calibration']
                    )
                except Exception as exc:
                    data['skipped_scan'].append(fpath.name)
                    data['skip_reason'].append(f'missing core XRF datasets: {exc}')
                    continue

                try:
                    scaler_names = _decode_array(f['MAPS/Scalers/Names'][:])
                    scaler_values = f['MAPS/Scalers/Values'][:]
                    quant_norm_value = scaler_values[scaler_names.index(quant_norm), ...]
                except Exception as exc:
                    data['skipped_scan'].append(fpath.name)
                    data['skip_reason'].append(f'missing scaler {quant_norm}: {exc}')
                    continue

                xbic_val = None
                if add_xbic:
                    try:
                        xbic_val = scaler_values[scaler_names.index(xbic_name), ...]
                    except ValueError:
                        print(f'Warning: {xbic_name} not found in {fpath.name}')

                data['scan'].append(fpath.name)
                data['x_axis'].append(np.asarray(x_axis))
                data['y_axis'].append(np.asarray(y_axis))
                data['int_spec'].append(np.asarray(int_spec))
                data['energy_calib'].append(np.asarray(energy_calib))
                if add_mca_arr:
                    try:
                        mca_arr, _ = _read_first_existing(f, ['MAPS/mca_arr', 'MAPS/Spectra/mca_arr'])
                        data['mca_arr'].append(np.asarray(mca_arr))
                    except Exception as exc:
                        data['skipped_scan'].append(fpath.name)
                        data['skip_reason'].append(f'missing mca_arr: {exc}')
                        continue

                for fit_name in fit_types:
                    try:
                        counts = f[f'MAPS/XRF_Analyzed/{fit_name}/Counts_Per_Sec'][:]
                        channel_names = _decode_array(f[f'MAPS/XRF_Analyzed/{fit_name}/Channel_Names'][:])
                        quant_val = f[
                            f'MAPS/Quantification/Calibration/{fit_name}/Calibration_Curve_{quant_norm}'
                        ][:]
                        quant_ch = _decode_array(
                            f[f'MAPS/Quantification/Calibration/{fit_name}/Calibration_Curve_Labels'][0, :]
                        )
                    except Exception as exc:
                        data['skipped_scan'].append(fpath.name)
                        data['skip_reason'].append(f'missing {fit_name} datasets: {exc}')
                        break

                    data[f'{fit_name}_arr'].append(counts)
                    data[f'{fit_name}_ch'].append(channel_names)
                    data[f'{fit_name}_{quant_norm}_quant_arr'].append(quant_val)
                    data[f'{fit_name}_{quant_norm}_quant_ch'].append(quant_ch)
                    if add_xbic:
                        data[f'{fit_name}_xbic'].append(None if xbic_val is None else np.asarray(xbic_val))
                else:
                    data[quant_norm].append(np.asarray(quant_norm_value))
                    continue

                for key in list(data.keys()):
                    if key in {'skipped_scan', 'skip_reason'}:
                        continue
                    if len(data[key]) > len(data['scan']) - 1:
                        data[key].pop()
        except Exception as exc:
            data['skipped_scan'].append(fpath.name)
            data['skip_reason'].append(f'file read error: {exc}')

    return data


def load_fitted_h5(img_dat_path, fit_type='Fitted', scaler_name='US_IC', fsizelim=0, scan_nums=None):
    return load_h5(
        img_dat_path,
        fit_type=fit_type,
        fsizelim=fsizelim,
        quant_norm=scaler_name,
        add_xbic=False,
        add_mca_arr=False,
        scan_nums=scan_nums,
    )


def build_xrf_dataframe(img_dat_path, fit_type='Fitted', scaler_name='US_IC', fsizelim=0, scan_nums=None):
    loaded = load_h5(
        img_dat_path,
        fit_type=fit_type,
        fsizelim=fsizelim,
        quant_norm=scaler_name,
        scan_nums=scan_nums,
    )
    scan_data = {k: v for k, v in loaded.items() if k not in {'skipped_scan', 'skip_reason'}}
    skipped = pd.DataFrame({
        'scan': loaded.get('skipped_scan', []),
        'reason': loaded.get('skip_reason', []),
    })

    if not scan_data.get('scan'):
        return pd.DataFrame(), skipped

    xrf_pd = pd.DataFrame(scan_data, index=scan_data['scan']).drop(columns='scan')
    return xrf_pd, skipped


def load_elms(
    img_dat_path,
    elms,
    fit_type='Fitted',
    fsizelim=1e6,
    quant_norm='US_FM',
    add_xbic=False,
    xbic_name='DS_IC',
    add_mca_arr=False,
    scan_nums=None,
):
    loaded = load_h5(
        img_dat_path,
        fit_type=fit_type,
        fsizelim=fsizelim,
        quant_norm=quant_norm,
        add_xbic=add_xbic,
        xbic_name=xbic_name,
        add_mca_arr=add_mca_arr,
        scan_nums=scan_nums,
    )
    skipped = pd.DataFrame({
        'scan': loaded.get('skipped_scan', []),
        'reason': loaded.get('skip_reason', []),
    })

    if not loaded.get('scan'):
        return pd.DataFrame(), skipped

    fit_arr_key = f'{fit_type}_arr'
    fit_ch_key = f'{fit_type}_ch'
    fit_quant_arr_key = f'{fit_type}_{quant_norm}_quant_arr'
    fit_quant_ch_key = f'{fit_type}_{quant_norm}_quant_ch'
    drop_keys = {fit_arr_key, fit_ch_key, fit_quant_arr_key, fit_quant_ch_key}
    scan_data = {
        k: v for k, v in loaded.items() if k not in {'skipped_scan', 'skip_reason'} | drop_keys
    }

    xrf_pd = pd.DataFrame(scan_data, index=scan_data['scan']).drop(columns='scan')
    source_pd = pd.DataFrame(
        {
            fit_arr_key: loaded[fit_arr_key],
            fit_ch_key: loaded[fit_ch_key],
            fit_quant_arr_key: loaded[fit_quant_arr_key],
            fit_quant_ch_key: loaded[fit_quant_ch_key],
        },
        index=loaded['scan'],
    )

    for elm in elms:
        xrf_pd[elm] = source_pd.apply(lambda row: _get_element_map(row, elm, fit_type=fit_type), axis=1)
        xrf_pd[f'{elm}_{quant_norm}_quant'] = source_pd.apply(
            lambda row: _get_element_quant(row, elm, fit_type=fit_type, quant_norm=quant_norm),
            axis=1,
        )

    return xrf_pd, skipped


def _get_quant_idx(element, quant_ch):
    shell_idx = 0
    if '_L' in element:
        shell_idx = 1
    elif '_M' in element:
        shell_idx = 2

    base_element = element.split('_')[0]
    if base_element not in quant_ch:
        raise KeyError(f'{base_element} not present in quantification channels')
    return shell_idx, quant_ch.index(base_element)


def _get_element_quant(row, element, fit_type='Fitted', quant_norm='US_FM'):
    quant_key = f'{element}_{quant_norm}_quant'
    if quant_key in row.index:
        return row[quant_key]

    quant_arr = np.asarray(row[f'{fit_type}_{quant_norm}_quant_arr'])
    quant_ch = row[f'{fit_type}_{quant_norm}_quant_ch']
    try:
        shell_idx, element_idx = _get_quant_idx(element, quant_ch)
    except KeyError:
        return None
    return float(quant_arr[shell_idx, element_idx])


def _get_element_map(row, element, fit_type='Fitted'):
    if element in row.index:
        return np.asarray(row[element], dtype=float)

    fit_ch_key = f'{fit_type}_ch'
    fit_arr_key = f'{fit_type}_arr'
    if fit_ch_key not in row.index or fit_arr_key not in row.index:
        raise KeyError(
            f'{fit_ch_key} not found. Load raw fit arrays or use load_elms() with {element} included.'
        )

    channels = row[fit_ch_key]
    if element not in channels:
        raise KeyError(f'{element} not present in {fit_type} channels')
    idx = channels.index(element)
    return np.asarray(row[fit_arr_key][idx], dtype=float)

import numpy as np
import h5py
from pathlib import Path
from typing import Any, Mapping, Tuple, List, Union
import os
import collections
import ast

from mic_vis.common.load_xrf import load_h5_file
from .mda import get_mda_positioners


def load_logs(folder_path: str):
    """
    Load logs from a folder.
    
    This function reads log files from a folder and returns a dictionary of log data.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing log files.
    
    Returns
    -------
    dict
        A dictionary containing log data.
    """
    
    files = os.listdir(folder_path)
    log_files = [f for f in files if f.endswith('.log')]
    
    log_data = collections.defaultdict(list)
    for log_file in log_files:
        scan_number = log_file.replace('.log', '')
        log_data['scan_number'].append(scan_number)
        with open(os.path.join(folder_path, log_file), 'r') as f:
            for i, line in enumerate(f):
                if i == 3:
                     # Parse the dict after the timestamp (format: "YYYY-MM-DD HH:MM:SS: dict_str")
                    try:
                        parts = line.split(': ', 1)
                        if len(parts) > 1:
                            scan_params = ast.literal_eval(parts[1].strip())
                            for key, value in scan_params.items():
                                if key != 'id' and key != 'status':
                                    log_data[key].append(value)
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing log file {log_file}: {e}")
    
    return log_data


def load_xrf_h5_file(file_path: str, fit_type: str = 'NNLS') -> Tuple[np.ndarray, List[str], float, float]:
    """
    Load XRF data from an HDF5 file.
    
    This function reads X-ray fluorescence (XRF) data from an HDF5 file, including
    channel data, channel names, and scaler data for upstream and downstream
    ion chambers.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing XRF data.
    fit_type : str, default='NNLS'
        Type of fit analysis to load. Must match a key in the 
        'MAPS/XRF_Analyzed/' group of the HDF5 file.
    
    Returns
    -------
    Tuple[np.ndarray, List[str], float, float]
        A tuple containing:
        - ch_data : np.ndarray
            Channel data as counts per second with shape (n_channels, n_points)
        - ch_names : List[str]
            List of channel names corresponding to the data
        - us_ic : float
            Upstream ion chamber value
        - ds_ic : float
            Downstream ion chamber value
    
    Raises
    ------
    FileNotFoundError
        If the specified file_path does not exist.
    KeyError
        If the required HDF5 groups or datasets are not found in the file.
    ValueError
        If the fit_type is not available in the file.
    
    Examples
    --------
    >>> ch_data, ch_names, us_ic, ds_ic, x_val, y_val = load_xrf_h5_file('data.bnp_fly0001.mda.h5')
    >>> print(f"Loaded {len(ch_names)} channels")
    >>> print(f"Upstream IC: {us_ic}")
    """
    
    
    # Validate file path exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if required groups exist
            if 'MAPS' not in f:
                raise KeyError("Required group 'MAPS' not found in HDF5 file")
            
            if 'XRF_Analyzed' not in f['MAPS']:
                raise KeyError("Required group 'MAPS/XRF_Analyzed' not found in HDF5 file")
            
            if fit_type not in f['MAPS']['XRF_Analyzed']:
                available_fits = list(f['MAPS']['XRF_Analyzed'].keys())
                raise ValueError(f"Fit type '{fit_type}' not found. Available types: {available_fits}")
            
            # Check if required datasets exist
            required_paths = [
                f"MAPS/XRF_Analyzed/{fit_type}/Counts_Per_Sec",
                f"MAPS/XRF_Analyzed/{fit_type}/Channel_Names",
                'MAPS/scalers',
                'MAPS/scaler_names',
                'MAPS/x_axis',
                'MAPS/y_axis',
                'MAPS/energy',
                'MAPS/int_spec'
            ]
            
            for path in required_paths:
                if path not in f:
                    raise KeyError(f"Required dataset '{path}' not found in HDF5 file")
            
            # Load data
            ch_data = f[f"MAPS/XRF_Analyzed/{fit_type}/Counts_Per_Sec"][:]
            ch_names = f[f"MAPS/XRF_Analyzed/{fit_type}/Channel_Names"][:].astype(str).tolist()
            scaler_data = f['MAPS/scalers'][:]
            scaler_names = f['MAPS/scaler_names'][:].astype(str).tolist()
            x_val = f['MAPS/x_axis'][:]
            y_val = f['MAPS/y_axis'][:]
            energy_val = f['MAPS/energy'][:]
            int_spec = f['MAPS/int_spec'][:]
            
            # Check if required scaler names exist
            if 'US_IC' not in scaler_names:
                raise KeyError("Required scaler 'US_IC' not found in scaler_names")
            if 'DS_IC' not in scaler_names:
                raise KeyError("Required scaler 'DS_IC' not found in scaler_names")
            
            us_ic = scaler_data[scaler_names.index('US_IC')]
            ds_ic = scaler_data[scaler_names.index('DS_IC')]
            
            dict_label = ["ch_data", "ch_names", "scaler_data", "scaler_names", 
                          "x_val", "y_val", "energy_val", "int_spec"]
            h5data = {}
            for l in dict_label:
                if l not in locals():
                    raise KeyError(f"Required dataset '{l}' not found in HDF5 file")
                else:
                    h5data[l] = locals()[l]
                    
            return h5data
            
            
    except h5py.HDF5Error as e:
        raise ValueError(f"Error reading HDF5 file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading XRF data: {e}")


BNP_SAMPLE_Z_PVS = ("9idbTAU:SM:SZ:ActPos",)
BNP_SAMPLE_THETA_PVS = ("9idbTAU:SM:ST:ActPos",)


def _decode_string_list(values: np.ndarray) -> list[str]:
    return [str(item) for item in values.astype(str).tolist()]


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_extra_pv_values(file_handle: h5py.File) -> dict[str, str]:
    if "MAPS/extra_pvs" not in file_handle:
        return {}

    extra_pvs = file_handle["MAPS/extra_pvs"][:]
    if getattr(extra_pvs, "shape", ())[:1] < (2,):
        return {}
    pv_names = _decode_string_list(extra_pvs[0])
    pv_values = _decode_string_list(extra_pvs[1])
    return {name: value for name, value in zip(pv_names, pv_values)}


def _find_pv_float(pv_values: Mapping[str, str], candidates: tuple[str, ...]) -> float | None:
    for candidate in candidates:
        if candidate in pv_values:
            value = _coerce_float(pv_values[candidate])
            if value is not None:
                return value
    for pv_name, pv_value in pv_values.items():
        if any(pv_name.endswith(candidate) for candidate in candidates):
            value = _coerce_float(pv_value)
            if value is not None:
                return value
    return None


def _resolve_bnp_mda_path(file_path: str) -> Path:
    h5_path = Path(file_path)
    parent = h5_path.parent
    if parent.name == "img.dat":
        return parent.parent / "mda" / h5_path.stem
    return h5_path.with_suffix("")


def load_bnp_h5_file(file_path: str, fit_type: str = "ROI") -> dict[str, Any]:
    data = dict(load_h5_file(file_path, fit_type))
    sample_z = None
    sample_theta = None

    with h5py.File(file_path, "r") as file_handle:
        pv_values = _extract_extra_pv_values(file_handle)
        if pv_values:
            sample_z = _find_pv_float(pv_values, BNP_SAMPLE_Z_PVS)
            sample_theta = _find_pv_float(pv_values, BNP_SAMPLE_THETA_PVS)

    if sample_z is None or sample_theta is None:
        mda_path = _resolve_bnp_mda_path(file_path)
        if mda_path.exists():
            try:
                positioners = get_mda_positioners(
                    str(mda_path),
                    get_z=True,
                    samz_pv=BNP_SAMPLE_Z_PVS[0],
                    get_theta=True,
                    theta_pv=BNP_SAMPLE_THETA_PVS[0],
                )
            except Exception:
                positioners = {}
            if sample_z is None:
                sample_z = _coerce_float(positioners.get("z_pos"))
            if sample_theta is None:
                sample_theta = _coerce_float(positioners.get("theta_pos"))

    data["sample_z"] = sample_z
    data["sample_theta"] = sample_theta
    return data

import numpy as np
import h5py
from typing import Tuple, List, Union
import os


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


import h5py
import os

def load_h5_file(file_path: str, fit_type: str = 'ROI'):
    """
    Load XRF data from an HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        fit_type: Type of fit analysis to load
    """

    layout_func = [("non_v9", load_h5_file_non_v9), ("v9", load_h5_file_v9)]
    for layout, func in layout_func:
        try:
            data = func(file_path, fit_type)
            print(f"Successfully loaded XRF data from {file_path} with {layout} layout")
            return data
        except:
            print(f"Failed to load XRF data from {file_path} with {layout} layout")
            continue
        
    raise RuntimeError(f"No valid layout found for file {file_path}")



def load_h5_file_v9(file_path: str, fit_type: str = 'ROI'):
    """
    Load XRF data from an HDF5 file with v9 layout.
    
    Args:
        file_path: Path to the HDF5 file
        fit_type: Type of fit analysis to load
    """
    
    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        data = f[f"MAPS/XRF_Analyzed/{fit_type}/Counts_Per_Sec"][:]
        ch_names = f[f"MAPS/XRF_Analyzed/{fit_type}/Channel_Names"][:].astype(str).tolist()
        scaler_data = f['MAPS/scalers'][:]
        scaler_names = f['MAPS/scaler_names'][:].astype(str).tolist()
        x_val = f['MAPS/x_axis'][:]
        y_val = f['MAPS/y_axis'][:]

    dict_label = ["data", "ch_names", "scaler_data", "scaler_names", 
                  "x_val", "y_val"]
    for l in dict_label:
        if l not in locals():
            raise KeyError(f"Required dataset '{l}' not found in HDF5 file")
        else:
            data_dict[l] = locals()[l]

    return data_dict


def load_h5_file_non_v9(file_path: str, fit_type: str = 'ROI'):
    """
    Load XRF data from an HDF5 file without v9 layout.
    
    Args:
        file_path: Path to the HDF5 file
        fit_type: Type of fit analysis to load
    """
    
    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        data = f[f'MAPS/XRF_Analyzed/{fit_type}/Counts_Per_Sec'][:]
        ch_names = f[f'MAPS/XRF_Analyzed/{fit_type}/Channel_Names'][:].astype(str).tolist()
        scaler_data = f['MAPS/Scalers/Values'][:]
        scaler_names = f['MAPS/Scalers/Names'][:].astype(str).tolist()
        x_val = f['MAPS/Scan/x_axis'][:]
        y_val = f['MAPS/Scan/y_axis'][:]

    dict_label = ["data", "ch_names", "scaler_data", "scaler_names", 
                  "x_val", "y_val"]
    for l in dict_label:
        if l not in locals():
            raise KeyError(f"Required dataset '{l}' not found in HDF5 file")
        else:
            data_dict[l] = locals()[l]

    return data_dict



import tifffile
import numpy as np
from mic_vis.bnp.mda import get_mda_positioners

def load_ptycho_tiff(tiff_filename: str) -> np.ndarray:
    tiff = tifffile.imread(tiff_filename)

    if tiff.dtype.kind == "u":
        vmax = np.iinfo(tiff.dtype).max
        norm = tiff.astype(np.float32) / vmax
        tiff = norm
        print(f"dtype of tiff is {tiff.dtype}")
        print(f"vmax: {tiff.max()}")
        print(f"vmin: {tiff.min()}")
    
    return tiff 

def load_mda(mda_filename: str, tiff: np.ndarray = None) -> dict[str, np.ndarray]:
    positioners = get_mda_positioners(mda_filename, get_theta=False, get_z=False)
    if tiff is not None:
        y = np.linspace(positioners["y_pos"][1], positioners["y_pos"][-1], tiff.shape[0])
        x = np.linspace(positioners["x_pos"][0], positioners["x_pos"][-1], tiff.shape[1])
    else:
        y = positioners["y_pos"]
        x = positioners["x_pos"]
    return {"x": x, "y": y}

def load_ptycho_data(tiff_filename: str, mda_filename: str) -> dict[str, np.ndarray]:
    try:
        tiff = load_ptycho_tiff(tiff_filename)
        position_info = load_mda(mda_filename, tiff)
        return {"data": tiff, "x_val": position_info["x"], "y_val": position_info["y"]}
    except Exception as e:
        print(e)
        return None

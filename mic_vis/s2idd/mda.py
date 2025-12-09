from mic_vis.common.readMDA import readMDA
import numpy as np

def get_roi_from_mda(
    mda_path: str, 
    roi_num: int = 16, 
    return_position: bool = True,
    verbose: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the ROI from the MDA file. This has been tested on the 1D line mda file

    Parameters
    ----------
    mda_path : str
        The path to the MDA file.
    roi_num : int
        The ROI number of the elements of interest.
    return_position : bool
        If True, the position data will be returned.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The ROI data and the position data. If return_position is False, the position data will be None.
    """
    roi_data = None
    position_data = None
    try:
        mda_data = readMDA(mda_path, verbose=verbose)
        for d in mda_data[1].d:
            if f'R{roi_num}' in d.name:
                print(f"Found ROI {roi_num} in {d.name}")
                roi_data = np.array(d.data)
                break
        if return_position:
            p = mda_data[1].p[0]
            position_data = np.array(p.data)
        return roi_data, position_data
    except Exception as e:
        print(f"Error getting ROI from MDA file: {e}")
        return roi_data, position_data
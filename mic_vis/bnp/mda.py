"""Collection of functions that retrieve data from MDA file"""

from mic_vis.common.readMDA import readMDA
import numpy as np

def get_mda_positioners(mda_file: str, 
                        get_z: bool = True, samz_pv: str = "21:D3:SM:SZ:ActPos",
                        get_theta: bool = True, theta_pv: str = "9idbTAU:SM:CT:RqsPos") -> tuple[np.ndarray, np.ndarray]:
    """Get the positioners from the MDA file
    
    Args:
        mda_file: Path to the MDA file
        get_z: Whether to get the z position
        samz_pv: PV name of the z position
        get_theta: Whether to get the theta position
        theta_pv: PV name of the theta position
    Returns:
        dict: Dictionary containing the positioners
    """
    z_pos = None
    theta_pos = None
    y_pos = None
    x_pos = None

    try:
        mda = readMDA(mda_file, verbose=0)
        y_pos = np.array(mda[1].p[0].data)
        x_pos = np.array(mda[2].p[0].data)[0]
    except:
        raise ValueError(f"MDA file {mda_file} is not valid")

    if get_z:
        for d_ in mda[2].d:
            if d_.name == samz_pv:
                z_pos = np.array(d_.data)[0][0]
                break

    if get_theta:
        for d_ in mda[1].d:
            if d_.name == theta_pv:
                theta_pos = np.array(d_.data)[0]
                break

    return {"y_pos": y_pos, "x_pos": x_pos, "z_pos": z_pos, "theta_pos": theta_pos}


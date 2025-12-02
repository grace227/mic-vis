import os
import h5py
import pandas as pd
import collections
import yaml

def load_bluesky_nexus(bluesky_dir: str, export_csv: bool = False) -> pd.DataFrame:
    """
    Load the bluesky nexus file and return a pandas dataframe.
    """

    if not os.path.exists(bluesky_dir):
        raise FileNotFoundError(f"Bluesky directory {bluesky_dir} not found")
    
    files = os.listdir(bluesky_dir)
    meta_dict = collections.defaultdict(list)
    
    for fn in files:
        if fn.endswith('.h5'):
            try:
                with h5py.File(os.path.join(bluesky_dir, fn), 'r') as f:
                    meta_dict['scan_number'].append(fn.replace('_run.h5', ''))
                    meta_dict['start_time'].append(f['entry/start_time'][()].decode('utf-8'))
                    meta_dict['plan_name'] = f['entry/plan_name'][()].decode('utf-8')
                    m = f['entry/instrument/bluesky/metadata/plan_args'][()].decode('utf-8')
                    m_dict = yaml.safe_load(m)
                    for k, v in m_dict.items():
                        meta_dict[k].append(v)
            except Exception as e:
                print(f"Error loading bluesky nexus file {fn}: {e}")
                continue
    
    df = pd.DataFrame(meta_dict)
    if export_csv:
        parent_dir = os.path.dirname(bluesky_dir)
        df.to_csv(os.path.join(parent_dir, 'bluesky_meta.csv'), index=False)
    return df
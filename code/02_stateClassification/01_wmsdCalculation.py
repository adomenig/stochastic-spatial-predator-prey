import os
import sys
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.io as pio
pio.renderers.default = "browser"
from multiprocessing import Pool
from haversine import haversine, Unit


### Getting the home directory from the bash script ##
if len(sys.argv) < 2:
    print("Usage: python process_lynx.py /path/to/home_directory")
    sys.exit(1)

home_dir = Path(sys.argv[1])
sys.path.insert(0, str(home_dir))
import helper_functions # type:ignore

data_path = Path(f"{home_dir}/data/processed/dataCleaning/final_lynx_df.csv")

colorscheme = ["#8fd7d7", "#00b0be", "#ff8ca1", "#f45f74", "#bdd373", "#98c127", "#ffcd8e", "#ffb255", "#c084d4"] 


############################ CALCULATING WMSD START #######################################
def compute_msd_parallel(df, w_hours=12, tau_hours=4, n_cores=None):
    """
    This function is a wrapper function to call the compute_single_lynx_msd function in parallel
    """
    if n_cores is None:
        n_cores = max(1, os.cpu_count() - 2)

    grouped = [(lynx_id, group, w_hours, tau_hours) for lynx_id, group in df.groupby("ID")]

    with Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap(compute_single_lynx_msd, grouped), total=len(grouped)))

    return {lynx_id: (time, msd) for lynx_id, time, msd in results if time is not None}

def compute_single_lynx_msd(args):
    lynx_id, traj_df, w_hours, tau_hours = args
    # just a safety check
    if w_hours < tau_hours:
        raise ValueError(f"Window size ({w_hours}h) must be >= lag time ({tau_hours}h)")

    traj = traj_df.sort_values("Time").copy()
    traj["time_float"] = (traj["Time"] - traj["Time"].iloc[0]).dt.total_seconds() / 3600
    time = traj["time_float"].values
    coords = traj[["Lat", "Long"]].values
    msd = np.full(len(time), np.nan)

    for idx in range(len(time)):
        # create a window of size w
        t_start = time[idx] 
        t_end = t_start + w_hours
         
        # get all the indices within that window
        idxs = np.where((time >= t_start) & (time <= t_end))[0]

        # now we take all valid pairs that are tau time apart
        valid_pairs = [
            (i, j) for i in idxs for j in idxs
            if i < j and np.isclose(time[j] - time[i], tau_hours, atol = 0.5)
        ]

        # if there are no valid pairs, we skip this point and keep it nan
        if not valid_pairs:
            continue

        # calculate the squared displacement
        dists_sq = [
            haversine(tuple(coords[i]), tuple(coords[j]), unit=Unit.KILOMETERS)**2
            for i, j in valid_pairs
        ]
        # get the median
        msd[idx] = np.median(dists_sq)

    # center the time points
    time_centered = time + w_hours / 2
    return lynx_id, time_centered, msd
############################ CALCULATING WMSD END #######################################


if __name__ == "__main__":
    df = pd.read_csv(data_path, parse_dates=["Time"])
    wmsd_params = [(24 * 7, 24 * 7 * 2), (24 * 7, 24 * 30)]  # wmsd values in hours (1 week and 2 weeks, 1 week and 1 month/30 days)
    wmsd_cache = Path(f"{home_dir}/data/processed/stateClassification")
    wmsd_cache.mkdir(parents=True, exist_ok=True)
    # tau is our lag size, w is our window size. we use these parameters to calculate our windowed median squared displacement for all lynx
    # and cache it. Since this calculation takes incredibly long, we cache the file and only run if if it hasn't already been calculated. 
    for tau, w in wmsd_params:
        fname = f"msd_tau{tau}_w{w}.pkl"
        cache_file = wmsd_cache / fname

        if cache_file.exists():
            print(f"MSD file for tau={tau}, w={w} already exists at {cache_file}, skipping computation.")
            continue

        print(f"Calculating MSD for tau={tau}, w={w}")
        msd_result = compute_msd_parallel(df, w_hours=w, tau_hours=tau, n_cores=os.cpu_count() - 1)
        with open(cache_file, "wb") as f:
            pickle.dump(msd_result, f)
        print(f"Saved MSD to {cache_file}")

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.io as pio
pio.renderers.default = "browser"

### Getting the home directory from the bash script ##
if len(sys.argv) < 2:
    print("Usage: python process_lynx.py /path/to/home_directory")
    sys.exit(1)

home_dir = Path(sys.argv[1])
sys.path.insert(0, str(home_dir))
#import helper_functions  # type:ignore

data_path = Path(f"{home_dir}/data/processed/dataCleaning/final_lynx_df.csv")
out_path = Path(f"{home_dir}/data/processed/stateClassification")
out_path.mkdir(parents=True, exist_ok=True)


colorscheme = ["#8fd7d7", "#00b0be", "#ff8ca1", "#f45f74", "#bdd373", "#98c127", "#ffcd8e", "#ffb255", "#c084d4"] 


def assign_states(df, msd_1w_2w, msd_1w_1m):
    """
    Assign behavioral states to all lynx trajectories based on two windowed
    MSD thresholds. We use the coefficient of variation to find suitable thresholds
    in both WMSD arrays and then assign states based on these thresholds, where anything
    above the threshold is considered a "high movement" state that we call exploratory.
    """
    all_states = {}
    all_trajs = {}
    
    # extract just the MSD values from the tuples
    all_msd_long_values = [msd for _, msd in msd_1w_1m.values() if msd is not None]
    all_msd_short_values = [msd for _, msd in msd_1w_2w.values() if msd is not None]

    # get all valid MSD values
    all_msd_long_valid = np.concatenate([arr[~np.isnan(arr)] for arr in all_msd_long_values if arr is not None])
    all_msd_short_valid = np.concatenate([arr[~np.isnan(arr)] for arr in all_msd_short_values if arr is not None])
    
    # minimizing the coefficient of variation to find the best thresholds in both msds
    print("Finding best thresholds.")
    msd_long_thresh, percentile_long = find_best_threshold(all_msd_long_valid)
    msd_short_thresh, percentile_short = find_best_threshold(all_msd_short_valid)

    print(f"Threshold for long WMSD: {percentile_long}")
    print(f"Threshold for short WMSD: {percentile_short} \n")

    # now we get to the actual state assigning part
    print("Assigning states.")
    for lynx_id in msd_1w_1m.keys(): # this includes all the lynx so we're still left with 142 trajectories
        traj = df[df['ID'] == lynx_id].copy().sort_values('Time')
        traj = traj.reset_index(drop=True)

        # get just the MSD arrays 
        _, msd_long = msd_1w_1m[lynx_id] if lynx_id in msd_1w_1m else (None, None)
        _, msd_short = msd_1w_2w[lynx_id] if lynx_id in msd_1w_2w else (None, None)
        
        # we first initiate all the states as nan
        states = np.full(len(traj), np.nan)
        # then, we check if it's above or below the threshold
        for i in range(len(traj)):
            # get the calculated wmsd values for these timepoints
            val_long = msd_long[i]
            val_short = msd_short[i]
            # if it's above the threshold for either WMSD, we assign it to the high movement state,
            # i.e. state 2 (which we refer to as exploratory)
            if val_long >= msd_long_thresh or val_short >= msd_short_thresh:
                states[i] = 2
            else:
            # otherwise, we assign state 1
                states[i] = 1
        
        # we run a smoothing algorithm over our data. This is because we want to avoid 
        # excessive swithing. We choose a smoothing filter of 2 weeks (which corresponds to
        # 82 since 1 timestep is 4 hours), but this is fully tunable. Smaller tuning filters means more noise. 
        states = smoothing(states, min_segment_length=84)
        all_states[lynx_id] = states
        all_trajs[lynx_id] = traj
    print("States assigned.")
    return all_states, all_trajs



def smoothing(states, min_segment_length):
    """
    We want to smooth over the trajectories so we don't get short blips of states. We 
    chose 2 weeks as our smoothing parameter, where if any state is assigned for less
    than 2 week, we attempt to smooth it. 
    """
    states = np.array(states)
    smoothed = states.copy()
    
    # handle nan segments surrounded by the same state by just overwriting them with that state
    isnan = np.isnan(states)
    i = 0
    while i < len(states):
        if isnan[i]:
            start = i
            while i < len(states) and isnan[i]:
                i += 1
            end = i

            left = states[start - 1] if start > 0 else np.nan
            right = states[end] if end < len(states) else np.nan

            if not np.isnan(left) and left == right:
                smoothed[start:end] = left
        else:
            i += 1

    # now we want to smooth non-nan segments
    isnan = np.isnan(smoothed)
    changes = np.diff(smoothed)
    boundaries = np.where((~isnan[:-1]) & (changes != 0))[0] + 1
    segment_starts = np.insert(boundaries, 0, 0)
    segment_ends = np.append(boundaries, len(smoothed))
    
    for start, end in zip(segment_starts, segment_ends):
        segment_len = end - start
        curr_state = smoothed[start]

        if np.isnan(curr_state) or segment_len >= min_segment_length:
            continue

        left_state = smoothed[start - 1] if start > 0 else np.nan # we check the left and right neighbors
        right_state = smoothed[end] if end < len(smoothed) else np.nan
        
        # if left state and right state are identical, you overwrite it
        if not np.isnan(left_state) and left_state == right_state and left_state != curr_state:
            smoothed[start:end] = left_state
        # if the left neighboring segment is longer than the current state and 
        # longer than the minimum segment length, then you can overwrite the current segment 
        elif not np.isnan(left_state) and left_state != curr_state:
            l = start - 1
            while l >= 0 and smoothed[l] == left_state:
                l -= 1
            if (start - 1 - l) >= min_segment_length:
                smoothed[start:end] = left_state
        # do the same thing for the right neighbor
        elif not np.isnan(right_state) and right_state != curr_state:
            r = end
            while r < len(smoothed) and smoothed[r] == right_state:
                r += 1
            if (r - end) >= min_segment_length:
                smoothed[start:end] = right_state
    return smoothed


def find_best_threshold(msd_values):
    """
    Finds the best threshold using the coefficient of variation. We iterate thorugh
    each possible percentile-based split into two groups and choose the one that
    has the smallest coefficient of variation. 
    """
    msd_values = np.array(msd_values)
    best_score = np.inf  # we're minimizing so start at infinity
    best_threshold = None
    best_percent = None
    scores = []
    thresholds = []

    for percent in range(1, 100):  # use the full percentile range
        # set the threshold
        threshold = np.percentile(msd_values, percent)
        group1 = msd_values[msd_values <= threshold]
        group2 = msd_values[msd_values > threshold]

        # get the means and standard deviations
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1), np.std(group2)

        # calculate the coefficient of variation in either group
        cv1 = std1 / mean1 if mean1 != 0 else np.inf
        cv2 = std2 / mean2 if mean2 != 0 else np.inf

        # calculate the sum of the two scores
        score = cv1 + cv2  # lower is better
        scores.append(score)
        thresholds.append(threshold)
        
        # get the smallest possible score from the entire range
        if score < best_score:
            best_score = score
            best_threshold = threshold
            best_percent = percent

    return best_threshold, best_percent


if __name__ == "__main__":
    df = pd.read_csv(data_path, parse_dates=["Time"])
    df = df.sort_values(["ID", "Time"]).reset_index(drop=True)

    wmd_params = [(24 * 7, 24 * 7 * 2), (24 * 7, 24 * 30)]  # msd values in hours (1 week and 2 weeks, 1 week and 1 month/30 days)
    wmd_cache = Path(f"{home_dir}/data/processed/stateClassification")

    # define the WMD filenames
    wmd_files = {"msd_1w_2w": "msd_tau168_w336.pkl", "msd_1w_1m": "msd_tau168_w720.pkl"}

    # load the pickled WMD results
    msd_results = {}
    for key, fname in wmd_files.items():
        with open(wmd_cache / fname, "rb") as f:
            msd_results[key] = pickle.load(f)

    # assign our states
    all_states, all_trajs = assign_states(df, msd_1w_2w=msd_results["msd_1w_2w"], msd_1w_1m=msd_results["msd_1w_1m"])
    
    # create a new column in the original df
    df["State"] = np.nan 
    for lynx_id, states in all_states.items(): 
        idxs = df[df["ID"] == lynx_id].index 
        df.loc[idxs, "State"] = states

    # save the new csv that is identical to the old one just with a new states column
    df.to_csv(out_path / "final_lynx_with_states.csv", index=False)
    print(f"Saved dataframe with states to {out_path / 'final_lynx_with_states.csv'}")

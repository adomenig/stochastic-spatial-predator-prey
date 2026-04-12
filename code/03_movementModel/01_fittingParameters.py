import sys
from pathlib import Path
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.stats import expon
from sklearn.linear_model import LogisticRegression

### Getting the home directory from the bash script ##
if len(sys.argv) < 2:
    print("Usage: python process_lynx.py /path/to/home_directory")
    sys.exit(1)

home_dir = Path(sys.argv[1])
sys.path.insert(0, str(home_dir))
import helper_functions  # type:ignore

# define paths
data_path = Path(f"{home_dir}/data/processed/stateClassification/final_lynx_with_states.csv")
colorscheme = ["#8fd7d7", "#00b0be", "#ff8ca1", "#f45f74", "#bdd373", "#98c127", "#ffcd8e", "#ffb255", "#c084d4"]

################################ SIMULATION LOGIC END ##################################
# state simulation functions
def simulate_state_trajectory(state_type, n_points, dt=4, **params):
    """
    Generic simulation for states 1, 2, or 3.

    Here, state 2 is our exploratory state and state 1 is our stationary state. State 
    3 is still within the exploratory phase, but it's part of the return loop so 
    we simulate it as state 1.
    """
    pos = np.zeros((n_points + 1, 2)) #normalize it
    if state_type == 2:
        # exploratory movement
        theta = np.random.uniform(0, 2*np.pi)
        v0 = np.random.uniform(params["v_lower"], params["v_higher"])
        D2 = np.random.uniform(params["D2_lower"], params["D2_higher"])
        Dtheta = np.random.uniform(params["Dtheta_lower"], params["Dtheta_higher"])
        for t in range(1, n_points + 1):
            theta += np.sqrt(2 * Dtheta * dt) * np.random.randn()
            velocity = v0 * np.array([np.cos(theta), np.sin(theta)])
            pos[t] = pos[t-1] + velocity * dt + np.sqrt(2 * D2 * dt) * np.random.randn(2)

    else:
        # state 1 or 3 (stationary / returning loop)
        X_home = params.get("X_home", np.array([0.0, 0.0] if state_type == 1 else [1e9, 1e9]))
        alpha = params["alpha"]
        D1 = np.random.uniform(params["D1_lower"], params["D1_higher"])
        for t in range(1, n_points + 1):
            displacement = X_home - pos[t-1]
            norm = np.linalg.norm(displacement)
            drift = alpha * (displacement / norm) if norm != 0 else np.zeros(2)
            pos[t] = pos[t-1] + drift * dt + np.sqrt(2 * D1 * dt) * np.random.randn(2)
    return pos
################################ SIMULATION LOGIC END ##################################






####################### START OF MSD CALCULATION LOGIC FOR EMPIRICAL DATA ############################
def extract_state_msds_from_df(df):
    """
    Compute MSDs per state using real timestamps.
    """
    state_data = defaultdict(lambda: {"msds": [], "lags": [], "ids": []})

    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])

    for lynx_id, group in df.groupby("ID"):

        group = group.sort_values("Time")

        states = group["State_Loop_Split"].values
        times = group["Time"].values

        coords = np.column_stack([group["Lat"].values, group["Long"].values])
        xy_meters = helper_functions.project_to_alaska_albers(coords)

        for state_val in np.unique(states):
            in_state = False
            start_idx = None

            for i, s in enumerate(states):
                if s == state_val:
                    if not in_state:
                        start_idx = i
                        in_state = True
                else:
                    if in_state:

                        segment = xy_meters[start_idx:i]
                        seg_time = times[start_idx:i]

                        if len(segment) < 2:
                            in_state = False
                            continue

                        msd_vals = helper_functions.compute_msd(
                            segment,
                            max_lag_steps=len(segment)//2
                        )

                        # TRUE time-aware lags (in hours)
                        t0 = seg_time[0]
                        lag_hours = np.array([
                            (seg_time[j] - t0) / np.timedelta64(1, "h")
                            for j in range(1, len(msd_vals) + 1)
                        ])

                        state_data[state_val]["msds"].append(msd_vals)
                        state_data[state_val]["lags"].append(lag_hours)
                        state_data[state_val]["ids"].append(lynx_id)

                        in_state = False

            # handle trailing segment
            if in_state:

                segment = xy_meters[start_idx:]
                seg_time = times[start_idx:]

                if len(segment) >= 2:

                    msd_vals = helper_functions.compute_msd(
                        segment,
                        max_lag_steps=len(segment)//2
                    )

                    t0 = seg_time[0]
                    lag_hours = np.array([
                        (seg_time[j] - t0) / np.timedelta64(1, "h")
                        for j in range(1, len(msd_vals) + 1)
                    ])
                    msd_km2_vals = msd_vals / 1e6
                    state_data[state_val]["msds"].append(msd_km2_vals)
                    state_data[state_val]["lags"].append(lag_hours)
                    state_data[state_val]["ids"].append(lynx_id)

    return state_data
####################### END OF MSD CALCULATION LOGIC FOR EMPIRICAL DATA ############################








########################### MOVEMENT MODEL PARAMETER FITTING START #################
def simulate_msd(state_type, params, n_points, n_sim):
    """
    Run multiple simulations for a given state and return array of MSDs
    """
    msds = []
    for _ in range(n_sim):
        traj = simulate_state_trajectory(state_type, n_points, **params)
        msds.append(helper_functions.compute_msd(traj, max_lag_steps=n_points))
    return np.array(msds)

# helper functions
def aggregate_msd(empirical_msds, empirical_lags, min_n=15):
    """
    Aggregate MSDs by lag, return lags, mean and std values.
    """
    msd_by_lag = defaultdict(list)
    for msd, lags in zip(empirical_msds, empirical_lags):
        for lag_val, msd_val in zip(lags, msd):
            if np.isfinite(msd_val):
                msd_by_lag[lag_val].append(msd_val)

    used_lags = sorted([lag for lag, values in msd_by_lag.items() if len(values) >= min_n])
    empirical_mean = np.array([np.mean(msd_by_lag[lag]) for lag in used_lags])
    empirical_std = np.array([np.std(msd_by_lag[lag]) for lag in used_lags])

    return np.array(used_lags), empirical_mean, empirical_std


# grid search parameters search
def grid_search_state2(empirical_msds, empirical_lags,
                       #v_lower=[0.01], v_higher=[3.0],
                       v_lower= np.arange(0.01, 0.11, 0.01),
                       v_higher= np.arange(0.05, 0.16, 0.01),
                       D2_lower=np.linspace(0.01, 0.05, 3),   
                       D2_higher=np.linspace(0.05, 0.15, 3),   
                       Dtheta_lower=np.linspace(0.0005, 0.0006, 3), 
                       Dtheta_higher=np.linspace(0.00060, 0.0007, 3), 
                       n_simulations=163,
                       dt=4):

    # aggregate empirical MSDs
    used_lags, empirical_mean, empirical_std = aggregate_msd(empirical_msds, empirical_lags)

    param_grid = list(product(v_lower, v_higher, D2_lower, D2_higher, Dtheta_lower, Dtheta_higher))

    best_score = float('inf')
    best_params = None
    results = []

    for v_l, v_h, D2_l, D2_h, Dth_l, Dth_h in tqdm(param_grid, desc="State2 grid search"):

        # simulate MSDs
        sim_msds = simulate_msd(
            2,
            {
                "v_lower": v_l, "v_higher": v_h,
                "D2_lower": D2_l, "D2_higher": D2_h,
                "Dtheta_lower": Dth_l, "Dtheta_higher": Dth_h
            },
            n_points=len(used_lags),
            n_sim=n_simulations
        )

        sim_mean = np.nanmean(sim_msds, axis=0)
        sim_std  = np.nanstd(sim_msds, axis=0)

        # construct simulation lag axis (in hours)
        sim_lags = np.arange(1, len(sim_mean) + 1) * dt

        # interpolate simulation onto empirical lags
        sim_mean_interp = np.interp(used_lags, sim_lags, sim_mean)
        sim_std_interp  = np.interp(used_lags, sim_lags, sim_std)

        # weights: inverse variance
        weights = 1.0 / (empirical_std**2 + 1e-6)

        # log-space comparison
        log_diff_mean = np.log(sim_mean_interp + 1e-6) - np.log(empirical_mean + 1e-6)
        log_diff_std  = np.log(sim_std_interp + 1e-6)  - np.log(empirical_std + 1e-6)

        loss = np.mean(weights * (log_diff_mean**2 + log_diff_std**2))

        results.append({
            "params": (v_l, v_h, D2_l, D2_h, Dth_l, Dth_h),
            "loss": loss
        })

        if loss < best_score:
            best_score = loss
            best_params = {
                "v_lower": v_l, "v_higher": v_h,
                "D2_lower": D2_l, "D2_higher": D2_h,
                "Dtheta_lower": Dth_l, "Dtheta_higher": Dth_h,
                "loss": loss
            }

    # report top results
    results_sorted = sorted(results, key=lambda x: x["loss"])[:5]
    print("\nTop 5 state2 parameter sets:")
    for i, res in enumerate(results_sorted, 1):
        print(f"{i}. {res['params']}, loss={res['loss']:.5f}")

    return best_params

def grid_search_state1(empirical_msds, empirical_lags, 
                       #alpha=np.array([0.07]),
                       alpha = np.arange(0.05, 0.11, 0.01),  
                       D1_lower = np.arange(0.010, 0.051, 0.01),     
                       D1_higher = np.arange(0.10, 0.15, 0.01),
                       n_simulations=200,
                       dt=4):

    used_lags, empirical_mean, empirical_std = aggregate_msd(empirical_msds, empirical_lags)
    param_grid = list(product(alpha, D1_lower, D1_higher))

    best_score = float('inf')
    best_params = None
    results = []

    for alpha_val, D1_l, D1_h in tqdm(param_grid, desc="State1 grid search"):

        sim_msds = simulate_msd(
            1,
            {
                "alpha": alpha_val,
                "D1_lower": D1_l,
                "D1_higher": D1_h
            },
            n_points=len(used_lags),
            n_sim=n_simulations
        )

        sim_mean = np.nanmean(sim_msds, axis=0)
        sim_std  = np.nanstd(sim_msds, axis=0)

        # lag alignment
        sim_lags = np.arange(1, len(sim_mean) + 1) * dt
        sim_mean_interp = np.interp(used_lags, sim_lags, sim_mean)
        sim_std_interp  = np.interp(used_lags, sim_lags, sim_std)

        # weights (inverse variance)
        weights = 1.0 / (empirical_std**2 + 1e-6)

        # log differences
        log_diff_mean = np.log(sim_mean_interp + 1e-6) - np.log(empirical_mean + 1e-6)
        log_diff_std  = np.log(sim_std_interp + 1e-6)  - np.log(empirical_std + 1e-6)

        loss = np.mean(weights * (log_diff_mean**2 + log_diff_std**2))

        results.append({
            "params": (alpha_val, D1_l, D1_h),
            "loss": loss
        })

        if loss < best_score:
            best_score = loss
            best_params = {
                "alpha": alpha_val,
                "D1_lower": D1_l,
                "D1_higher": D1_h,
                "loss": loss
            }

    results_sorted = sorted(results, key=lambda x: x["loss"])[:5]
    print("\nTop 5 state1 parameter sets:")
    for i, res in enumerate(results_sorted, 1):
        print(f"{i}. {res['params']}, loss={res['loss']:.6f}")

    return best_params

def grid_search_state3(empirical_msds, empirical_lags, 
                       alpha=np.arange(0.05, 0.16, 0.01),  
                       D1_lower=np.array([0.05]),
                       D1_higher=np.array([0.14]),
                       n_simulations=200,
                       dt=4):

    used_lags, empirical_mean, empirical_std = aggregate_msd(empirical_msds, empirical_lags)
    param_grid = list(product(alpha, D1_lower, D1_higher))

    best_score = float('inf')
    best_params = None
    results = []

    for alpha_val, D1_l, D1_h in tqdm(param_grid, desc="State3 grid search"):

        sim_msds = simulate_msd(
            3,
            {
                "alpha": alpha_val,
                "D1_lower": D1_l,
                "D1_higher": D1_h
            },
            n_points=len(used_lags),
            n_sim=n_simulations
        )

        sim_mean = np.nanmean(sim_msds, axis=0)
        sim_std  = np.nanstd(sim_msds, axis=0)

        # lag alignment
        sim_lags = np.arange(1, len(sim_mean) + 1) * dt
        sim_mean_interp = np.interp(used_lags, sim_lags, sim_mean)
        sim_std_interp  = np.interp(used_lags, sim_lags, sim_std)

        # weights (inverse variance)
        weights = 1.0 / (empirical_std**2 + 1e-6)

        # log differences
        log_diff_mean = np.log(sim_mean_interp + 1e-6) - np.log(empirical_mean + 1e-6)
        log_diff_std  = np.log(sim_std_interp + 1e-6)  - np.log(empirical_std + 1e-6)

        loss = np.mean(weights * (log_diff_mean**2 + log_diff_std**2))

        results.append({
            "params": (alpha_val, D1_l, D1_h),
            "loss": loss
        })

        if loss < best_score:
            best_score = loss
            best_params = {
                "alpha": alpha_val,
                "D1_lower": D1_l,
                "D1_higher": D1_h,
                "loss": loss
            }

    results_sorted = sorted(results, key=lambda x: x["loss"])[:5]
    print("\nTop 5 state3 parameter sets:")
    for i, res in enumerate(results_sorted, 1):
        print(f"{i}. {res['params']}, loss={res['loss']:.6f}")

    return best_params
########################### MOVEMENT MODEL PARAMETER FITTING START #################






##################  TRANSITION RATES START #############################
def compute_transition_rates_collapsed(all_states, all_trajs, dt_hours=4):
    """
    Same as your function, but collapses states 1 and 3 into a single state (1),
    and keeps state 2 as-is.
    """
    transitions_count = defaultdict(int)
    state_time_totals = Counter()
    transitions_detail = []

    for lynx_id, traj in all_trajs.items():
        states = all_states[lynx_id]
        traj = traj.copy()
        traj["Time"] = pd.to_datetime(traj["Time"])

        if len(traj) != len(states):
            print(f"Skipping {lynx_id} due to length mismatch.")
            continue

        # 🔑 collapse states
        collapsed_states = np.where(np.isin(states, [1, 3]), 1, 2)

        df = pd.DataFrame({
            "Time": traj["Time"],
            "state": collapsed_states
        })
        df = df.sort_values("Time").reset_index(drop=True)

        # Regular 4-hour grid
        t_start, t_end = df["Time"].iloc[0], df["Time"].iloc[-1]
        regular_times = pd.date_range(start=t_start, end=t_end, freq=f"{dt_hours}h")

        df_interp = pd.merge_asof(
            pd.DataFrame({"Time": regular_times}),
            df,
            on="Time",
            direction="backward"
        )

        interp_states = df_interp["state"].values

        # Count transitions
        for i in range(len(interp_states) - 1):
            s0, s1 = interp_states[i], interp_states[i + 1]

            state_time_totals[s0] += dt_hours
            transitions_count[(s0, s1)] += 1

            if s0 != s1:
                transitions_detail.append({
                    "Location": lynx_id,
                    "From": s0,
                    "To": s1,
                    "Time": df_interp["Time"].iloc[i + 1]
                })

        if len(interp_states) > 0:
            state_time_totals[interp_states[-1]] += dt_hours

    # Compute probabilities per timestep
    transition_probs = {}
    for (from_state, to_state), count in transitions_count.items():
        total_time = state_time_totals[from_state]
        num_steps = total_time / dt_hours
        transition_probs[(from_state, to_state)] = count / num_steps

    return transition_probs, state_time_totals, transitions_detail
##################  TRANSITION RATES END #############################







####################### CONVEX HULL FOR STATE 1 START #####################
def compute_convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume / 1e6  # convert to km^2
    except:
        return 0.0
    

def extract_state1_hull_areas(df):
    """
    Extract convex hull areas for each contiguous state 1 segment.
    (State 3 (Return Loop) is treated as state 1 (Stationary State).)
    """
    areas = []

    for lynx_id, group in df.groupby("ID"):
        group = group.sort_values("Time")

        lat = group["Lat"].values
        lon = group["Long"].values

        states = group["State_Loop_Split"].values
        states = np.where(np.isin(states, [1, 3]), 1, states)

        # project
        coords = np.column_stack([lat, lon])
        xy = helper_functions.project_to_alaska_albers(coords)   # meters in Alaska Albers

        # extract contiguous state-1 segments
        in_state = False
        start_idx = None

        for i, s in enumerate(states):
            if s == 1:
                if not in_state:
                    start_idx = i
                    in_state = True
            else:
                if in_state:
                    segment = xy[start_idx:i]
                    area = compute_convex_hull_area(segment)
                    areas.append(area)
                    in_state = False

        # handle trailing segment
        if in_state:
            segment = xy[start_idx:]
            area = compute_convex_hull_area(segment)
            areas.append(area)

    return np.array(areas)

def plot_hull_area_distribution(areas, out_path):
    areas = areas[areas > 0]  # remove degenerate segments
    
    # full histogram
    plt.figure(figsize=(6, 5))
    counts, bins, _ = plt.hist(areas, bins=40, density=True, alpha=0.6, color=colorscheme[0], edgecolor='k', label='Observed')

    # fit exponential distribution
    loc, scale = expon.fit(areas, floc=0)  
    x = np.linspace(0, areas.max(), 500)
    pdf = expon.pdf(x, loc=loc, scale=scale)
    plt.plot(x, pdf, 'k--', lw=2, label=f'Exponential fit')
    
    plt.xlabel("Convex Hull Area (km²)")
    plt.ylabel("Density")
    plt.title("Stationary State Convex Hull Area Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "state1_hull_area_distribution.png", dpi=300)
    plt.close()
        
    return scale 

####################### CONVEX HULL FOR STATE 1 END #####################






####################### LOOPING DIAGNOSTICS START #########################
def extract_state2_transition_dataset(df):
    """
    Build dataset for logistic regression:
    X = distance to home at moment of leaving state 2
    y = 1 if next state is 3, else 0 (goes to state 1)

    Returns:
        distances (km), labels (0 or 1)
    """
    distances = []
    labels = []

    for lynx_id, group in df.groupby("ID"):
        group = group.sort_values("Time")

        states = group["State_Loop_Split"].values
        coords = np.column_stack([group["Lat"].values, group["Long"].values])
        xy = helper_functions.project_to_alaska_albers(coords)

        # define "home" as last state 1 position before state 2
        last_home = None

        for i in range(len(states)):
            if states[i] == 1:
                last_home = xy[i]

            # detect exit from state 2
            if states[i] == 2 and i < len(states) - 1:
                if states[i + 1] != 2 and last_home is not None:

                    dist = np.linalg.norm(xy[i] - last_home) / 1000  # km

                    next_state = states[i + 1]
                    label = 1 if next_state == 3 else 0

                    distances.append(dist)
                    labels.append(label)

    return np.array(distances), np.array(labels)

def fit_state2_transition_logistic(distances, labels, out_path):
    """
    Fit P(state 3 | distance) using logistic regression.
    """

    X = distances.reshape(-1, 1)
    y = labels

    model = LogisticRegression()
    model.fit(X, y)

    beta0 = model.intercept_[0]
    beta1 = model.coef_[0][0]

    # plot fit
    # bin distances
    bins = np.linspace(0, distances.max(), 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(distances, bins)

    prob_empirical = []
    counts = []
    stderr = []

    for i in range(1, len(bins)):
        mask = digitized == i
        n = np.sum(mask)

        if n > 0:
            p = np.mean(labels[mask])
            prob_empirical.append(p)
            counts.append(n)

            # binomial standard error
            stderr.append(np.sqrt(p * (1 - p) / n))
        else:
            # empty bin
            prob_empirical.append(np.nan)
            counts.append(0)
            stderr.append(np.nan)

    prob_empirical = np.array(prob_empirical)
    stderr = np.array(stderr)
    counts = np.array(counts)

    # logistic fit
    x_plot = np.linspace(0, distances.max(), 200)
    prob_fit = 1 / (1 + np.exp(-(beta0 + beta1 * x_plot)))

    # plot
    plt.figure(figsize=(6, 5))

    # error bars
    plt.errorbar(
        bin_centers,
        prob_empirical,
        yerr=stderr,
        fmt='o',
        capsize=3,
        alpha=0.7,
        label="Empirical (binned)",
        color=colorscheme[2]
    )

    # scale point size by counts
    plt.scatter(
        bin_centers,
        prob_empirical,
        s=20 + 5 * counts,
        alpha=0.6,
        color=colorscheme[2]
    )

    # logistic curve
    plt.plot(x_plot, prob_fit, 'k--', lw=2, label="Logistic fit")

    plt.xlabel("Distance to home (km)")
    plt.ylabel("P(next state = 3)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "state2_transition.png", dpi=300)
    plt.close()

    return beta0, beta1
####################### LOOPING DIAGNOSTICS END #########################





############################ MAIN SIMULATION ############################
if __name__ == "__main__":
    # load the data
    grid_search = False
    df = pd.read_csv(data_path, parse_dates=["Time"])
    param_file = home_dir / "data/processed/movementModel/fitParameters.csv"
    out_path =  home_dir / "outputs/movement_diagnostics/01_distributions"
    out_path.mkdir(parents=True, exist_ok=True)

    # these two have distribution plots so we want to run them anyways just so we can get the right diagnostics
    print("Computing convex hull areas for state 1 segments.")
    # compute convex hull areas for state 1
    areas = extract_state1_hull_areas(df)
    territory_size_distribution = plot_hull_area_distribution(areas, out_path)
    print(f"Diagnostic plot saved to {out_path / "state1_hull_area_distribution.png"} \n")

    print("Fitting probabilistic transition model for state 2")
    distances, labels = extract_state2_transition_dataset(df)
    beta0, beta1 = fit_state2_transition_logistic(distances, labels, out_path)
    print(f"Diagnostic plot saved to {out_path / "state2_transition.png"} \n")


    if param_file.exists():
        print(f"Parameters already fitted. Check {param_file} to find the fit parameters.")
    else:
        print("Computing the switching rates\n")

        all_states = {}
        all_trajs = {}

        for lynx_id, group in df.groupby("ID"):
            group = group.sort_values("Time")
            all_states[lynx_id] = group["State_Loop_Split"].values
            all_trajs[lynx_id] = group[["Time"]]

        transition_probs, state_time_totals, transitions_detail = compute_transition_rates_collapsed(all_states, all_trajs, dt_hours=4)

        # extract both directions
        rate_12 = transition_probs.get((1, 2), 0)
        rate_21 = transition_probs.get((2, 1), 0)
        
        # THIS PART OF THE CODE WAS JUST INCLUDED TO GIVE AN IDEA OF HOW THE GRID SEARCHES WORKED. 
        # HOWEVER, ULTIMATELY THE PARAMETERS CHOSEN WERE THE ONES BELOW THAT WERE FIT USING A COMBINATION
        # OF GRID SEARCHES AND QUALITATIVE ADJUSTMENTS. 
        if grid_search:
            print("Running grid search\n")
            # compute the MSDs from the assigned states
            state_msds = extract_state_msds_from_df(df)
            n_segments = []
            state_ids = []
            for state_val, data in state_msds.items():
                n_segments.append(len(data["msds"]))  # number of segments
                state_ids.append(state_val) # keep track of which state

            # to make the distribution comparable, we want to simulate the same number of segments as we actually have
            segments_df = pd.DataFrame({"state": state_ids, "n_segments": n_segments})
            n_sim_state1 = int(segments_df.loc[segments_df["state"] == 1, "n_segments"].values[0])
            n_sim_state2 = int(segments_df.loc[segments_df["state"] == 2, "n_segments"].values[0])
            n_sim_state3 = int(segments_df.loc[segments_df["state"] == 3, "n_segments"].values[0])
            
            state1_params = grid_search_state1(state_msds[1]["msds"], state_msds[1]["lags"], n_simulations=n_sim_state1)
            state2_params = grid_search_state2(state_msds[2]["msds"], state_msds[2]["lags"], n_simulations=n_sim_state2)
            state3_params = grid_search_state3(state_msds[3]["msds"], state_msds[3]["lags"], n_simulations=n_sim_state3)
        else: 
            print("Using fitted parameters selected through qualitative fitting, with quantitative grid " \
            "searches used to guide and constrain the parameter choices (see 01_fittingParameters for the grid search code). \n")
            # state 1 parameters
            state1_params = {
                "alpha": 0.07,
                "D1_lower": 0.025,
                "D1_higher": 0.5
            }

            # state 2 parameters
            state2_params = {
                "v_lower": 0.01,
                "v_higher": 0.5,
                "D2_lower": 0.025,
                "D2_higher": 0.1,
                "Dtheta_lower": 0.00055,
                "Dtheta_higher": 0.00065
            }

            # state 3 parameters
            state3_params = {
                "alpha": 0.15,
                "D1_lower": 0.025,   
                "D1_higher": 0.5
            }
            
        rows = []

        # add transition rates first
        rows.append({"parameter": "lambda_12", "value": rate_12})
        rows.append({"parameter": "lambda_21", "value": rate_21})

        # add exponential fit parameter for the territory size
        rows.append({"parameter": "territory_size_distribution", "value": territory_size_distribution})

        # add exponential fit parameter for the state 3 transition probabilities
        rows.append({"parameter": "state2_logistic_beta0", "value": beta0})
        rows.append({"parameter": "state2_logistic_beta1", "value": beta1})

        # add state parameters
        for state_name, params in zip(["state1", "state2", "state3"], [state1_params, state2_params, state3_params]):
            for key, val in params.items():
                rows.append({
                    "parameter": f"{state_name}_{key}",
                    "value": val
                })

        # save the parameters
        param_df = pd.DataFrame(rows)
        param_df.to_csv(param_file, index=False)
        print(f"Parameters saved to {param_file}.")




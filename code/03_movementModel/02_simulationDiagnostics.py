import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy.stats import expon

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


################################## SIMULATION CODE START ######################################
# helper functions
def alpha_func(territory_size, alpha_state1, alpha_state3, dist_to_home):
    """
    Distance-dependent attraction strength toward the home location.

    This function interpolates between two behavioral regimes:
    a stationary-state attraction strength (alpha_state1) and a
    return-movement attraction strength (alpha_state3), as a function
    of the distance from the current position to the home location. The 
    transition between regimes is modeled using a sigmoid centered
    at the territory size.

    Returns
        - Distance-dependent attraction strength alpha.
    """
    steepness = 10
    return alpha_state1 + (alpha_state3 - alpha_state1) / (1 + np.exp(-steepness * (dist_to_home - territory_size)))

def prob_switch_to_3(dist, beta0, beta1):
    """
    Probability of switching from exploratory to territorial state
    as a function of distance from the home range.

    This function evaluates the logistic model fit to empirical state
    transition data, where the probability of initiating a new
    territory increases with distance from the current home location.

    Returns
        - Probability of switching to a new territory (state 3).
    """
    return 1 / (1 + np.exp(-(beta0 + beta1 * dist)))

# state simulation functions
def simulate_trajectory(params_state1, params_state2, params_state3,
                        beta0, beta1, territory_scale,
                        lambda_12, lambda_21,
                        n_traj=140, n_points=4000, dt=4.0):
    """
    Simulates individual movement trajectories under a
    two-state stochastic movement model.

    The model combines:
    (i) a stationary state with distance-dependent attraction,
    (ii) an exploratory state with correlated random walks,
    we also have an inferred return/out-of-range state that is triggered by
    exceeding a territory boundary.

    State transitions are stochastic and depend on both fixed rates
    and distance-dependent switching probabilities. Home locations
    may be updated upon return to the stationary state based on a
    logistic decision rule.

    Returns
        - trajectories : Simulated positions with shape (n_traj, n_points+1, 2).
        - all_states : Corresponding discrete state labels for each trajectory.
    """
    trajectories = []
    all_states = []
    territory_sizes = []


    for traj_idx in range(n_traj):
        pos = np.zeros((n_points + 1, 2))
        X_home = np.zeros(2)
        states = np.zeros(n_points + 1, dtype=int)

        state = 1
        states[0] = state

        territory_size = np.sqrt(expon.rvs(scale=territory_scale) / np.pi) # we want the radius so we need to divide by pi and take the sqrt
        D1 = np.random.uniform(params_state1["D1_lower"], params_state1["D1_higher"])
        alpha_state1 = params_state1["alpha"]
        alpha_state3 = params_state3["alpha"]

        for t in range(1, n_points + 1):
            if state == 1:
                displacement = X_home - pos[t-1]
                norm = np.linalg.norm(displacement)
                dist_to_home = norm

                alpha = alpha_func(territory_size, alpha_state1, alpha_state3, dist_to_home)
                drift = alpha * displacement / norm if norm != 0 else np.zeros(2)

                pos[t] = pos[t-1] + drift * dt + np.sqrt(2 * D1 * dt) * np.random.randn(2)

                if np.random.rand() < lambda_12:
                    state = 2
                    theta = np.random.uniform(0, 2*np.pi)
                    v0 = np.random.uniform(params_state2["v_lower"], params_state2["v_higher"])
                    D2 = np.random.uniform(params_state2["D2_lower"], params_state2["D2_higher"])
                    Dtheta = np.random.uniform(params_state2["Dtheta_lower"], params_state2["Dtheta_higher"])

            elif state == 2:
                theta += np.sqrt(2 * Dtheta * dt) * np.random.randn()
                velocity = v0 * np.array([np.cos(theta), np.sin(theta)])

                pos[t] = pos[t-1] + velocity * dt + np.sqrt(2 * D2 * dt) * np.random.randn(2)

                if np.random.rand() < lambda_21:
                    state = 1
                    D1 = np.random.uniform(params_state1["D1_lower"], params_state1["D1_higher"])
                    dist_to_home = np.linalg.norm(pos[t] - X_home)
                    p3 = prob_switch_to_3(dist_to_home, beta0, beta1)

                    if np.random.rand() > p3: # p3 gives us the probability of NOT updating home
                        X_home = pos[t].copy()
                        territory_size = np.sqrt(expon.rvs(scale=territory_scale) / np.pi) # we want the radius so we need to divide by pi and take the sqrt
                        
            if state == 1 and dist_to_home > territory_size:
                states[t] = 3
            else: 
                states[t] = state
            
        trajectories.append(pos)
        all_states.append(states)

    return np.array(trajectories), np.array(all_states)
################################## SIMULATION CODE END ######################################


################################## MSD PLOTS START ######################################
def split_into_state_segments(traj, states):
    """
    Split a trajectory into contiguous segments of constant state.
    Returns a dict: {state: [segments]} Each segment is an array of positions.
    """
    segments = {1: [], 2: [], 3: []}

    start = 0
    current_state = states[0]

    for t in range(1, len(states)):
        if states[t] != current_state:
            segment = traj[start:t]
            if len(segment) > 2:  # require minimum length
                segments[current_state].append(segment)

            start = t
            current_state = states[t]

    # last segment
    segment = traj[start:]
    if len(segment) > 2:
        segments[current_state].append(segment)

    return segments

def compute_segment_msds(segments, max_lag):
    """
    Compute MSD curves for each segment. Returns list of MSD arrays.
    """
    msds = []
    for seg in segments:
        max_valid_lag = min(max_lag, len(seg) - 1)
        if max_valid_lag < 100:
            continue
        msd = helper_functions.compute_msd(seg, max_lag_steps=max_valid_lag // 2)
        msds.append(msd) 
    return msds

def simulate_segmented_msds(params_state1, params_state2, params_state3,
                           beta0, beta1, territory_scale,
                           lambda_12, lambda_21,
                           n_traj=160, n_points=10000, max_lag=1000):
    """
    Computes median squared displacement for each
    behavioral state from simulated movement trajectories.

    We first generate stochastic movement trajectories. 
    Each trajectory is then segmented according to the underlying 
    discrete state sequence. Median squared displacement curves are 
    computed separately for each contiguous state segment and 
    aggregated across trajectories.

    Returns
        - dictionary mapping each state (1, 2, 3) to a list of MSD
        curves computed from all corresponding trajectory segments.
    """
    all_msds = {1: [], 2: [],  3: []}

    trajs, states = simulate_trajectory(
        params_state1, params_state2, params_state3,
        beta0, beta1, territory_scale,
        lambda_12, lambda_21,
        n_traj=n_traj, n_points=n_points
    )

    for traj, state_seq in zip(trajs, states):
        segs = split_into_state_segments(traj, state_seq)

        for s in [1, 2, 3]:
            msds = compute_segment_msds(segs[s], max_lag)
            all_msds[s].extend(msds)

    return all_msds


def plot_segmented_msds(all_msds, out_path, dt=4.0):
    """Plots the calculated MSD segments split by state."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    color = [colorscheme[2], colorscheme[4], colorscheme[6]]
    names = ["Stationary", "Exploratory", "Return Loop"]
    for idx, s in enumerate([1, 2, 3]):
        ax = axs[idx]

        for msd in all_msds[s]:
            lags = np.arange(1, len(msd) + 1) * dt
            msd = np.maximum(msd, 1e-6)
            ax.plot(lags, msd, alpha=0.2, color = color[s - 1])

        ax.set_title(f"{names[s - 1]} State (Simulated)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Lag")
        ax.set_ylim(1e-3, 1e7)
        ax.grid(True)

    axs[0].set_ylabel("MSD")

    plt.tight_layout()
    plt.savefig(out_path / "MSDs.png", dpi=300)
################################## MSD PLOTS END ######################################



######################## VELOCITY TURNING ANGLE PLOTS START ######################################
def simulate_velocity_turn_data(params_state1, params_state2, params_state3,
                               beta0, beta1, territory_scale,
                               lambda_12, lambda_21,
                               n_traj=200, n_points=10000, dt=4.0):
    """
    Computes velocity and turning angle distributions for each behavioral state. 

    We first generate stochastic movement trajectories. 
    Each trajectory is then segmented according to the underlying 
    discrete state sequence. Then, velocity and turning angles are computed 
    separately for each contiguous state segment and 
    aggregated across trajectories.

    Returns
        - dictionary mapping each state (1, 2, 3) to a list of velocities
        and turning angles computed from all corresponding trajectory segments.
    """
    vel_turn_data = {
        1: {"velocities": [], "turning_angles": []},
        2: {"velocities": [], "turning_angles": []},
        3: {"velocities": [], "turning_angles": []},
    }

    # simulate trajectories
    trajs, states = simulate_trajectory(
        params_state1, params_state2, params_state3,
        beta0, beta1, territory_scale,
        lambda_12, lambda_21,
        n_traj=n_traj,
        n_points=n_points,
        dt=dt
    )

    # split into segments + compute metrics
    for traj, state_seq in zip(trajs, states):

        segments = split_into_state_segments(traj, state_seq)

        for s in [1, 2, 3]:
            for seg in segments[s]:
                if len(seg) < 3:
                    continue

                velocities, angles = compute_velocity_and_turning_angles(seg, dt)

                vel_turn_data[s]["velocities"].extend(velocities)
                vel_turn_data[s]["turning_angles"].extend(angles)

    # convert to arrays
    for s in [1, 2, 3]:
        vel_turn_data[s]["velocities"] = np.array(vel_turn_data[s]["velocities"])
        vel_turn_data[s]["turning_angles"] = np.array(vel_turn_data[s]["turning_angles"])

    return vel_turn_data

def compute_velocity_and_turning_angles(traj, dt):
    """
    Since we're no longer working on a sphere, we can just compute them directly 
    as opposed to our previous calculations in 02_stateClassification/03b_loopDiagnostics.py
    """
    displacements = np.diff(traj, axis=0)

    # compute velocities
    velocities = np.linalg.norm(displacements, axis=1) / dt  # normalize it so units match up with our data

    # compute turning angles
    v1 = displacements[:-1]
    v2 = displacements[1:]

    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)

    cos_angle = dot / (norm1 * norm2 + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)

    angles = np.arccos(cos_angle)

    return velocities[1:], angles  # align lengths


def plot_velocity_turn_heatmaps(vel_turn_data, out_path, n_angle_bins=36, n_vel_bins=40, max_vel=8):
    """
    Plotting the velocity and turning angle distributions
    """
    state_info = [(1,'Stationary'), (2,'Exploratory'), (3,'Return Loop')]

    fig, axs = plt.subplots(1,3, figsize=(18,5), sharey=True)

    cmap_state1 = LinearSegmentedColormap.from_list("state1_cmap", ["white", colorscheme[2]])
    cmap_state2 = LinearSegmentedColormap.from_list("state2_cmap", ["white", colorscheme[4]])
    cmap_state3 = LinearSegmentedColormap.from_list("state3_cmap", ["white", colorscheme[6]])

    cmaps = [cmap_state1, cmap_state2, cmap_state3]

    for ax, (s, title), cmap in zip(axs, state_info, cmaps):

        angles = vel_turn_data[s]["turning_angles"]
        velocities = vel_turn_data[s]["velocities"]

        mask = velocities < max_vel
        angles = angles[mask]
        velocities = velocities[mask]

        H, _, _ = np.histogram2d(
            angles, velocities,
            bins=[n_angle_bins, n_vel_bins],
            range=[[0, np.pi], [0, max_vel]],
            density=True
        )

        H = H.T
        H = np.clip(H, 1e-10, None)

        im = ax.imshow(
            H,
            origin='lower',
            aspect='auto',
            extent=[0, np.pi, 0, max_vel],
            cmap=cmap,
            norm=mcolors.LogNorm(vmin=1e-10, vmax=H.max())
        )

        ax.set_title(title)
        ax.set_xlabel("Turning angle (rad)")
        ax.grid(False)

        fig.colorbar(im, ax=ax).set_label("Density (log scale)")

    axs[0].set_ylabel("Velocity (km/h)")
    plt.tight_layout()
    plt.savefig(out_path / "velocity_turning_angle_dist.png", dpi=300)
######################## VELOCITY TURNING ANGLE PLOTS END ######################################


########################### TRAJECTORY PLOT START ###########################################
def plot_full_trajectory(pos, states, run, out_path):
    plt.figure(figsize=(6, 6))

    for s, c in zip([1, 2, 3],
                    [colorscheme[2], colorscheme[4], colorscheme[6]]):
        mask = states == s
        plt.plot(pos[mask, 0], pos[mask, 1], '.', alpha=0.4, color=c, label=f"State {s}")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Unbounded Trajectory")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path / f"trajectory_unbounded_{run}.png", dpi=300)
    plt.close()
########################### TRAJECTORY PLOT END ###########################################


if __name__ == "__main__":
    param_file = home_dir / "data/processed/movementModel/fitParameters.csv"
    params_df = pd.read_csv(param_file)

    if param_file.exists():
        print("Loading existing parameters")

        # extract state1 parameters 
        params_state1 = params_df[params_df['parameter'].str.startswith("state1_")]
        params_state1 = {row['parameter'].replace("state1_", ""): row['value'] for _, row in params_state1.iterrows()}

        # extract state2 parameters
        params_state2 = params_df[params_df['parameter'].str.startswith("state2_")]
        params_state2 = {row['parameter'].replace("state2_", ""): row['value'] for _, row in params_state2.iterrows()}

        # extract state3 parameters
        params_state3 = params_df[params_df['parameter'].str.startswith("state3_")]
        params_state3 = {row['parameter'].replace("state3_", ""): row['value'] for _, row in params_state3.iterrows()}

        # extract loop distance scale (exponential fit)
        beta0 = float(params_df.loc[params_df['parameter'] == "state2_logistic_beta0", 'value'].values[0])
        beta1 = float(params_df.loc[params_df['parameter'] == "state2_logistic_beta1", 'value'].values[0])

        territory_size_row = params_df[params_df['parameter'].str.contains("territory_size_distribution")]
        territory_scale = float(territory_size_row['value'].values[0])

        # extract switching rates
        lambda_12 = float(params_df[params_df['parameter'] == "lambda_12"]['value'].values[0])
        lambda_21 = float(params_df[params_df['parameter'] == "lambda_21"]['value'].values[0])


        out_path = home_dir / "outputs" / "movement_diagnostics/02_diagnostics"
        out_path.mkdir(parents=True, exist_ok=True)

        print("\nPlotting MSD split by state")
        ### Plotting the MSD distributions ####
        all_msds = simulate_segmented_msds(
            params_state1=params_state1,
            params_state2=params_state2,
            params_state3=params_state3,
            beta0=beta0,
            beta1=beta1,
            territory_scale=territory_scale,
            lambda_12=lambda_12,
            lambda_21=lambda_21,
            n_traj=160,     
            n_points=10000,    
            max_lag=10000
        )

        plot_segmented_msds(all_msds, out_path=out_path, dt=4.0)
        print("Finished plotting MSD split by state\n")

        print("Plotting turning angle and velocity distributions")
        ### Plotting the velocity and turning angle distributions ###
        vel_turn_data = simulate_velocity_turn_data(
            params_state1=params_state1,
            params_state2=params_state2,
            params_state3=params_state3,
            beta0=beta0,
            beta1=beta1,
            territory_scale=territory_scale,
            lambda_12=lambda_12,
            lambda_21=lambda_21,
            n_traj=200,
            n_points=10000,
            dt=4.0
        )

        plot_velocity_turn_heatmaps(vel_turn_data, out_path=out_path)
        print("Finished plotting turning angle and velocity distributions \n")


        print("Plotting example trajectories")
        ### Plotting three random example trajectories ###
        n_traj = 3
        pos, states = simulate_trajectory(
            params_state1=params_state1,
            params_state2=params_state2,
            params_state3=params_state3,
            beta0=beta0,
            beta1=beta1,
            territory_scale=territory_scale,
            lambda_12=lambda_12,
            lambda_21=lambda_21,
            n_traj=n_traj,
            n_points=5000
        )

        for i in range(n_traj):
            plot_full_trajectory(pos[i], states[i], i, out_path)  
        print("Finished plotting example trajectories\n")

        print(f"See completed diagnostic plots at: {out_path}")


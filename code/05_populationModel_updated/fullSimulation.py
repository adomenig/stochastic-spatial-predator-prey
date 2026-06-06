import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
import os
import time
import h5py
import argparse
from scipy.stats import expon
import sys

# Command-line arguments
parser = argparse.ArgumentParser(description="Run lynx-hare simulations with different initial conditions")
parser.add_argument("--initial", type=str, choices=["EW", "uniform", "gaussian"], default="uniform",
                    help="Type of initial conditions for lynx positions")
parser.add_argument("--parameters", type=str, required=True, help="File path to fitParameters.csv")

args = parser.parse_args()
initial_type = args.initial
parameter_path = Path(args.parameters)

print(f"Using initial conditions: {initial_type}")

########################## PARAMETER INITIALIZATION START #######################
# INITIALIZING PARAMETERS
L_w, L_h = 1000, 1000 # height and width of our field. This is fit so 1 unit = 1 km. 
T_MAX = 200 # maximum time of simulation. This is fit so 1 time unit = 1 year
DT = 4.0 / (24*365.25) # our time step is 4 hours. 
STEPS = int(T_MAX / DT) # the number of steps we need to take calculated by dividing the nr of years we have by our 4-hour time step.

# Hare Parameters
beta0 = 2 # Hare birth rate (they will give between 1-3 times a year on average)
sigma = 1/6 # Hare death rate  (they will die from natural causes (not predation-related) every 3 years on average)
D_B = 5 # diffusion rate (they will go to one of their neighboring fields around 50 times a year on average) 

# Lynx parameters
alpha0 = 1.0 # Lynx birth rate (lynx will give birth once a year on average)
delta = 1/15 # Lynx death rate (lynx will die every 15 years from natural causes on average if hares are abundant)
mu = 365.25 / 2.5 # Predation rate (with abundant hares available, they will on average eat a hare every 2-3 days)

# Initial concentrations of lynx and hare. These values will stabilize after the first 1-2 cycles.
initial_hare = 1
initial_lynx = 0.05

params_state1 = {}
params_state2 = {}
params_state3 = {}
lambda_21 = 0
territory_scale = 0
beta0_logistic = 0
beta1_logistic = 0

wipeout_done = False 
wipeout_timing = 100

if parameter_path.exists():
    params_df = pd.read_csv(parameter_path)
    # extract state1 parameters 
    state1_params = params_df[params_df['parameter'].str.startswith("state1_")]
    state1_params = {row['parameter'].replace("state1_", ""): row['value'] for _, row in state1_params.iterrows()}

    # extract state2 parameters
    state2_params = params_df[params_df['parameter'].str.startswith("state2_")]
    state2_params = {row['parameter'].replace("state2_", ""): row['value'] for _, row in state2_params.iterrows()}

    # extract state3 parameters
    state3_params = params_df[params_df['parameter'].str.startswith("state3_")]
    state3_params = {row['parameter'].replace("state3_", ""): row['value'] for _, row in state3_params.iterrows()}

    # extract switching rates
    #lambda_12 = float(params_df[params_df['parameter'] == "lambda_12"]['value'].values[0])
    lambda_21 = float(params_df[params_df['parameter'] == "lambda_21"]['value'].values[0])


    territory_size_row = params_df[params_df['parameter'].str.contains("territory_size_distribution")]
    territory_scale = float(territory_size_row['value'].values[0])

    # extract loop distance scale (exponential fit)
    beta0_logistic = float(params_df.loc[params_df['parameter'] == "state2_logistic_beta0", 'value'].values[0])
    beta1_logistic = float(params_df.loc[params_df['parameter'] == "state2_logistic_beta1", 'value'].values[0])
else: 
    print(f"File with saved parameters missing ({parameter_path} not found). Make sure your input is corect.")
    sys.exit(1)

STATE1, STATE2 = 1, 2
# lambda_12 values we are testing
lambda_12_values = [0.0001, 0.00025, 0.0005, 0.001, 0.0015]
n_runs = 9  # number of repeats per scenario
snapshot_interval_steps = 546 # roughly every 3 months
########################## PARAMETER INITIALIZATION END #######################






############################ HELPER FUNCTIONS FOR SIMULATION START ##################
# RATE FUNCTIONS
def delta_of_B(B_local):
    """
    Predator death rate as a function of local prey density. This is where the main 
    dependance in predator-prey dynamics comes from. This is a hill function with the 
    idea that low prey densities means higher predator death rates. 
    """
    del_max = 1
    B_half = 5
    h = 5
    return delta + (del_max - delta) / (1 + (B_local / B_half) ** h)

# Number of offspring lynx
def k_of_B(B_local):
    """
    Number of offspring per lynx as a function of local prey density. This is a 
    sigmoid step function the n_births approaches 0 at low prey and k_max at high prey. 
    """
    k_min, k_max = 0, 8
    B_half = 5
    h = 1 # steepness
    sigmoid = (k_max) / (1 + np.exp(-h * (B_local - B_half)))
    return np.clip(sigmoid, k_min, k_max).astype(int)


def alpha_func(territory_size, alpha_state1, alpha_state3, dist_to_home):
    """
    Distance-dependent attraction strength toward the home location.

    This function interpolates between two behavioral regimes:
    a stationary-state attraction strength (alpha_state1) and a
    return-movement attraction strength (alpha_state3), as a function
    of the distance from the current position to the home location. The 
    transition between regimes is modeled using a sigmoid centered
    at the territory size. We clip to avoid overflow errors but it's functionally
    the same. 

    Returns
        - Distance-dependent attraction strength alpha.
    """
    steepness = 10
    x = steepness * (dist_to_home - territory_size)
    x = np.clip(x, -50, 50)
    sigmoid = 1 / (1 + np.exp(-x))
    return alpha_state1 + (alpha_state3 - alpha_state1) * sigmoid

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

############################ HELPER FUNCTIONS FOR SIMULATION END ##################







####################### MOVEMENT SIMULATION START #################################

def periodic_displacement(a, b, L_w, L_h):
    """
    Compute shortest wrapped displacement vector a - b
    on a periodic domain.
    """
    diff = a - b

    diff[:, 0] = (diff[:, 0] + L_w / 2) % L_w - L_w / 2
    diff[:, 1] = (diff[:, 1] + L_h / 2) % L_h - L_h / 2

    return diff


def periodic_distance(a, b, L_w, L_h):
    """
    Compute shortest wrapped Euclidean distance
    between two coordinate arrays.
    """
    diff = periodic_displacement(a, b, L_w, L_h)
    return np.linalg.norm(diff, axis=1)

# LYNX MOVEMENT
def move_lynx(pos, state, params, B_density, lambda_12_val, dt=4.0):

    N = len(pos)

    if N == 0:
        return pos, state, params

    ############################################################
    # STATE 1: HOME-RANGE MOVEMENT
    ############################################################

    idx1 = state == STATE1
    n1 = idx1.sum()

    if n1 > 0:

        pos1 = pos[idx1]
        home1 = params["home"][idx1]

        # --- PERIODIC DISPLACEMENT TO HOME ---
        diff = periodic_displacement(pos1, home1, L_w, L_h)

        r = np.linalg.norm(diff, axis=1, keepdims=True)
        r[r == 0] = 1.0

        dist_to_home = r.flatten()

        # distance-dependent attraction strength
        alpha = alpha_func(
            params["territory"][idx1],
            state1_params["alpha"],
            state3_params["alpha"],
            dist_to_home
        )

        alpha = alpha[:, None]

        # deterministic pull toward home
        pos1 -= alpha * diff / r * dt

        # diffusion
        pos1 += np.sqrt(
            2 * params["D1"][idx1][:, None] * dt
        ) * np.random.randn(n1, 2)

        # switching to exploratory state
        switch_mask = np.random.rand(n1) < lambda_12_val

        idx_switch = np.where(idx1)[0][switch_mask]

        if len(idx_switch) > 0:

            state[idx_switch] = STATE2

            params["v"][idx_switch] = np.random.uniform(
                state2_params["v_lower"],
                state2_params["v_higher"],
                size=len(idx_switch)
            )

            params["D2"][idx_switch] = np.random.uniform(
                state2_params["D2_lower"],
                state2_params["D2_higher"],
                size=len(idx_switch)
            )

            params["Dtheta"][idx_switch] = np.random.uniform(
                state2_params["Dtheta_lower"],
                state2_params["Dtheta_higher"],
                size=len(idx_switch)
            )

            params["theta"][idx_switch] = np.random.uniform(
                0, 2*np.pi,
                size=len(idx_switch)
            )

        pos[idx1] = pos1

    ############################################################
    # STATE 2: EXPLORATORY MOVEMENT
    ############################################################

    idx2 = state == STATE2
    n2 = idx2.sum()

    if n2 > 0:

        theta2 = params["theta"][idx2]
        v2 = params["v"][idx2]
        D2 = params["D2"][idx2]
        Dtheta = params["Dtheta"][idx2]

        # angular diffusion
        theta2 += np.sqrt(2 * Dtheta * dt) * np.random.randn(n2)

        # persistent motion
        dx = (
            v2[:, None]
            * np.column_stack([
                np.cos(theta2),
                np.sin(theta2)
            ])
        ) * dt

        # translational diffusion
        dx += np.sqrt(
            2 * D2[:, None] * dt
        ) * np.random.randn(n2, 2)

        pos2 = pos[idx2] + dx

        # switching back to territorial state
        switch_mask = np.random.rand(n2) < lambda_21

        idx_switch = np.where(idx2)[0][switch_mask]

        if len(idx_switch) > 0:

            state[idx_switch] = STATE1

            params["D1"][idx_switch] = np.random.uniform(
                state1_params["D1_lower"],
                state1_params["D1_higher"],
                size=len(idx_switch)
            )

            ####################################################
            # PERIODIC DISTANCE TO HOME
            ####################################################

            dist_to_home = periodic_distance(
                pos[idx_switch],
                params["home"][idx_switch],
                L_w,
                L_h
            )

            p3 = prob_switch_to_3(
                dist_to_home,
                beta0_logistic,
                beta1_logistic
            )

            update_mask = np.random.rand(len(idx_switch)) > p3

            idx_update = idx_switch[update_mask]

            if len(idx_update) > 0:

                # establish new home range
                params["home"][idx_update] = pos[idx_update].copy()

                params["territory"][idx_update] = np.sqrt(
                    expon.rvs(
                        scale=territory_scale,
                        size=len(idx_update)
                    ) / np.pi
                )

        params["theta"][idx2] = theta2
        pos[idx2] = pos2

    ############################################################
    # PERIODIC WRAPPING
    ############################################################

    pos[:, 0] = pos[:, 0] % L_w
    pos[:, 1] = pos[:, 1] % L_h

    return pos, state, params
####################### MOVEMENT SIMULATION END #################################








####################### SIMULATION REACTIONS START #################################
# REACTION 1: HARE BIRTH B -> B + 0-4
#K_hare = 50  # carrying capacity per cell

def do_hare_birth(B):
    #local_rate = beta0 * B * (1 - B / K_hare)
    #local_rate = np.maximum(local_rate, 0)
    #birth_counts = np.random.poisson(local_rate * DT)
    birth_counts = np.random.poisson(beta0 * B * DT)
    total_births = birth_counts.sum()

    if total_births > 0:
        parent_i, parent_j = np.nonzero(birth_counts)
        repeats = birth_counts[parent_i, parent_j]

        parents_i = np.repeat(parent_i, repeats)
        parents_j = np.repeat(parent_j, repeats)

        # litter size 1–4 per birth event
        litter_sizes = np.random.randint(1, 5, size=len(parents_i))

        # expand parents according to litter size
        parents_i = np.repeat(parents_i, litter_sizes)
        parents_j = np.repeat(parents_j, litter_sizes)

        directions = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        choices = np.random.randint(0, 4, size=len(parents_i))
        di_dj = directions[choices]

        offspring_i = (parents_i + di_dj[:,0]) % L_h
        offspring_j = (parents_j + di_dj[:,1]) % L_w

        np.add.at(B, (offspring_i, offspring_j), 1)

    return B


# REACTION 2: HARE DEATH B -> NULL
def do_hare_death(B):
    deaths = np.random.poisson(sigma * B * DT)
    B -= np.minimum(deaths, B)
    return B

# REACTION 3: HARE DIFFUSION WITH PERIODIC BOUNDARIES
def do_hare_diffusion(B):
    if D_B <= 0 or B.sum() == 0:
        return B

    move_prob = 1 - np.exp(-D_B * DT)

    hare_coords = np.argwhere(B > 0)
    if len(hare_coords) == 0:
        return B

    counts = B[hare_coords[:, 0], hare_coords[:, 1]]

    # number of movers from each cell
    movers_per_cell = np.random.binomial(counts, move_prob)
    total_movers = movers_per_cell.sum()

    if total_movers == 0:
        return B

    # expand movers into individual particles
    mover_i = np.repeat(hare_coords[:, 0], movers_per_cell)
    mover_j = np.repeat(hare_coords[:, 1], movers_per_cell)

    # directions: down, up, right, left
    directions = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])

    # choose random direction for each mover
    choices = np.random.randint(0, 4, size=total_movers)
    moves = directions[choices]

    # periodic wrapping
    new_i = (mover_i + moves[:, 0]) % L_h
    new_j = (mover_j + moves[:, 1]) % L_w

    # update grid
    B_new = B.copy()

    # remove movers from original positions
    np.add.at(B_new, (mover_i, mover_j), -1)

    # add movers to wrapped positions
    np.add.at(B_new, (new_i, new_j), 1)

    return B_new

# REACTION 4: LYNX DEATH A -> NULL
def do_lynx_death(pos, state, params, B_density):
    if len(pos) == 0:
        return pos, state, params

    i = pos[:, 1].astype(int)
    j = pos[:, 0].astype(int)

    death_rate = delta_of_B(B_density[i, j])  # lynx death rate depends on local prey
    death_events = np.random.poisson(death_rate * DT)

    alive_mask = death_events == 0
    pos = pos[alive_mask]
    state = state[alive_mask]
    for key in params:
        params[key] = params[key][alive_mask]

    return pos, state, params

# REACTION 5: LYNX BIRTH A -> A + A
def do_lynx_birth(pos, state, params, B_density):
    if len(pos) == 0:
        return pos, state, params

    # only lynx in state 1 can give birth
    #state1_mask = state == STATE1
    #birth_events = np.zeros(len(pos), dtype=int)
    #birth_events[state1_mask] = np.random.poisson(alpha0 * DT, size=state1_mask.sum())

    #parents_mask = birth_events > 0
    birth_events = np.random.poisson(alpha0 * DT, size=len(pos))

    parents_mask = birth_events > 0


    if not np.any(parents_mask):
        return pos, state, params

    parents = pos[parents_mask]
    i = parents[:, 1].astype(int)
    j = parents[:, 0].astype(int)
    B_local = B_density[i, j]

    offspring_counts = k_of_B(B_local)  # number of offspring per parent
    offspring_counts *= birth_events[parents_mask]  # account for multiple birth events

    total_offspring = offspring_counts.sum()
    if total_offspring == 0:
        return pos, state, params

    # Offspring appear at parent's location
    new_pos = np.repeat(parents, offspring_counts, axis=0)
    new_pos[:, 0] = np.clip(new_pos[:, 0], 0, L_w - 1)
    new_pos[:, 1] = np.clip(new_pos[:, 1], 0, L_h - 1)

    # Update positions and states
    pos = np.vstack([pos, new_pos])
    state = np.concatenate([state, np.ones(len(new_pos), dtype=int)])
    params["home"] = np.vstack([params["home"], new_pos])
    params["D1"] = np.concatenate([params["D1"], np.random.uniform(state1_params["D1_lower"], state1_params["D1_higher"], len(new_pos))])
    new_territories = np.sqrt(expon.rvs(scale=territory_scale, size=len(new_pos)) / np.pi)
    params["territory"] = np.concatenate([params["territory"], new_territories])
    for key in ["v", "D2", "theta", "Dtheta"]:
        params[key] = np.concatenate([params[key], np.zeros(len(new_pos))])

    return pos, state, params

# REACTION 6: PREDATION A + B -> A
def do_predation(B, pos, mu):
    if len(pos) == 0 or B.sum() == 0:
        return B

    i = pos[:, 1].astype(int)
    j = pos[:, 0].astype(int)

    B_local = B[i, j]
    expected_pred = mu * B_local * DT
    num_eaten = np.random.poisson(expected_pred)

    total_eaten = np.zeros_like(B)
    np.add.at(total_eaten, (i, j), num_eaten)
    total_eaten = np.minimum(total_eaten, B)

    B -= total_eaten
    return B

# REACTION 7: LYNX MOVEMENT A -> MOVE USING STOCHASTIC MOVEMENT MODEL
def lynx_movement_reaction(pos, state, params, B_density, lambda_12):
    pos, state, params = move_lynx(pos, state, params, B_density, lambda_12)
    return pos, state, params
####################### SIMULATION REACTIONS END #################################




###################################### MAIN SIMULATION START ####################################################
# SIMULATION
def simulate(output_dir, lambda_12_val, run):

    B = np.full((L_h, L_w), initial_hare, dtype=np.int32)

    N = int(L_h * L_w * initial_lynx)

    pos = np.zeros((N, 2))
    state = np.ones(N, dtype=int)

    if initial_type == "uniform":
        pos[:, 0] = np.random.uniform(0, L_w-1, N)
        pos[:, 1] = np.random.uniform(0, L_h-1, N)

    elif initial_type == "EW":
        pos[:, 0] = np.random.uniform(L_w*0.8, L_w-1, N)
        pos[:, 1] = np.random.uniform(0, L_h-1, N)

    elif initial_type == "gaussian":

        n_blobs = 3
        blob_size = N // n_blobs
        std_dev = min(L_w, L_h) * 0.05
        margin = 3 * std_dev

        for i in range(n_blobs):

            center_x = np.random.uniform(margin, L_w - margin)
            center_y = np.random.uniform(margin, L_h - margin)

            start = i * blob_size
            end = (i + 1) * blob_size if i < n_blobs - 1 else N

            pos[start:end, 0] = np.random.normal(center_x, std_dev, end-start)
            pos[start:end, 1] = np.random.normal(center_y, std_dev, end-start)

    params = {
        "home": pos.copy(),
        "D1": np.random.uniform(state1_params["D1_lower"], state1_params["D1_higher"], N),
        "v": np.zeros(N),
        "D2": np.zeros(N),
        "theta": np.zeros(N),
        "Dtheta": np.zeros(N),
        "territory": np.sqrt(expon.rvs(scale=territory_scale, size=N) / np.pi)
    }

    history = []

    t = 0.0

    output_dir.mkdir(exist_ok=True)


    snapshot_interval_steps = 182# once a month
    h5_path = output_dir / f"snapshots_{run}.h5"

    h5file = h5py.File(h5_path, "w")

    # variable-length arrays for lynx coordinates
    vlen_float = h5py.vlen_dtype(np.float32)

    num_snapshots = (STEPS // snapshot_interval_steps) + 1

    # full hare field
    dset_B = h5file.create_dataset(
        "B",
        shape=(num_snapshots, L_h, L_w),
        dtype="uint32",
        compression="gzip",
        chunks=(1, L_h, L_w)
    )

    # simulation time
    dset_time = h5file.create_dataset(
        "time",
        shape=(num_snapshots,),
        dtype="float32"
    )

    # STATE1 lynx positions
    dset_state1_pos = h5file.create_dataset(
        "state1_pos",
        shape=(num_snapshots,),
        dtype=vlen_float
    )

    # STATE2 lynx positions
    dset_state2_pos = h5file.create_dataset(
        "state2_pos",
        shape=(num_snapshots,),
        dtype=vlen_float
    )

    snapshot_idx = 0
    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------

    wipeout_done = False  # flag to ensure the event fires exactly once

    for step in range(STEPS):

        # WIPEOUT EVENT: clear all hares in [300:700, 300:700] at t = 50
        if not wipeout_done and t >= wipeout_timing:
            B[250:751, 250:751] = 0
            print(f"[t={t:.2f}] Wipeout event: hares cleared in region x=[250,750], y=[250,750]")
            wipeout_done = True

        reactions = [
            lambda: do_hare_death(B),
            lambda: do_hare_birth(B),
            lambda: do_hare_diffusion(B),
            lambda: do_predation(B, pos, mu),
            lambda: do_lynx_death(pos, state, params, B),
            lambda: do_lynx_birth(pos, state, params, B),
            lambda: lynx_movement_reaction(pos, state, params, B, lambda_12_val)
        ]

        np.random.shuffle(reactions)

        for reaction in reactions:

            result = reaction()

            if isinstance(result, tuple):
                pos, state, params = result
            else:
                B = result

        # extinction checks
        if B.sum() == 0:
            print(f"Extinction: hares died out at step {step}, time {t:.2f}")
            break

        if len(pos) == 0:
            print(f"Extinction: lynx died out at step {step}, time {t:.2f}")
            break

        # update time
        t += DT

        n_state1 = np.sum(state == 1)
        n_state2 = np.sum(state == 2)

                # SAVE SNAPSHOT EVERY WEEK (42 STEPS)
        if step % snapshot_interval_steps == 0:

            print(f"Saving snapshot {snapshot_idx} at step {step}")

            # save hare field
            dset_B[snapshot_idx] = B.astype(np.uint32)

            # save time
            dset_time[snapshot_idx] = t

            # separate lynx by state
            state1_mask = state == STATE1
            state2_mask = state == STATE2

            state1_pos = pos[state1_mask].astype(np.float32)
            state2_pos = pos[state2_mask].astype(np.float32)

            # flatten because hdf5 vlen datasets are 1D
            dset_state1_pos[snapshot_idx] = state1_pos.ravel()
            dset_state2_pos[snapshot_idx] = state2_pos.ravel()

            snapshot_idx += 1

        history.append((t, B.sum(), n_state1, n_state2))

    h5file.close()
    return (np.array(history), B, pos, state)

def run_simulation(scenario_idx, run_idx, lambda_12_val):
    try:
        seed = int(time.time() * 1e6) % (2**32) + run_idx + scenario_idx * 1000
        np.random.seed(seed)

        print(f"Starting scenario {scenario_idx}, run {run_idx}, lambda_12={lambda_12_val}")

        output_dir = Path(f"/project/jnirody/Alissa/Results/fullSimulation_replicates/lambda12_{lambda_12_val}/")
        output_dir.mkdir(parents=True, exist_ok=True)

        history, B_final, lynx_final, state = simulate(output_dir, lambda_12_val, run_idx)

        job_id = os.environ.get("SLURM_JOB_ID", "local")

        np.savez_compressed(
            output_dir / f"scenario{scenario_idx}_run{run_idx}.npz",
            history=np.array(history),
            B_final=B_final,
            lynx_pos=lynx_final,
            lynx_state=state
        )

        print(f"Finished scenario {scenario_idx}, run {run_idx}")

    except Exception as e:
        print(f"ERROR in scenario {scenario_idx}, run {run_idx}: {e}")
###################################### MAIN SIMULATION END ####################################################





if __name__ == "__main__":

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    # n_repeats per run
    n_reps = 10

    #### 0.00025 = roughly every 2 years it goes on an excursion
    ### 0.00005 = roughly every 10 years it goes on an excursion
    ### 0.0001 = roughly every 5 years it goes on an excursion
    ### 0.0005 = roughly every year it goes on an excursion
    ### 0.001 = roughly every year it goes on 2 excursions 
    ### 0.005 = roughly every year it goes on 10 excursions

    lambda_12_values = [0, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.005]

    n_scenarios = len(lambda_12_values)

    total_jobs = n_scenarios * n_reps

    if task_id >= total_jobs:
        raise ValueError(f"Task ID {task_id} exceeds total jobs {total_jobs}")

    # scenario index: 0?~@~S5
    scenario_idx = task_id // n_reps

    # replicate index: 0?~@~S9
    run_idx = task_id % n_reps

    lambda_12_val = lambda_12_values[scenario_idx]

    print(f"Task {task_id}: "f"scenario {scenario_idx}, " f"replicate {run_idx}, " f"lambda_12={lambda_12_val}")

    run_simulation(scenario_idx, run_idx, lambda_12_val)



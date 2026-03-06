import numpy as np
from pathlib import Path
import multiprocessing as mp
import os
import time
import h5py

# PARAMETERS
L_w, L_h = 1000, 1000
T_MAX = 50
DT = 4.0 / (24*365.25)
STEPS = int(T_MAX / DT)

# Hare Parameters
beta0 = 2.0 # Hare birth rate (on average, they give birth between 1-3 times a year)
sigma = 1/3 # Hare death rate 
D_B = 50 # diffusion rate 

# Lynx parameters
alpha0 = 1.0 # Lynx birth rate (lynx give birth once a year)
lambda0 = 1/15 # Lynx death rate (their lifespan is generally 15 years if they don't die earlier)
mu = 365.25 / 2.5 # Predation rate (with abundant hares available, they generally eat a hare every 2-3 days)


# Initial concentrations of lynx and hare. These values stabilize after the first 1-2 cycles
initial_hare = 10
initial_lynx = 0.01

STATE1, STATE2 = 1, 2

# Movement parameters, fit to 4-hour itnervals
state1_params = {'alpha': 0.07, 'D1_0': 0.01, 'D1_final': 0.5,}
state2_params = {'v_lower': 0.01, 'v_higher': 3.0, 'D2_lower': 0.05, 'D2_higher': 0.25, 'Dtheta_lower': 0.0055, 'Dtheta_higher': 0.0065,}
# state switching parameters, also fit at 4 hours
STATE1, STATE2 = 1, 2
lambda_12 = 0.00034 ## FIT AT 4 HOURS
lambda_21 = 0.0021 ## FIT AT 4 HOURS


# =========================
# RATE FUNCTIONS
# =========================
def lambda_of_B(B):
    """
    Predator death rate as a function of local prey density. This is where the main 
    dependance in predator-prey dynamics comes from. This is a hill function with the 
    idea that low prey densities means higher predator death rates. 
    """
    lam_max = 0.5
    B_half = initial_hare * 5
    h = 2
    return lambda0 + (lam_max - lambda0) / (1 + (B / B_half) ** h)

def lambda_12_of_B(B_local):
    """
    Swithing rate from stationary to exploratory state as a function of local prey density. 
    This is a hill function, with the idea that switching rate decreases as prey increases i.e.
    predators are less likely to leave their habitat when they have enough prey in the area. 
    """
    B_half = initial_hare
    h = 2.0
    return (lambda_12) / (1 + (B_local / B_half)**h)


# Number of offspring lynx
def k_of_B(B_local):
    """
    Number of offspring per lynx as a function of local prey density. This is a 
    sigmoid step function the n_births approaches 0 at low prey and k_max at high prey. 
    """
    k_min, k_max = 0, 8
    B_half, h = initial_hare * 5, 2  # midpoint and steepness
    sigmoid = (k_max) / (1 + np.exp(-h * (B_local - B_half)))
    return np.clip(sigmoid, k_min, k_max).astype(int)


# =========================
# LYNX MOVEMENT
# =========================
def move_lynx(pos, state, params, B_density, dt=4.0):
    """
    This function is the core logic of the lynx movement. Unlike the MC simulation, all movement was
    fit at dt=4, so therefore we use 4 to multiply the rates and values in this part. The movement consists of two
    states: Stationary and Exploratory. The stationary movement consists of diffusion with a pull towards home
    whereas the exploratory movement is diffusive with an additional velocity term. 
    """
    N = len(pos)

    if N == 0:
        return pos, state, params

    # -----------------------------
    # STATE1: attraction + diffusion + switching
    # -----------------------------
    idx1 = state == STATE1
    n1 = idx1.sum()
    if n1 > 0:
        pos1 = pos[idx1]
        home1 = params["home"][idx1]
        diff = pos1 - home1
        r = np.linalg.norm(diff, axis=1, keepdims=True)
        r[r == 0] = 1  # avoid division by zero
        pos1 -= state1_params["alpha"] * diff / r * dt
        pos1 += np.sqrt(2 * params["D1"][idx1][:, None] * dt) * np.random.randn(n1, 2)

        # clip before sampling B_density
        pos1[:, 0] = np.clip(pos1[:, 0], 0, L_w-1)
        pos1[:, 1] = np.clip(pos1[:, 1], 0, L_h-1)

        i = pos1[:, 1].astype(int)
        j = pos1[:, 0].astype(int)
        B_local = B_density[i, j]


        lambda_12_vals = lambda_12_of_B(B_local)
        #lambda_12_vals = lambda_12
        switch_mask = np.random.rand(n1) < lambda_12_vals

        idx_switch = np.where(idx1)[0][switch_mask]
        if len(idx_switch) > 0:
            state[idx_switch] = STATE2
            params["v"][idx_switch] = np.random.uniform(
                state2_params["v_lower"], state2_params["v_higher"], size=len(idx_switch)
            )
            params["D2"][idx_switch] = np.random.uniform(
                state2_params["D2_lower"], state2_params["D2_higher"], size=len(idx_switch)
            )
            params["Dtheta"][idx_switch] = np.random.uniform(
                state2_params["Dtheta_lower"], state2_params["Dtheta_higher"], size=len(idx_switch)
            )
            params["theta"][idx_switch] = np.random.uniform(0, 2*np.pi, size=len(idx_switch))
        pos[idx1] = pos1

    # -----------------------------
    # STATE2: persistent motion + diffusion + switching
    # -----------------------------
    idx2 = state == STATE2
    n2 = idx2.sum()
    if n2 > 0:
        theta2 = params["theta"][idx2]
        v2 = params["v"][idx2]
        D2 = params["D2"][idx2]
        Dtheta = params["Dtheta"][idx2]

        theta2 += np.sqrt(2 * Dtheta * dt) * np.random.randn(n2)

        # Fix broadcasting: expand v2 along columns
        dx = (v2[:, None] * np.column_stack([np.cos(theta2), np.sin(theta2)])) * dt
        dx += np.sqrt(2 * D2[:, None] * dt) * np.random.randn(n2, 2)
        pos2 = pos[idx2] + dx

        # switching to STATE1
        #switch_mask = np.random.rand(n2) < lambda_21 * dt
        #idx2_global = np.where(idx2)[0]
        #local_B = np.array([
        #    B[nearest_node(pos[k])] for k in idx2_global
        #])
        lambda_21_vals = lambda_21
        #lambda_21_vals = lambda_21_of_B(B_density)
        switch_mask = np.random.rand(n2) < lambda_21_vals

        idx_switch = np.where(idx2)[0][switch_mask]
        if len(idx_switch) > 0:
            state[idx_switch] = STATE1
            params["home"][idx_switch] = pos[idx_switch].copy()
            params["D1"][idx_switch] = np.random.uniform(
                state1_params["D1_0"], state1_params["D1_final"], size=len(idx_switch)
            )

        params["theta"][idx2] = theta2
        pos[idx2] = pos2

    xmax = L_w - 1
    ymax = L_h - 1

    # X boundaries
    over_x = pos[:, 0] > xmax
    under_x = pos[:, 0] < 0

    if np.any(over_x):
        pos[over_x, 0] = xmax
        params["theta"][over_x] = np.random.uniform(0, 2*np.pi, size=over_x.sum())

    if np.any(under_x):
        pos[under_x, 0] = 0
        params["theta"][under_x] = np.random.uniform(0, 2*np.pi, size=under_x.sum())

    # Y boundaries
    over_y = pos[:, 1] > ymax
    under_y = pos[:, 1] < 0

    if np.any(over_y):
        pos[over_y, 1] = ymax
        params["theta"][over_y] = np.random.uniform(0, 2*np.pi, size=over_y.sum())

    if np.any(under_y):
        pos[under_y, 1] = 0
        params["theta"][under_y] = np.random.uniform(0, 2*np.pi, size=under_y.sum())

    return pos, state, params

# =========================
# SIMULATION
# =========================
def simulate(output_dir, snapshot_interval=1.0):

    B = np.full((L_h, L_w), initial_hare, dtype=int)

    N = int(L_h * L_w * initial_lynx)

    pos = np.zeros((N, 2))
    #pos[:, 0] = np.random.uniform(L_w*0.8, L_w-1, N)
    pos[:, 0] = np.random.uniform(0, L_w-1, N)
    pos[:, 1] = np.random.uniform(0, L_h-1, N)

    state = np.ones(N, dtype=int)

    params = {
        "home": pos.copy(),
        "D1": np.random.uniform(state1_params["D1_0"],
                                state1_params["D1_final"], N),
        "v": np.zeros(N),
        "D2": np.zeros(N),
        "theta": np.zeros(N),
        "Dtheta": np.zeros(N),
    }

    history = []


    t = 0.0
    output_dir.mkdir(exist_ok=True)
    #h5_path = output_dir / "simulation.h5"
    #h5file = h5py.File(h5_path, "w")
    # Variable-length lynx storage
    #vlen_float = h5py.vlen_dtype(np.float32)
    #vlen_int = h5py.vlen_dtype(np.int32)

    # Main grid dataset 
    #dset_B = h5file.create_dataset("B",shape=(STEPS, L_h, L_w), dtype="int16", compression="gzip", chunks=(1, L_h, L_w),)
    #dset_time = h5file.create_dataset("time", shape=(STEPS,), dtype="float32",)
    #dset_pos = h5file.create_dataset("lynx_pos", shape=(STEPS,), dtype=vlen_float,)
    #dset_state = h5file.create_dataset("lynx_state", shape=(STEPS,), dtype=vlen_int,)

    for step in range(STEPS):

        # =============================
        # Precompute density (CRITICAL)
        # =============================
        B_density = (
            B
            + np.roll(B, 1, axis=0)
            + np.roll(B, -1, axis=0)
            + np.roll(B, 1, axis=1)
            + np.roll(B, -1, axis=1)
        )

        # =============================
        # Hare dynamics (vectorized, fast)
        # =============================
        # Hare death
        B -= np.random.poisson(sigma * B * DT)
        B = np.maximum(B, 0)

        # Hare birth into neighbors
        birth_counts = np.random.poisson(beta0 * B * DT)
        total_births = birth_counts.sum()

        if total_births > 0:
            # Create flat arrays of parent coordinates
            parent_i, parent_j = np.nonzero(birth_counts)
            repeats = birth_counts[parent_i, parent_j]
            parents_i = np.repeat(parent_i, repeats)
            parents_j = np.repeat(parent_j, repeats)

            # Choose random neighbor offsets
            directions = np.array([[1,0], [-1,0], [0,1], [0,-1]])
            choices = np.random.randint(0, 4, size=len(parents_i))
            di_dj = directions[choices]
            
            offspring_i = parents_i + di_dj[:,0]
            offspring_j = parents_j + di_dj[:,1]

            # Clip to grid
            offspring_i = np.clip(offspring_i, 0, L_h-1)
            offspring_j = np.clip(offspring_j, 0, L_w-1)

            # Apply births
            np.add.at(B, (offspring_i, offspring_j), 1)

        # =============================
        # Lynx movement
        # =============================
        pos, state, params = move_lynx(pos, state, params, B_density, 4.0)

        if len(pos) == 0:
            break

        # =============================
        # Vectorized predation
        # =============================
        # Lynx grid locations
        i = pos[:, 1].astype(int)
        j = pos[:, 0].astype(int)

        # Local hare density (ONLY the actual cell)
        B_local = B[i, j]

        # Expected number eaten per lynx
        expected_pred = mu * B_local * DT

        # Draw predation events
        num_eaten = np.random.poisson(expected_pred)

        # Aggregate total predation per grid cell
        total_eaten = np.zeros_like(B)
        np.add.at(total_eaten, (i, j), num_eaten)

        # Prevent removing more hares than exist
        total_eaten = np.minimum(total_eaten, B)

        # Apply predation
        B -= total_eaten

        # =============================
        # Vectorized death
        # =============================
        death_rate = lambda_of_B(B_density[i, j])
        # Number of death events per lynx
        death_events = np.random.poisson(death_rate * DT)
        # If ≥1 event → lynx dies
        alive_mask = death_events == 0
        pos = pos[alive_mask]
        state = state[alive_mask]
        for key in params:
            params[key] = params[key][alive_mask]


        # =============================
        # Vectorized lynx births 
        # =============================
        if len(pos) > 0:
            birth_events = np.random.poisson(alpha0 * DT, size=len(pos))

            parents_mask = birth_events > 0

            if np.any(parents_mask):

                parents = pos[parents_mask]
                B_local = B_density[
                    parents[:, 1].astype(int),
                    parents[:, 0].astype(int)
                ]
                offspring_counts = k_of_B(B_local)
                # If multiple birth events happened in same step (which theoretically shouldn't happen), multiply offspring accordingly
                offspring_counts *= birth_events[parents_mask]
                total_offspring = offspring_counts.sum()

                if total_offspring > 0:
                    # All offspring appear at the parent's current position
                    new_pos = np.repeat(parents, offspring_counts, axis=0)

                    # Clip to grid (safety, though should be redundant)
                    new_pos[:, 0] = np.clip(new_pos[:, 0], 0, L_w-1)
                    new_pos[:, 1] = np.clip(new_pos[:, 1], 0, L_h-1)

                    # Add new lynx
                    pos = np.vstack([pos, new_pos])
                    state = np.concatenate([state, np.ones(len(new_pos), dtype=int)])

                    params["home"] = np.vstack([params["home"], new_pos])
                    params["D1"] = np.concatenate([
                        params["D1"],
                        np.random.uniform(state1_params["D1_0"],
                                        state1_params["D1_final"],
                                        len(new_pos))
                    ])
                    for key in ["v", "D2", "theta", "Dtheta"]:
                        params[key] = np.concatenate([params[key], np.zeros(len(new_pos))])
        # Hare movement (diffusion)
        if D_B > 0 and B.sum() > 0:
            # Probability to move per hare per timestep
            move_prob = 1 - np.exp(-D_B * DT)

            hare_coords = np.argwhere(B > 0)  # positions of hares
            n_hare = len(hare_coords)

            if n_hare > 0:
                move_mask = np.random.rand(n_hare) < move_prob
                moving_hares = hare_coords[move_mask]

                if len(moving_hares) > 0:
                    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
                    choices = np.random.randint(0, 4, size=len(moving_hares))
                    displacements = directions[choices]

                    new_positions = moving_hares + displacements
                    new_positions[:, 0] = np.clip(new_positions[:, 0], 0, L_h-1)
                    new_positions[:, 1] = np.clip(new_positions[:, 1], 0, L_w-1)

                    # Flatten indices
                    old_idx = np.ravel_multi_index((moving_hares[:,0], moving_hares[:,1]), dims=B.shape)
                    new_idx = np.ravel_multi_index((new_positions[:,0], new_positions[:,1]), dims=B.shape)

                    # Remove from old and add to new positions
                    B_ravel = B.ravel()
                    np.add.at(B_ravel, old_idx, -1)
                    np.add.at(B_ravel, new_idx, 1)
                    B = B_ravel.reshape(B.shape)

                    # Ensure no negative densities
                    B = np.maximum(B, 0)

        t += DT
        print(t)
        history.append((t, B.sum(), len(pos)))
        # Append one timestep directly to disk
        #dset_B[step] = B.astype(np.int16)
        #dset_time[step] = t
        #dset_pos[step] = pos.astype(np.float32).ravel()
        #dset_state[step] = state.astype(np.int32)
    #h5file.close()
    return (np.array(history),
            B, pos, state)


# Worker function
def run_simulation(i):
    print(f"Starting run {i}")

    output_dir = Path(
        f"/Users/alissadomenig/repositories/solitary_animals/"
        f"solitary_animals/Code/CleanRepo/SpedUp/"
        f"outputs/mc_{L_h}x{L_w}/density_switching"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    history, B_final, lynx_final, state = simulate(output_dir)

    history_array = np.array(history)

    np.savez_compressed(
        output_dir / "simulation_data.npz",
        history=history_array
    )

    print(f"Finished run {i}")
    return i


if __name__ == "__main__":

    n_runs = 1
    n_cores = min(mp.cpu_count(), 8)

    print(f"Running {n_runs} simulations on {n_cores} cores")

    start_time = time.time()

    with mp.Pool(processes=n_cores) as pool:
        pool.map(run_simulation, range(n_runs))

    print("All simulations complete.")
    print("Total wall time:", time.time() - start_time)
# Simulation Output File Structure

For each simulation replicate, two output files are produced.

## 1. `scenarioX_runY.npz`

This file contains summary information for the simulation and the final system state.

### Contents

#### `history`

Time series of global population statistics.

Columns:

1. Simulation time (years)
2. Total hare abundance
3. Number of lynx in State 1 (territorial)
4. Number of lynx in State 2 (exploratory)

#### `B_final`

Final hare density grid.

- Shape: `(1000, 1000)`
- Contains the number of hares in each spatial cell at the end of the simulation.

#### `lynx_pos`

Final lynx coordinates.

- Shape: `(N_lynx, 2)`
- Columns correspond to `(x, y)` position in kilometers.

#### `lynx_state`

Behavioral state of each lynx at the final timestep.

- Shape: `(N_lynx,)`
- State 1 = territorial
- State 2 = exploratory

### Example

```python
import numpy as np

data = np.load("scenario0_run0.npz")

history = data["history"]
B_final = data["B_final"]
lynx_pos = data["lynx_pos"]
lynx_state = data["lynx_state"]
```

---

## 2. `snapshots_Y.h5`

This file contains spatial snapshots of the simulation through time.

### Datasets

#### `B`

Hare density field.

- Shape: `(n_snapshots, 1000, 1000)`

Each slice:

```python
B[i]
```

contains the full hare density grid at snapshot `i`.

#### `time`

Simulation time corresponding to each snapshot.

- Shape: `(n_snapshots,)`

Example:

```python
time[i]
```

returns the simulation time (years) of snapshot `i`.

#### `state1_pos`

Coordinates of all territorial lynx at each snapshot.

- Variable-length dataset
- Stored as a flattened array:

```text
[x1, y1, x2, y2, x3, y3, ...]
```

To reconstruct coordinates:

```python
coords = state1_pos[i].reshape(-1, 2)
```

#### `state2_pos`

Coordinates of all exploratory lynx at each snapshot.

- Variable-length dataset
- Same format as `state1_pos`

To reconstruct coordinates:

```python
coords = state2_pos[i].reshape(-1, 2)
```

### Example

```python
import h5py

with h5py.File("snapshots_0.h5", "r") as f:

    t = f["time"][10]

    state1 = f["state1_pos"][10].reshape(-1, 2)
    state2 = f["state2_pos"][10].reshape(-1, 2)

    B = f["B"][10]
```

In this example:

- `B` is the hare density field at snapshot 10.
- `state1` contains the coordinates of all territorial lynx.
- `state2` contains the coordinates of all exploratory lynx.
- `t` is the simulation time associated with the snapshot.

---

## Summary

### `.npz` file

Stores:

- Global population time series (`history`)
- Final hare density field (`B_final`)
- Final lynx coordinates (`lynx_pos`)
- Final lynx behavioral states (`lynx_state`)

### `.h5` file

Stores time-resolved spatial snapshots:

- Hare density field through time (`B`)
- Simulation times (`time`)
- Territorial lynx locations (`state1_pos`)
- Exploratory lynx locations (`state2_pos`)

This structure allows reconstruction of both population dynamics and spatial distributions throughout the simulation.

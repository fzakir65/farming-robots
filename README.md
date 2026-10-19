# Robot-Assisted Farming: a multi-agent system in Mesa

Amphibious picker robots and explorer drones work together to harvest strawberries on a
simulated farm with rows of trees, lanes and a river. There are three switchable operation modes:

| Mode | What happens |
|---|---|
| **Basic** | Pickers explore at random and return to base when full or low on battery. Drones find ripe crops and hover over them until a picker arrives. |
| **Extended** | Crops age, rot and regrow. Pickers wait at base until a drone signals crop locations, either **broadcast** to all pickers or sent **direct** to the nearest free one. |
| **Novel (Contract Net)** | Drones auction each crop area. Pickers bid their estimated cost (distance, queue, battery, storage) and the cheapest bidder wins the job. |

On top of the three modes, each part of the system can be swapped for a more advanced
technique: the auction format, the pickers' "brain", path finding, the drones' search strategy,
and the network the messages travel over. See [Advanced options](#advanced-options).

## Quick start (Windows PowerShell)

```powershell
cd farming-robots
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd src
python run.py                      # opens http://127.0.0.1:8521 in the browser
```

In the browser, pick the **Operation mode** and the advanced options from the drop-downs, adjust
the sliders, press **Reset**, then **Start**. The canvas, live KPI panel and charts update as the
simulation runs.

| Colour | Meaning |
|---|---|
| red / dark green / pale green squares | ripe / growing / empty trees |
| blue column | river |
| B / H | picker base / drone landing pad |
| circles | pickers (the number is strawberries carried; colour shows state, see legend) |
| black squares | drones (`z2` = altitude 2) |

## Advanced options

Every option is a `FarmModel(...)` argument and a drop-down in the web UI. The defaults reproduce
the original three modes exactly.

| Part | Options | Library | Module idea it extends |
|---|---|---|---|
| **Task allocation** (`allocation`) | Contract Net (first-price), **Vickrey** (second-price), **English** (descending clock), **Combinatorial** (bundle bids, CP-SAT winner determination), **Central planner** (OR-Tools vehicle routing, an "optimal" baseline), **LLM coordinator** (local Ollama) | OR-Tools, Ollama | WS7 auctions |
| **Picker brain** (`picker_brain`) | Rule-based, **BDI** (AgentSpeak plans, bold or cautious commitment), **Learned** (MaskablePPO policy shared by all pickers) | agentspeak, PettingZoo, Stable-Baselines3 | WS2–3 architectures |
| **Path finding** (`path_planner`) | Independent Dijkstra, **Windowed Conflict-Based Search** (multi-agent path finding with space-time A*) | — | WS2–3 Activity 3 |
| **Drone strategy** (`drone_strategy`) | Random roaming, **Nash sector game** (congestion game, best-response dynamics), **Fictitious play** | nashpy | WS6 game theory |
| **Network** (`network`) | Ideal, **Lossy radio** (dropped / delayed messages), **SPADE / XMPP** (real FIPA-style messages over a local XMPP server) | SPADE 4 | Messaging in Extended mode |
| **Bidders** (`strategic_bidders`) | Share of pickers that shade their bids to game the auction | — | WS7 Activity 3 |

```python
from farm.model import FarmModel
from farm.constants import NOVEL, VICKREY, CBS, NASH
m = FarmModel(mode=NOVEL, allocation=VICKREY, path_planner=CBS, drone_strategy=NASH, seed=1)
for _ in range(1000):
    m.step()
```

### Other interfaces

| Command (from `src/`) | What it opens |
|---|---|
| `python run.py` | Mesa 2 web UI (the module's version), http://127.0.0.1:8521 |
| `python run_geo.py` | The farm on a real OpenStreetMap map (Mesa-Geo), http://127.0.0.1:8522; `--export farm.geojson` writes GeoJSON for QGIS |
| `..\.venv-mesa3\Scripts\solara run app_solara.py` | The same model on **Mesa 3.x** with the Solara UI (separate environment: `requirements-mesa3.txt`) |
| `python train_rl.py` | Trains the learned picker policy (150k decisions → `models/picker_policy.zip`, already included) |

The model runs on both Mesa 2.1 and Mesa 3.x through `farm/compat.py`, and the test suite passes on
both.

### Local LLM coordinator

Needs the [Ollama](https://ollama.com) app running locally:

```powershell
ollama pull qwen2.5:3b
```

The drone sends one prompt describing the ripe areas and the pickers, and the model must answer
in a JSON schema. Every answer is validated (unknown ids, duplicates and overloaded pickers are
removed); if the call fails or times out the drone falls back to the nearest-free-picker rule and
the failure is counted. Nothing leaves the machine.

### ROS 2 + Gazebo (optional, Linux / WSL)

`ros2/farm_ros` is a ROS 2 Humble package that runs the simulation as a node and publishes the
robots as RViz markers, or generates a Gazebo world and keeps its robot models in sync.

Tested on WSL Ubuntu with ROS 2 Humble from [RoboStack](https://robostack.github.io) (micromamba,
no system install needed):

```bash
# 1. ROS 2 Humble + Gazebo bridge (one-off, a few GB)
micromamba create -n ros2_humble -c conda-forge -c robostack-staging \
    python=3.11 ros-humble-desktop ros-humble-ros-gz colcon-common-extensions
micromamba activate ros2_humble
pip install "mesa==2.1.0" ortools agentspeak "numpy==$(python -c 'import numpy; print(numpy.__version__)')"

# 2. build (copy out of OneDrive first; --symlink-install fails with new setuptools on Humble)
mkdir -p ~/farm_ws/src && cp -r ros2/farm_ros ~/farm_ws/src/
cd ~/farm_ws && colcon build && source install/setup.bash
export FARM_SRC="/mnt/c/.../farming-robots/src"     # where the Mesa model lives

# 3. run
ros2 launch farm_ros farm_rviz.launch.py mode:=Novel n_pickers:=6
ros2 launch farm_ros farm_gazebo.launch.py
```

The sim node publishes `/farm/pickers`, `/farm/drones` (PoseArray), `/farm/markers` (RViz),
`/farm/kpis` and `/farm/robot_poses` (JSON) plus `/tf`. Checked headless: every topic
publishes at the requested rate (10 Hz), and the generated `farm.sdf` passes `ign sdf -k` and
loads in Gazebo (ground, crops, river, bases and the robots). The RViz and Gazebo windows
themselves need a display (WSLg on Windows 11).

## Tests

```powershell
cd src
python -m pytest                    # 149 tests
python -m unittest discover -s tests -t .   # the original 66 unittest tests only
```

They cover:
* environment creation (layout, river, rows of 3, every tree reachable, random layouts)
* path finding, including CBS (no two pickers ever share a cell or swap places)
* the crop life-cycle
* every picker and drone action and decision rule, collision avoidance and battery safety
* both messaging styles, the lossy radio and a real XMPP exchange
* every auction format, CP-SAT winner determination and the VRP planner
* the BDI and learned picker brains, the drones' sector game
* the LLM coordinator's validation and fallback (with a fake client)
* end-to-end runs of every mode on Mesa 2 and Mesa 3
* **property-based tests** (Hypothesis): invariants checked on many random farms and auctions, e.g.
  strawberries are conserved and stay within capacity, CBS paths are conflict-free, second-price
  never pays less than first-price, and combinatorial winner determination is always feasible

## Experiments and KPI analysis

```powershell
cd src
python experiments.py --sweep                  # original modes: 4 configurations x 10 seeds x 1000 steps
jupyter notebook analysis.ipynb                # plots and critical analysis

python advanced_experiments.py compare         # every advanced option, 20 seeds x 1000 steps, in parallel
python advanced_experiments.py batch           # Mesa batch_run fleet-design grid
python advanced_experiments.py sobol           # SALib Sobol sensitivity analysis
python advanced_experiments.py optuna          # Optuna multi-objective fleet design
python advanced_experiments.py llm --seeds 3   # LLM coordinator (slow)
python advanced_experiments.py xmpp --seeds 3  # real XMPP network vs ideal bus
python advanced_experiments.py stats           # statistical tests on saved results
```

Add `--workers 4` if the machine runs short of memory. Results are written to `results/` and
`results/advanced/` as CSV files and PNG plots.

### Original modes (mean over 10 seeds)

| Configuration | Harvest rate | Energy per strawberry | Pick latency (steps) | Messages |
|---|---|---|---|---|
| Basic (no ageing) | 1.00 | 16.9 | 155 | 0 |
| Extended, broadcast | 0.58 | 6.7 | 58 | 100 |
| Extended, direct | 0.62 | 5.8 | 58 | 93 |
| Novel, Contract Net | 0.61 | 5.9 | 58 | 848 |

Contract Net matches direct messaging and varies less between runs, but uses about 9× more
messages. Fleet size matters more than coordination: going from 2 to 8 pickers lifts the harvest
rate from about 0.42 to about 0.74.

### Advanced options (mean over 20 seeds)

Each configuration runs on the same 20 seeds. Differences are tested with Mann-Whitney U, Holm
correction across the comparisons, the Vargha-Delaney A12 effect size and bootstrap confidence
intervals (`stats_tools.py`, full table in `results/advanced/stats.csv`).

| Comparison | Harvest rate | Other KPIs | Significant? |
|---|---|---|---|
| Contract Net vs **Vickrey** | 0.61 vs 0.61 | identical allocations | no — with truthful bidders first- and second-price pick the same winner; only the price differs |
| Contract Net vs **English** | 0.61 vs 0.63 | 856 vs **5,930** messages | no |
| Contract Net vs **Combinatorial** | 0.61 vs 0.61 | 856 vs 659 messages | no |
| Contract Net vs **Central VRP** | 0.61 vs 0.59 | 3–4× slower to run | no — decentralised bidding keeps up with the central optimiser |
| Independent vs **CBS**, 4 pickers | 0.62 vs **0.65** | blocked steps 175 → **8** | **yes** |
| Independent vs **CBS**, 8 pickers | 0.75 vs 0.76 | blocked steps 444 → **17** | no (harvest), large drop in blocking |
| Random roaming vs **Nash sectors** | 0.62 vs **0.59** | | **yes, worse** — stable for each drone, not best for the team |
| Random roaming vs **Fictitious play** | 0.62 vs 0.63 | | no |
| Ideal vs **lossy radio** 10 / 30 / 50 % | 0.61 vs 0.60 / **0.56** / **0.43** | | **yes** from 30 % loss |
| Rule-based vs **BDI** / **Learned (PPO)** | 0.60 vs 0.59 / 0.57 | | no — the hand-written rules are already near what the policy learned |
| Contract Net vs **LLM coordinator** (5 seeds) | 0.60 vs 0.62 | 881 vs **172** messages; **38×** slower; ~20 invalid assignments caught per run | no |
| Ideal bus vs **SPADE / XMPP** (5 seeds) | 0.60 vs 0.64 | no messages lost; 5× slower | no |

The full analysis of every comparison, with plots and tests, is section 6 of `src/analysis.ipynb`.

### What drives the results (Sobol sensitivity, 512 runs)

Total-order Sobol indices (`ST`, share of the output's variance a setting explains, including its
interactions; `results/advanced/sobol.csv`):

| Setting | Harvest rate | Energy per strawberry |
|---|---|---|
| Number of pickers | **0.52** | 0.16 |
| Number of drones | 0.46 | **0.71** |
| Picker capacity | 0.28 | 0.35 |
| Initial ripe patches | 0.16 | 0.11 |
| Message loss | 0.15 | 0.15 |
| Share of strategic bidders | 0.14 | 0.11 |

Fleet design (how many pickers and drones) matters far more than how the robots bid or talk. With
64 base samples the confidence intervals are wide, so read the ranking, not the exact values.

### Cheapest fleet for a target harvest (Optuna, 64 trials)

A two-objective search (maximise harvest rate, minimise fleet cost) gives a Pareto front
(`results/advanced/optuna_pareto.png`). For example, about **0.80** harvest needs 4 pickers + 4
drones with capacity 20 (cost 72), and about **0.90** needs 6 pickers + 5 drones (cost 104). Almost
every Pareto-optimal fleet used the Vickrey auction.

## Project layout

```
farming-robots/
├── README.md
├── requirements.txt          # Mesa 2.1 environment (everything)
├── requirements-mesa3.txt    # optional Mesa 3.x + Solara environment
├── docs/design.md            # statecharts, protocol sequence diagram, KPI definitions, references
├── models/picker_policy.zip  # trained MaskablePPO policy
├── results/                  # experiment output (CSV + PNG); advanced/ for the new options
├── ros2/farm_ros/            # ROS 2 package: sim node, RViz + Gazebo launch files
└── src/
    ├── run.py                # Mesa 2 web UI
    ├── run_geo.py            # Mesa-Geo map UI
    ├── app_solara.py         # Mesa 3 Solara UI
    ├── experiments.py        # original batch runs + plots
    ├── advanced_experiments.py   # comparisons, batch_run, Sobol, Optuna, LLM, XMPP
    ├── stats_tools.py        # Mann-Whitney U, Holm, A12, bootstrap CIs
    ├── train_rl.py           # trains the learned picker policy
    ├── analysis.ipynb        # KPI analysis notebook
    ├── farm/
    │   ├── constants.py      # every tunable parameter and option name
    │   ├── environment.py    # Tree/Water/Base/Ground agents, layout, FarmMap path finding
    │   ├── messages.py       # message bus, lossy radio, performatives
    │   ├── xmpp_bus.py       # SPADE / XMPP message bus
    │   ├── picker.py         # PickerRobot agent
    │   ├── drone.py          # ExplorerDrone agent
    │   ├── auctions.py       # Vickrey, English, combinatorial (CP-SAT)
    │   ├── planner.py        # OR-Tools VRP central planner
    │   ├── llm.py            # Ollama coordinator
    │   ├── mapf.py           # windowed Conflict-Based Search
    │   ├── bdi.py, asl/      # BDI picker (AgentSpeak)
    │   ├── rl.py             # PettingZoo / Gymnasium environments
    │   ├── sectors.py        # drones' sector game (Nash, fictitious play)
    │   ├── geo.py            # Mesa-Geo model
    │   ├── compat.py         # runs on Mesa 2.1 and Mesa 3.x
    │   ├── model.py          # FarmModel + KPI reporters
    │   ├── portrayal.py      # canvas drawing
    │   └── server.py         # Mesa ModularServer, option drop-downs and charts
    └── tests/                # unittest + pytest + Hypothesis suite
```

## Notes

* Built for **Mesa 2.1** (`ModularServer` / `mesa.time` schedulers), as the module uses. Mesa 3
  support is a separate optional environment.
* Every behaviour parameter (battery costs, crop ages, arm reach, auction deadline…) is in
  `src/farm/constants.py`.
* **Generative AI:** this code base, its tests, the notebook and the documentation were produced
  with Claude Code (Anthropic). If you adapt any of it for assessed work, check your module's
  rules on GenAI use and declare it as they require.

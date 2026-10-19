"""Shared constants and tunable parameters for the robot-assisted farm simulation.

Everything that controls the behaviour of the environment or the robots lives here,
so experiments can change one value without hunting through the code.
"""

# --------------------------------------------------------------------------
# Operation modes (switchable in the UI or via ``FarmModel(mode=...)``)
# --------------------------------------------------------------------------
BASIC = "Basic"
EXTENDED = "Extended"
NOVEL = "Novel"
MODES = [BASIC, EXTENDED, NOVEL]

# How drones contact pickers in the extended mode
BROADCAST = "Broadcast"
DIRECT = "Direct"
COMM_MODES = [BROADCAST, DIRECT]

# How crop areas are allocated to pickers in the novel mode
CONTRACT_NET = "Contract Net (first-price)"
VICKREY = "Vickrey (second-price)"
ENGLISH = "English (descending clock)"
COMBINATORIAL = "Combinatorial (CP-SAT)"
CENTRAL_VRP = "Central planner (OR-Tools VRP)"
LLM = "LLM coordinator (Ollama)"
ALLOCATIONS = [CONTRACT_NET, VICKREY, ENGLISH, COMBINATORIAL, CENTRAL_VRP, LLM]
AUCTIONS = [CONTRACT_NET, VICKREY, ENGLISH, COMBINATORIAL]

# Decision architecture of the picker robots
RULES = "Rule-based"
BDI = "BDI (AgentSpeak)"
RL = "Learned (MaskablePPO)"
PICKER_BRAINS = [RULES, BDI, RL]
BOLD = "Bold"               # BDI commitment strategies (Kinny & Georgeff, 1991)
CAUTIOUS = "Cautious"
COMMITMENTS = [BOLD, CAUTIOUS]

# Multi-robot path planning
INDEPENDENT = "Independent (Dijkstra)"
CBS = "Windowed CBS"
PATH_PLANNERS = [INDEPENDENT, CBS]

# Where the drones search
ROAM = "Random roaming"
NASH = "Nash sector game"
FICTITIOUS = "Fictitious play"
DRONE_STRATEGIES = [ROAM, NASH, FICTITIOUS]

# How messages travel
IDEAL = "Ideal"
LOSSY = "Lossy radio"
XMPP = "SPADE / XMPP"
NETWORKS = [IDEAL, LOSSY, XMPP]

# --------------------------------------------------------------------------
# Environment layout
# --------------------------------------------------------------------------
DEFAULT_WIDTH = 32          # fits 6 rows of trees + river, like Figure 1 of the brief
DEFAULT_HEIGHT = 30
TREE_ROW_WIDTH = 3          # trees are organised in rows of 3
LANE_WIDTH = 2              # walkable lane between two rows of trees
RIVER_WIDTH = 2
PATCH_LENGTH = 3            # a crop "area" is a 3 x 3 block of trees
TREE_HEIGHT = 1             # drones must fly above this altitude to cross a tree

# --------------------------------------------------------------------------
# Crop life-cycle (only active in the extended and novel modes)
# --------------------------------------------------------------------------
RIPE_AGE = 250              # steps from seeding until strawberries are ripe
ROT_AGE = 400               # ripe strawberries not picked by this age are wasted
REPLANT_JITTER = 150        # random delay so replanted areas do not ripen together
FRUITING_PROB = 0.5         # chance that a seeded tree bears fruit this cycle

# --------------------------------------------------------------------------
# Amphibious picker robots
# --------------------------------------------------------------------------
PICKER_CAPACITY = 10
ARM_REACH = 2               # Chebyshev reach of the arm: covers a 3-wide row from a lane
SENSOR_RANGE = 3            # how far a picker can see ripe strawberries
WATER_SLOWDOWN = 2          # steps needed to leave a water cell (speed limited in water)
PICKER_BATTERY = 100.0
PICKER_IDLE_COST = 0.05     # energy drained every step the robot is switched on
PICKER_MOVE_COST = 0.3
PICKER_PICK_COST = 0.2
PICKER_CHARGE_RATE = 5.0
PICKER_RESERVE = 10.0       # safety margin kept when deciding to return to base
RETURN_MARGIN = 1.5         # energy-to-base is multiplied by this (allows for traffic)
EXPLORE_PATIENCE = 150      # basic mode: unload partial storage after this many fruitless steps
URGENCY_WEIGHT = 20.0       # how many path steps one unit of urgency is worth

# --------------------------------------------------------------------------
# Explorer drones (6 movement directions: front, back, left, right, up, down)
# --------------------------------------------------------------------------
DRONE_BATTERY = 100.0
DRONE_HOVER_COST = 0.1
DRONE_ALTITUDE_COST = 0.05  # extra drain per altitude level
DRONE_MOVE_COST = 0.15
DRONE_CHARGE_RATE = 6.0
DRONE_RESERVE = 4.0
MIN_ALTITUDE = 0
MAX_ALTITUDE = 3
CRUISE_ALTITUDE = 2
ALTITUDE_CHANGE_PROB = 0.05  # chance per step of switching between altitudes 2 and 3
DRONE_MAX_WAIT = 80          # basic mode: stop waiting for a picker after this many steps
REREPORT_INTERVAL = 40       # extended/novel: re-announce crops still unpicked after this
CFP_DEADLINE = 2             # novel: steps a drone collects proposals before awarding

# --------------------------------------------------------------------------
# Auctions and other allocation methods (novel mode)
# --------------------------------------------------------------------------
STRATEGIC_MARKUP = (0.1, 0.8)   # a strategic bidder inflates first-price bids by this share
ENGLISH_START_PRICE = 120.0     # descending clock: opening price (cost units)
ENGLISH_DECREMENT = 4.0         # price drop per round
BUNDLE_WINDOW = 3               # combinatorial/LLM: steps a drone gathers areas before a call
MAX_BUNDLE_AREAS = 4            # areas offered in one combinatorial call
MAX_BUNDLE_SIZE = 3             # areas a picker may bid for together
AREA_VALUE = 100.0              # auctioneer's value of one strawberry (winner determination)
VRP_INTERVAL = 10               # central planner: steps between re-optimisations
VRP_TIME_LIMIT = 0.2            # seconds per OR-Tools solve
LLM_MODEL = "qwen2.5:3b"        # local Ollama model used by the LLM coordinator
LLM_TIMEOUT = 60.0              # seconds before an LLM call falls back to the heuristic

# --------------------------------------------------------------------------
# Path planning, drones' sector game and networks
# --------------------------------------------------------------------------
CBS_WINDOW = 12                 # CBS resolves conflicts this many steps ahead
CBS_MAX_NODES = 60              # high-level search budget before prioritised planning
SECTOR_PERIOD = 60              # drones re-choose their search sector this often
SECTOR_LEARNING_RATE = 0.2      # weight of new observations in a drone's sector values
DEFAULT_DROP_PROB = 0.1         # lossy radio: chance a message is lost
DEFAULT_MSG_DELAY = 1           # lossy radio: steps before a message arrives
XMPP_PORT = 5222

# --------------------------------------------------------------------------
# Simulation control
# --------------------------------------------------------------------------
DEFAULT_MAX_STEPS = 1000

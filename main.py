""" Algae Bioreactor RL vs Static Controller Simulation """
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

''' Biological Constants '''
MU_MAX = 0.06             # maximum specific growth rate
KS = 0.5                  # nutrient half-saturation
KI = 100.0                # light half-saturation
K_LIGHT = 0.01            # light attenuation coefficient

T_OPT = 25.0              # optimal temperature (C)
SIGMA_T = 5.0             # temp tolerance
K_DECAY = 0.004           # base respiration

DT = 1.0
MAX_TIME = 500

Q_MIN = 0.05              # minimal internal nutrient quote
U_XS = 0.5                # biomass yield on substrate

# Thermal properties
HEAT_CAPACITY = 4180
CULTURE_VOLUME = 2.0
HEAT_LOSS = 0.02
AMBIENT_TEMP = 22.0
MAX_HEATER_POWER = 400

# Initial Conditions
X_INIT = 1.0           # biomass concentration
S_INIT = 5.0           # nutrient concentration
T_INIT = 25.0          # temperature
I_INIT = 60.0          # light intensity

# Limits
MAX_LIGHT = 200.0
MAX_NUTRIENT_RATE = 1.0
REACTOR_DEPTH = 0.2    # meters

# Resource budgets
DAILY_NUTRIENT_BUDGET = 50.0
DAILY_LIGHT_BUDGET = 50000.0
DAILY_HEAT_BUDGET = 200000.0

DO_YIELD = 1.0

''' Limitation Functions '''
def nutrient_limitation(S, X):
    # Droop Model
    Q = S / (X + 1e-6)
    mu_N = np.maximum(0.0, 1 - Q_MIN / Q)
    return mu_N

def light_limitation(I, X):
    # Beer-Lambert law integrated over reactor depth
    if X < 1e-6:
        I_avg = I
    else:
        I_avg = I * (1 - np.exp(-K_LIGHT * X * REACTOR_DEPTH)) / (K_LIGHT * X * REACTOR_DEPTH)
    mu_I = I_avg / (KI + I_avg)
    return mu_I

def temperature_limitation(T):
    # Gaussian
    return np.exp(-((T - T_OPT) / SIGMA_T) ** 2)

def growth_rate(X, S, T, I):
    mu = MU_MAX * nutrient_limitation(S, X) * light_limitation(I, X) * temperature_limitation(T)
    dXdt = mu * X - K_DECAY * X
    return dXdt

''' Environment '''
class AlgaeBioreactorEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Observation: [biomass, nutrient, temperature, light]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([7.0, 20.0, 40.0, MAX_LIGHT], dtype=np.float32)
        )

        # Action: normalized [-1,1] for nutrient feed, light intensity, and heater power
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.X = X_INIT
        self.S = S_INIT
        self.T = T_INIT
        self.I = I_INIT
        self.time = 0

        self.nutrient_budget = DAILY_NUTRIENT_BUDGET
        self.light_budget = DAILY_LIGHT_BUDGET
        self.heat_budget = DAILY_HEAT_BUDGET

        self.history = []

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.X, self.S, self.T, self.I], dtype=np.float32)

    def step(self, action):
        nutrient_a, light_a, heater_a = action

        nutrient_rate = (nutrient_a+1)/2 * MAX_NUTRIENT_RATE
        light_level = (light_a+1)/2 * MAX_LIGHT
        heater_power = (heater_a+1)/2 * MAX_HEATER_POWER

        # Resource limits
        nutrient_use = nutrient_rate*DT
        light_use = light_level*DT
        heat_use = heater_power*DT

        # Reset daily budget every 24 hours
        if int(self.time) % 24 == 0 and self.time != 0:
            self.nutrient_budget = DAILY_NUTRIENT_BUDGET
            self.light_budget = DAILY_LIGHT_BUDGET
            self.heat_budget = DAILY_HEAT_BUDGET

        # Stop if budgets are depleted
        if self.nutrient_budget<=0:
            nutrient_rate=0
        if self.light_budget<=0:
            light_level=0
        if self.heat_budget<=0:
            heater_power=0

        self.nutrient_budget-=nutrient_use
        self.light_budget-=light_use
        self.heat_budget-=heat_use

        self.I = light_level
        self.S += nutrient_rate*DT

        # Temperature dynamics
        thermal_mass = CULTURE_VOLUME*HEAT_CAPACITY
        dT = (heater_power - HEAT_LOSS*(self.T-AMBIENT_TEMP))/thermal_mass
        self.T += dT*DT

        # Growth
        dX = growth_rate(self.X,self.S,self.T,self.I)*DT
        self.X += dX
        self.S -= max(dX,0)*0.5
        self.S = max(self.S,0)

        DO_output = max(dX,0)*DO_YIELD
        reward = DO_output

        self.time+=DT
        terminated = bool(self.time>=MAX_TIME or self.X<=0)
        truncated=False

        self.history.append([
            self.time,self.X,self.S,self.T,self.I,
            nutrient_rate,light_level,heater_power,DO_output
        ])

        return self._get_obs(), reward, terminated, truncated, {}

''' Static Controller '''
def static_controller(state):
    return np.array([0.0,0.0,0.0],dtype=np.float32)

''' Train RL '''
env = AlgaeBioreactorEnv()
check_env(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99
)

model.learn(total_timesteps=300_000)

''' Run Simulation '''
def run_controller(env, controller=None, is_rl=True):
    obs, _ = env.reset()
    done = False

    while not done:
        if is_rl:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            action=static_controller(obs)

        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

    return pd.DataFrame(env.history, columns=[
        "time", "biomass", "nutrient", "temperature", "light",
        "nutrient_rate", "light_level", "heater_power", "DO"
    ])

rl_data = run_controller(AlgaeBioreactorEnv(), model, True)
static_data = run_controller(AlgaeBioreactorEnv(), None, False)

''' Save Data '''
rl_data.to_csv("rl_bioreactor_results.csv", index=False)
static_data.to_csv("static_bioreactor_results.csv", index=False)

''' Plots'''
plt.figure()
plt.plot(rl_data.time,rl_data.biomass,label="RL")
plt.plot(static_data.time,static_data.biomass,label="Static")
plt.legend()
plt.title("Biomass Growth")
plt.show()

plt.figure()
plt.plot(rl_data.time,rl_data.DO,label="DO Output RL")
plt.title("Carbon Fixation / DO")
plt.show()

fig, ax1 = plt.subplots(figsize=(10,5))

ax1.plot(rl_data.time, rl_data.nutrient_rate, 'g-', label='Nutrient rate')
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Nutrient rate (mg/L/h)', color='g')
ax1.tick_params(axis='y', labelcolor='g')

ax2 = ax1.twinx()  # second y-axis
ax2.plot(rl_data.time, rl_data.light_level, 'b-', label='Light')
ax2.plot(rl_data.time, rl_data.heater_power, 'r-', label='Heater')
ax2.set_ylabel('Light (μmol/m²/s) & Heater Power (W)', color='b')
ax2.tick_params(axis='y', labelcolor='b')

fig.suptitle("RL Decisions Over Time")
fig.tight_layout()
fig.legend(loc='upper right')
plt.show()


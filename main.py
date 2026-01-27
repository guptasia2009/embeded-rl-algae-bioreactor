""" Algae Bioreactor RL vs Static Controller Simulation """
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# ---------------------- MODEL PARAMETERS ---------------------- #
MU_MAX = 0.04          # 1/hour
KS = 0.5               # nutrient half-saturation
KI = 50.0              # light half-saturation
K_LIGHT = 0.1          # light attenuation coefficient
T_OPT = 25.0           # °C
SIGMA_T = 5.0           # temperature tolerance
K_DECAY = 0.005        # decay / respiration rate

# Simulation parameters
DT = 1.0               # hours per step
MAX_TIME = 500         # hours per episode

# Initial conditions
X_INIT = 0.1           # biomass concentration
S_INIT = 5.0           # nutrient concentration
T_INIT = 25.0          # temperature
I_INIT = 50.0           # light intensity

# Action limits
MAX_LIGHT = 200.0
MAX_NUTRIENT = 1.0

# ---------------------- LIMITATION FUNCTIONS ---------------------- #
def nutrient_limitation(S):
    return S / (KS + S)

def light_limitation(I, X):
    I_eff = I * np.exp(-K_LIGHT * X)
    return I_eff / (KI + I_eff)

def temperature_limitation(T):
    return np.exp(-((T - T_OPT) / SIGMA_T) ** 2)

def algae_growth_rate(X, S, T, I):
    mu = MU_MAX * nutrient_limitation(S) * light_limitation(I, X) * temperature_limitation(T)
    dXdt = mu * X - K_DECAY * X
    return dXdt

# ---------------------- ALGAE BIOREACTOR ENV ---------------------- #
class AlgaeBioreactorEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Observation: [biomass, nutrient, temperature, light]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 20.0, 40.0, 300.0], dtype=np.float32),
            dtype=np.float32
        )

        # Action: normalized [-1,1] for both nutrient feed and light intensity
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.X = X_INIT
        self.S = S_INIT
        self.T = T_INIT
        self.I = I_INIT
        self.time = 0
        self.history = []
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.X, self.S, self.T, self.I], dtype=np.float32)

    def step(self, action):
        # Scale normalized actions [-1,1] to actual nutrient/light
        nutrient_input = 0.5 * (action[0] + 1.0) * MAX_NUTRIENT
        light_input = 0.5 * (action[1] + 1.0) * MAX_LIGHT

        # Apply controls
        self.S += nutrient_input * DT
        self.I = np.clip(light_input, 0, MAX_LIGHT)

        # Biomass growth
        dX = algae_growth_rate(self.X, self.S, self.T, self.I) * DT
        self.X += dX

        # Nutrient consumption
        self.S -= max(dX, 0) * 0.5
        self.S = max(self.S, 0)

        # Reward: scaled biomass growth minus resource cost
        reward = 100*dX - 0.1*nutrient_input - 0.01*self.I

        # Episode termination
        terminated = bool(self.time >= MAX_TIME or self.X <= 0)
        truncated = False
        info = {}

        self.time += DT
        self.history.append([self.time, self.X, self.S, self.T, self.I, nutrient_input, light_input, reward])

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Time: {self.time:.1f}, Biomass: {self.X:.3f}")

# ---------------------- STATIC CONTROLLER ---------------------- #
def static_controller(state):
    nutrient_feed = 0.2
    light_level = 80.0
    return np.array([nutrient_feed, light_level])

# ---------------------- TRAIN RL AGENT ---------------------- #
env = AlgaeBioreactorEnv()
check_env(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99
)

model.learn(total_timesteps=500_000)  # longer training for slow dynamics

# ---------------------- EVALUATION LOOP ---------------------- #
def run_controller(env, controller=None, is_rl=True):
    obs, _ = env.reset()
    done = False
    while not done:
        if is_rl:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            # Convert static values to normalized [-1,1] scale
            static_action = static_controller(obs)
            action = np.array([
                2*static_action[0]/MAX_NUTRIENT - 1.0,
                2*static_action[1]/MAX_LIGHT - 1.0
            ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    df = pd.DataFrame(env.history, columns=[
        "time", "biomass", "nutrient", "temperature", "light",
        "nutrient_input", "light_input", "reward"
    ])
    return df

rl_data = run_controller(AlgaeBioreactorEnv(), model, is_rl=True)
static_data = run_controller(AlgaeBioreactorEnv(), None, is_rl=False)

# ---------------------- SAVE DATA ---------------------- #
rl_data.to_csv("rl_bioreactor_results.csv", index=False)
static_data.to_csv("static_bioreactor_results.csv", index=False)

# ---------------------- PLOT RESULTS ---------------------- #
plt.figure(figsize=(8,5))
plt.plot(rl_data["time"], rl_data["biomass"], label="RL Control")
plt.plot(static_data["time"], static_data["biomass"], label="Static Control")
plt.xlabel("Time (hours)")
plt.ylabel("Biomass Concentration")
plt.legend()
plt.title("RL vs Static Bioreactor Control")
plt.show()

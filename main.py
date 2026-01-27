""" INPUTS AND SETUP """
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

""" MODEL PARAMTERS & CONSTANTS """
# Growth parameters
MU_MAX = 0.04          # 1/hour
KS = 0.5               # nutrient half-saturation
KI = 50.0              # light half-saturation
K_LIGHT = 0.1          # light attenuation coefficient
T_OPT = 25.0           # °C
SIGMA_T = 5.0           # temperature tolerance
K_DECAY = 0.005         # decay / respiration rate

# Simulation parameters
DT = 1.0               # hours per step
MAX_TIME = 500         # hours per episode

# Initial conditions
X_INIT = 0.1            # biomass concentration
S_INIT = 5.0            # nutrient concentration
T_INIT = 25.0           # temperature
I_INIT = 50.0           # light intensity

# Action limits
MAX_LIGHT = 200.0
MAX_NUTRIENT = 1.0

""" LIMITATION FUNCTIONS (MONOD-BASED)"""
def nutrient_limitation(S):
    return S / (KS + S)

def light_limitation(I, X):
    I_eff = I * np.exp(-K_LIGHT * X)
    return I_eff / (KI + I_eff)

def temperature_limitation(T):
    return np.exp(-((T - T_OPT) / SIGMA_T) ** 2)

""" ALGAE GROWTH DYNAMICS """
def algae_growth_rate(X, S, T, I):
    mu = (
        MU_MAX
        * nutrient_limitation(S)
        * light_limitation(I, X)
        * temperature_limitation(T)
    )
    dXdt = mu * X - K_DECAY * X
    return dXdt

""" GYM ENVIRONMENT: ALGAE BIOREACTOR """
class AlgaeBioreactorEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        # State: [biomass, nutrient, temperature, light]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 10.0, 0.0]),
            high=np.array([10.0, 20.0, 40.0, 300.0]),
            dtype=np.float32
        )

        # Action: [nutrient feed rate, light intensity]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([MAX_NUTRIENT, MAX_LIGHT]),
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
        nutrient_input, light_input = action

        # Apply controls
        self.S += nutrient_input * DT
        self.I = np.clip(light_input, 0, MAX_LIGHT)

        # Integrate growth
        dX = algae_growth_rate(self.X, self.S, self.T, self.I) * DT
        self.X += dX

        # Nutrient consumption
        self.S -= max(dX, 0) * 0.5
        self.S = max(self.S, 0)

        # Reward: biomass growth - resource waste
        reward = dX - 0.01 * nutrient_input - 0.001 * self.I

        self.time += DT
        terminated = bool(self.time >= MAX_TIME or self.X <= 0)
        truncated = False  # no truncation used
        info = {}

        self.history.append([self.time, self.X, self.S, self.T, self.I, reward])

        return self._get_obs(), reward, terminated, truncated, info


    def render(self, mode="human"):
        print(f"Time: {self.time:.1f}, Biomass: {self.X:.3f}")

""" STATIC (RULE-BASED CONTROLLER) """
def static_controller(state):
    nutrient_feed = 0.2
    light_level = 80.0
    return np.array([nutrient_feed, light_level])

""" TRAIN THE RL AGENT """
env = AlgaeBioreactorEnv()
check_env(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99
)

model.learn(total_timesteps=200_000)

""" EVALUATION LOOP (RL VS. STATIC) """
def run_controller(env, controller, is_rl=True):
    obs, _ = env.reset()
    done = False

    while not done:
        if is_rl:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            action = static_controller(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return pd.DataFrame(
        env.history,
        columns=["time", "biomass", "nutrient", "temperature", "light", "reward"]
    )

rl_data = run_controller(AlgaeBioreactorEnv(), model, is_rl=True)
static_data = run_controller(AlgaeBioreactorEnv(), None, is_rl=False)

""" SAVE DATA """
rl_data.to_csv("rl_bioreactor_results.csv", index=False)
static_data.to_csv("static_bioreactor_results.csv", index=False)

""" PLOT RESULTS """
plt.figure()
plt.plot(rl_data["time"], rl_data["biomass"], label="RL Control")
plt.plot(static_data["time"], static_data["biomass"], label="Static Control")
plt.xlabel("Time (hours)")
plt.ylabel("Biomass Concentration")
plt.legend()
plt.title("RL vs Static Bioreactor Control")
plt.show()

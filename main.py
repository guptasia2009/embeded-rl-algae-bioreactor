""" INPUTS AND SETUP """
import numpy as np
import gym
from gym import spaces
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

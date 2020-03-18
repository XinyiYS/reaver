from .base import *
from .random import RandomAgent
from .a2c import AdvantageActorCriticAgent
from .ppo import ProximalPolicyOptimizationAgent
from .hai import HumanAIInteractionAgent

A2C = AdvantageActorCriticAgent
PPO = ProximalPolicyOptimizationAgent
HAI = HumanAIInteractionAgent

registry = {
    'a2c': A2C,
    'ppo': PPO,
    'hai': HAI,
}

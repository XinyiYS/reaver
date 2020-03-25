import gin.tf
import tensorflow.compat.v1 as tf

from reaver.envs.base import Spec
from reaver.utils import StreamLogger
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType
from reaver.agents.base import ActorCriticAgent, DEFAULTS
from .a2c import AdvantageActorCriticAgent


LOGGING_MSG_HEADER = "LOGGING FROM <reaver.reaver.agents.hai> "

# @gin.configurable('A2CAgent')
class HumanAIInteractionAgent(AdvantageActorCriticAgent):
    """
    HAI: an extension of the existing A2C agent class to
    include human command interaction
    """

    def __init__(
        self,
        obs_spec: Spec,
        act_spec: Spec,
        n_subagents: int = 0,
        subagents: list = [],  # need to specify a list of subagents: trained agents
        model_fn: ModelBuilder = None,
        policy_cls: PolicyType = None,
        sess_mgr: SessionManager = None,
        optimizer: tf.train.Optimizer = None,
        n_envs=1,
        value_coef=DEFAULTS['value_coef'],
        entropy_coef=DEFAULTS['entropy_coef'],
        traj_len=DEFAULTS['traj_len'],
        batch_sz=DEFAULTS['batch_sz'],
        discount=DEFAULTS['discount'],
        gae_lambda=DEFAULTS['gae_lambda'],
        clip_rewards=DEFAULTS['clip_rewards'],
        clip_grads_norm=DEFAULTS['clip_grads_norm'],
        normalize_returns=DEFAULTS['normalize_returns'],
        normalize_advantages=DEFAULTS['normalize_advantages'],
    ):
        AdvantageActorCriticAgent.__init__(self, obs_spec, act_spec, sess_mgr=sess_mgr, n_envs=n_envs)

        self.current_subagent_index = 0

    def _run(self, env, n_steps):
        self.on_start()
        obs, *_ = env.reset()
        obs = [o.copy() for o in obs]
        for step in range(self.start_step, self.start_step + n_steps):

            received_message = env.listen()
            self.parse_message(received_message)

            action, value = self.get_action_and_value(obs)
            self.next_obs, reward, done = env.step(action)
            self.on_step(step, obs, action, reward, done, value)
            obs = [o.copy() for o in self.next_obs]
        env.stop()
        self.on_finish()

    # def get_action_and_value(self, obs):
        # return self.subagents[self.current_subagent_index].get_action(obs), None

    def select_subagent(self, message):
        """
        Test phase:
        selecting between a MoveToBeacon agent and a DefeatZerglings agent
        """
        if 'beacon' in message:
            self.current_subagent_index = 0
            return 0
        elif 'attack' in message:
            self.current_subagent_index = 1
            return 1
        else:
            print(LOGGING_MSG_HEADER + " invalid message, selecting default sub_module")
            self.current_subagent_index = 0
            return 0

    def parse_message(self, message):
        # TODO add in logic for parsing
        if message and 'roger' not in message: # filter the roger messages
            print(LOGGING_MSG_HEADER + " {} ".format(message))
            self.current_subagent_index = self.select_subagent(message)
            print("selected agent is : ", self.current_subagent_index)
            print("selected agent produces the action-value {}  - current subagent index", self.current_subagent_index)
            print()

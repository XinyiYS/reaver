import gin.tf
import tensorflow.compat.v1 as tf

from reaver.envs.base import Spec
from reaver.utils import StreamLogger
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType
from reaver.agents.base import ActorCriticAgent, DEFAULTS
from .a2c import AdvantageActorCriticAgent


@gin.configurable('HAIAgent')
class HumanAIInteractionAgent(AdvantageActorCriticAgent):
    """
    HAI: an extension of the existing A2C agent class to
    include human command interaction
    """

    def __init__(
        self,
        obs_spec: Spec,
        act_spec: Spec,
        subagents=[],  # need to specify a list of subagents: trained agents
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
        AdvantageActorCriticAgent().__init__(obs_spec, act_spec)

        self.subagents = subagents
        for subagent in self.subagents:
            subagent.__init__(obs_spec, act_spec)
            pass

        self.message_hub = []

    def _run(self, env, n_steps):
        self.on_start()
        obs, *_ = env.reset()
        obs = [o.copy() for o in obs]
        for step in range(self.start_step, self.start_step + n_steps):
            chat_message = self._get_chat_message(env)
            if chat_message:
                print(chat_message)

            action, value = self.get_action_and_value(obs)
            self.next_obs, reward, done = env.step(action)
            # self.next_obs, reward, done, chat_message = env.step_with_chat_message(action)
            self.on_step(step, obs, action, reward, done, value)
            obs = [o.copy() for o in self.next_obs]
        env.stop()
        self.on_finish()

    def _get_chat_message(env):
        reaver_sc2env = env.envs[0]._env
        try:
            chat_received = reaver_sc2env._get_chat_message()
        except Excetion as e:
            pass
        return

    def get_action_and_value(self, obs):
        if not self.message_hub:
            subagent_index = 0  # default use the first sub_module
        else:
            message = self.message_hub.pop()
            subagent_index = self._select_subagent(message)

        return self.subagents[subagent_index].get_action(obs), None

    def _select_subagent(self, message):
        """
        Test phase:
        selecting between a MoveToBeacon agent and a DefeatZerglings agent
        """
        if 'beacon' in message:
            return 0
        elif 'attack' in message:
            return 1
        else:
            print("invalid message, selecting default sub_module")
            return 0

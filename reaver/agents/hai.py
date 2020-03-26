import os
import gin.tf
import tensorflow.compat.v1 as tf

from reaver.envs.base import Spec
from reaver.utils import StreamLogger
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType
from reaver.agents.base import ActorCriticAgent, DEFAULTS
from .a2c import AdvantageActorCriticAgent


LOGGING_MSG_HEADER = "LOGGING FROM <reaver.reaver.agents.hai> "
MAIN_AGENT_INDEX = -1

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
        AdvantageActorCriticAgent.__init__(
            self, obs_spec, act_spec, sess_mgr=sess_mgr, n_envs=n_envs)

        # since we are using separate session managers
        # we do not have to modify the a2c class at all
        self.subagents_dir = 'subagents/'
        if self.check_subagents_dir():
            self.init_subagents(
                self.n_subagents * [model_fn],
                self.n_subagents * [obs_spec],
                self.n_subagents * [act_spec],
                self.n_subagents * [policy_cls],
                self.n_subagents,
                self.subagent_variable_scopes
            )

        self.selected_subagent_idx = MAIN_AGENT_INDEX

    def check_subagents_dir(self):
        if os.path.isdir(self.subagents_dir):
            self.subagent_variable_scopes = []
            # to record the individual subagent_dir
            self.subagent_dirs = {}
            for subagent_dir in os.listdir(self.subagents_dir):
                subagent_variable_scope = '_'.join(subagent_dir.split('_')[:2])
                self.subagent_variable_scopes.append(subagent_variable_scope)
                self.subagent_dirs[subagent_variable_scope] = os.path.join(
                    self.subagents_dir, subagent_dir)
            self.n_subagents = len(self.subagent_variable_scopes)
        else:
            self.n_subagents = 0
        print(LOGGING_MSG_HEADER +
              ": Found a total of {} subagents".format(self.n_subagents))
        return self.n_subagents != 0

    def init_subagents(self, model_fns, obs_specs, act_specs, policy_clses,
                       n_subagents=0, subagent_variable_scopes=[]):
        assert n_subagents == len(model_fns) == len(obs_specs) == len(policy_clses) == len(act_specs) == len(
            subagent_variable_scopes), "The number of subagents is not equal to the number of model_fns, or obs_specs, or act_specs"

        self.subagents = {}
        for model_fn, obs_spec, act_spec, policy_cls, subagent_variable_scope in zip(model_fns, obs_specs, act_specs, policy_clses, subagent_variable_scopes):
            subagent = Subagent()
            subagent_dir = self.subagent_dirs[subagent_variable_scope]

            print(LOGGING_MSG_HEADER + ': resetting tf graph')
            tf.reset_default_graph()
            subagent.sess_mgr = SessionManager(base_path=subagent_dir,
                                               training_enabled=False,
                                               model_variable_scope=subagent_variable_scope)
            subagent.sess = subagent.sess_mgr.sess
            subagent.variable_scope = subagent_variable_scope

            with subagent.sess.graph.as_default():
                with tf.name_scope(subagent.sess_mgr.main_tf_vs.original_name_scope):
                    subagent.model = model_fn(obs_spec, act_spec)
                    subagent.value = subagent.model.outputs[-1]
                    subagent.policy = policy_cls(act_spec, subagent.model.outputs[:-1])
                    print(LOGGING_MSG_HEADER+': ', subagent.variable_scope, 'model setup successful')

                    subagent.sess_mgr.restore_or_init()
                    print(LOGGING_MSG_HEADER+': ', subagent.variable_scope, 'model restore successful')

            self.subagents[subagent_variable_scope] = subagent

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

    """
    Superclass' implementations of get_action_and_value() and get_action()
    Need to override to execute the session-manager of the selected subagent
    """

    def get_action_and_value(self, obs):
        if self.selected_subagent_idx == MAIN_AGENT_INDEX:
            return self.sess_mgr.run([self.policy.sample, self.value], self.model.inputs, obs)
        else:
            print(LOGGING_MSG_HEADER + ": producing action from a loaded subagent")
            selected_subagent = self.subagents[self.selected_subagent_idx]
            return selected_subagent.sess_mgr.run([selected_subagent.policy.sample, selected_subagent.value], selected_subagent.model.inputs, obs)

    def get_action(self, obs):
        if self.selected_subagent_idx == MAIN_AGENT_INDEX:
            return self.sess_mgr.run(self.policy.sample, self.model.inputs, obs)
        else:
            print(LOGGING_MSG_HEADER + ": producing action from a loaded subagent")
            selected_subagent = self.subagents[self.selected_subagent_idx]
            return selected_subagent.sess_mgr.run(selected_subagent.policy.sample, selected_subagent.model.inputs, obs)

    # def get_action_and_value(self, obs):
    #     return self.sess_mgr.run([self.policy.sample, self.value], self.model.inputs, obs)

    # def get_action(self, obs):
    #     return self.sess_mgr.run(self.policy.sample, self.model.inputs, obs)

    def select_subagent(self, message):
        """
        Test phase:
        selecting between a MoveToBeacon agent and a DefeatZerglings agent
        """
        if 'beacon' in message:
            self.selected_subagent_idx = 'MoveToBeacon_a2c'
        elif 'attack' in message:
            self.selected_subagent_idx = 1
        else:
            print(LOGGING_MSG_HEADER +
                  ": invalid message, selecting default agent")
            self.selected_subagent_idx = MAIN_AGENT_INDEX

    def parse_message(self, message):
        # TODO add in logic for parsing
        if message and 'roger' not in message:  # filter the roger messages
            print(LOGGING_MSG_HEADER + ": {} ".format(message))
            print("Previous selected agent index is {}".format(self.selected_subagent_idx))
            self.select_subagent(message)
            print("swictched to this index {} : ".format(self.selected_subagent_idx))


class Subagent():
    def __init__(self):
        pass

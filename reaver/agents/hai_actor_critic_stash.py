import gin.tf
import tensorflow.compat.v1 as tf

from reaver.envs.base import Spec
from reaver.agents.base import MemoryAgent
from reaver.utils import Logger
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType
from reaver.agents.base import ActorCriticAgent, DEFAULTS


@gin.configurable('HAIACAgent')
class HAIActorCriticAgent(ActorCriticAgent, MemoryAgent):
    """
    Extends the actor_critic class to allow for subagents

    override the __init__(), get_action_and_value(), get_action() functions

    Needs to implement a loss_fn(), if not the extending class needs to implement it
    """

    def __init__(
        self,
        obs_spec: Spec,
        act_spec: Spec,
        # need to specify a list of subagents: tf_checkpoints
        subagents_checkpoints: list = [],
        n_subagents: int = 0,
        model_fn: ModelBuilder = None,
        policy_cls: PolicyType = None,
        sess_mgr: SessionManager = None,
        optimizer: tf.train.Optimizer = None,
        n_envs: int = 1,
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
        MemoryAgent.__init__(self, obs_spec, act_spec, traj_len, batch_sz)

        assert len(subagents_checkpoints) == n_subagents, "The number of checkpoints \
        is not equal to the number of subagents."
        # currently assume all subagents share the model structure completely

        if not sess_mgr:
            sess_mgr = SessionManager(
                subagent_checkpoints=subagents_checkpoints, n_subagents=n_subagents)

        if not optimizer:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=DEFAULTS['learning_rate'])

        self.sess_mgr = sess_mgr
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_rewards = clip_rewards
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages

        with tf.variable_scope('main_model'):
            self.model = model_fn(obs_spec, act_spec)

        self.subagent_models = self.init_subagent_models(
            model_fn, obs_spec, act_spec, n_subagents)

        self.value = self.model.outputs[-1]
        self.policy = policy_cls(act_spec, self.model.outputs[:-1])
        self.loss_op, self.loss_terms, self.loss_inputs = self.loss_fn()

        grads, vars = zip(*optimizer.compute_gradients(self.loss_op))
        self.grads_norm = tf.global_norm(grads)
        if clip_grads_norm > 0.:
            grads, _ = tf.clip_by_global_norm(
                grads, clip_grads_norm, self.grads_norm)
        self.train_op = optimizer.apply_gradients(
            zip(grads, vars), global_step=sess_mgr.global_step)
        self.minimize_ops = self.make_minimize_ops()

        sess_mgr.restore_or_init()
        print(self.model.inputs)
        exit()
        self.n_batches = sess_mgr.start_step
        self.start_step = sess_mgr.start_step * traj_len

        self.logger = Logger()
        self.message_hub = []

    def init_subagent_models(self, model_fns, obs_specs, act_specs, n_subagents=0):
        assert n_subagents == len(model_fns) == len(obs_specs) == act_specs, "The \
            number of subagents is not equal to the number of model_fns, or obs_specs, or act_specs"
        subagent_models = []
        for model_fn, obs_spec, act_spec, sub_agent_index in zip(model_fns, obs_specs, act_specs, range(n_subagents)):
            with tf.variable_scope('subagent_' + str(sub_agent_index)):
                subagent_models.append(model_fn(obs_spec, act_spec))
        return subagent_models

    # TODO implement the interaction with chat message
    def _run(self, env, n_steps):
        # @overrides the method in RunningAgent

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

    # TODO implement the selection logic
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

    def get_action_and_value(self, obs):
        # @overrides the method in RunningAgent

        # TODO need to choose the actions based on
        # the command and the subagent's actions
        return self.sess_mgr.run([self.policy.sample, self.value], self.model.inputs, obs)

    def get_action(self, obs):
        # @overrides the method in RunningAgent

        # TODO need to choose the actions based on
        # the command and the subagent's actions
        return self.sess_mgr.run(self.policy.sample, self.model.inputs, obs)

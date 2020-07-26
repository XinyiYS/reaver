import copy
import gin.tf
import numpy as np
import tensorflow.compat.v1 as tf
from abc import abstractmethod

from reaver.envs.base import Spec
from reaver.agents.base import MemoryAgent
from reaver.utils import Logger
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType

DEFAULTS = dict(
    model_fn=None,
    policy_cls=None,
    optimizer=None,
    learning_rate=0.0003,
    value_coef=0.5,
    entropy_coef=0.01,
    traj_len=16,
    batch_sz=16,
    discount=0.99,
    gae_lambda=0.95,
    clip_rewards=0.0,
    clip_grads_norm=0.0,
    normalize_returns=False,
    normalize_advantages=False,
    model_variable_scope=None,
)
LOGGING_MSG_HEADER = "LOGGING FROM <reaver.reaver.agents.base.actor_critic> "


@gin.configurable('ACAgent')
class ActorCriticAgent(MemoryAgent):
    """
    Abstract class, unifies deep actor critic functionality
    Handles on_step callbacks, either updating current batch
    or executing one training step if the batch is ready

    Extending classes only need to implement loss_fn method
    """

    def __init__(
        self,
        obs_spec: Spec,
        act_spec: Spec,
        model_variable_scope=DEFAULTS['model_variable_scope'],
        model_fn: ModelBuilder = None,
        policy_cls: PolicyType = None,
        sess_mgr: SessionManager = None,
        optimizer: tf.train.Optimizer = None,
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
        **kwargs,
    ):
        MemoryAgent.__init__(self, obs_spec, act_spec, traj_len, batch_sz)

        if not sess_mgr:
            sess_mgr = SessionManager()

        subenvs = kwargs['subenvs'] if 'subenvs' in kwargs else [None]

        if optimizer:
            optimizers = [copy.deepcopy(optimizer) for subenv in subenvs]
        else:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=DEFAULTS['learning_rate'])
            optimizers = [tf.train.AdamOptimizer(learning_rate=DEFAULTS['learning_rate']) for subenv in subenvs]

        self.sess_mgr = sess_mgr
        self.model_variable_scope = self.sess_mgr.model_variable_scope
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_rewards = clip_rewards
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages

        print(LOGGING_MSG_HEADER + " : the current model_variable_scope is", self.model_variable_scope)
        # implement the a2c to support multiple subagents
        # self.model = model_fn(obs_spec, act_spec)
        with sess_mgr.sess.graph.as_default():
            # note this is name_scope as opposed to variable_scope, important
            with tf.name_scope(self.sess_mgr.main_tf_vs.original_name_scope):
                
                if subenvs:
                    from collections import defaultdict
                    self.subenv_dict = defaultdict(list)
                    print("Attempting to create models for each individual subenvs: ", subenvs)

                    for i, subenv in enumerate(subenvs):
                        subenv_model = model_fn(obs_spec, act_spec)
                        self.subenv_dict['models'].append(subenv_model)

                        subenv_value = subenv_model.outputs[-1]
                        self.subenv_dict['values'].append(subenv_value)
                        subenv_policy = policy_cls(act_spec, subenv_model.outputs[:-1])
                        self.subenv_dict['policies'].append(subenv_policy)

                        subenv_loss_op, subenv_loss_terms, subenv_loss_inputs = self.loss_fn(policy=subenv_policy, value=subenv_value)
                        self.subenv_dict['loss_ops'].append(subenv_loss_op)
                        self.subenv_dict['loss_terms'].append(subenv_loss_terms)
                        self.subenv_dict['loss_inputs'].append(subenv_loss_inputs)

                        subenv_optimizer = optimizers[i]
                        grads, vars = zip(*subenv_optimizer.compute_gradients(subenv_loss_op))

                        subenv_grads_norm = tf.global_norm(grads)
                        self.subenv_dict['grads_norms'].append(subenv_grads_norm)
                        if clip_grads_norm > 0 :
                            grads, _ = tf.clip_by_global_norm(grads, clip_grads_norm, subenv_grads_norm)
                        self.subenv_dict['train_ops'].append(subenv_optimizer.apply_gradients(
                            zip(grads, vars), global_step=sess_mgr.global_step))
                        self.subenv_dict['minimize_ops'].append(self.make_minimize_ops( subenv_id=i) ) 
                    print("Successfully created models for each individual subenvs")
                else:

                    self.model = model_fn(obs_spec, act_spec)
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

        print(LOGGING_MSG_HEADER + " : main_model setup on sess and graph complete")
        sess_mgr.restore_or_init()
        print(LOGGING_MSG_HEADER + " : main_model weights restore/init complete")

        self.n_batches = sess_mgr.start_step
        self.start_step = sess_mgr.start_step * traj_len

        self.logger = Logger()

    def get_action_and_value(self, obs, subenv_id=None):
        if subenv_id is None:
            return self.sess_mgr.run([self.policy.sample, self.value], self.model.inputs, obs)
        else:
            return self.sess_mgr.run([ self.subenv_dict['policies'][subenv_id].sample, self.subenv_dict['values'][subenv_id]],  self.subenv_dict['models'][subenv_id].inputs, obs)


    def get_action(self, obs, subenv_id=None):
        if subenv_id is None:
            return self.sess_mgr.run(self.policy.sample, self.model.inputs, obs)
        else:
            return self.sess_mgr.run(self.subenv_dict['policies'][subenv_id].sample,  self.subenv_dict['models'][subenv_id].inputs, obs)

    def on_step(self, step, obs, action, reward, done, value=None, subenv_id=None):
        MemoryAgent.on_step(self, step, obs, action, reward, done, value)
        self.logger.on_step(step, reward, done)

        if not self.batch_ready():
            return

        if subenv_id is None:
            next_values = self.sess_mgr.run(
                self.value, self.model.inputs, self.last_obs)
        else:
            assert self.subenv_dict, "Missing subenv_dict implementation"
            next_values = self.sess_mgr.run(
                self.subenv_dict['values'][subenv_id], self.subenv_dict['models'][subenv_id].inputs, self.last_obs)

        adv, returns = self.compute_advantages_and_returns(next_values)
        loss_terms, grads_norm = self.minimize(adv, returns, subenv_id=subenv_id)

        self.sess_mgr.on_update(self.n_batches)
        logs = self.logger.on_update(self.n_batches, loss_terms,
                              grads_norm, returns, adv, next_values)
        return logs

    def minimize(self, advantages, returns, subenv_id=None):
        inputs = self.obs + self.acts + [advantages, returns]
        inputs = [a.reshape(-1, *a.shape[2:]) for a in inputs]

        if subenv_id is None:
            tf_inputs = self.model.inputs + self.policy.inputs + self.loss_inputs

            loss_terms, grads_norm, * \
                _ = self.sess_mgr.run(self.minimize_ops, tf_inputs, inputs)
        else:
            assert self.subenv_dict, "Missing subenv_dict implementation"
            tf_inputs = self.subenv_dict['models'][subenv_id].inputs + self.subenv_dict['policies'][subenv_id].inputs + self.subenv_dict['loss_inputs'][subenv_id]
            loss_terms, grads_norm, * \
                _ = self.sess_mgr.run(self.subenv_dict['minimize_ops'][subenv_id], tf_inputs, inputs)

        return loss_terms, grads_norm

    def compute_advantages_and_returns(self, bootstrap_value):
        """
        GAE can help with reducing variance of policy gradient estimates
        """
        if self.clip_rewards > 0.0:
            np.clip(self.rewards, -self.clip_rewards,
                    self.clip_rewards, out=self.rewards)

        rewards = self.rewards.copy()
        rewards[-1] += (1-self.dones[-1]) * self.discount * bootstrap_value

        masked_discounts = self.discount * (1-self.dones)

        returns = self.discounted_cumsum(rewards, masked_discounts)

        if self.gae_lambda > 0.:
            values = np.append(self.values, np.expand_dims(
                bootstrap_value, 0), axis=0)
            # d_t = r_t + g * V(s_{t+1}) - V(s_t)
            deltas = self.rewards + masked_discounts * values[1:] - values[:-1]
            adv = self.discounted_cumsum(
                deltas, self.gae_lambda * masked_discounts)
        else:
            adv = returns - self.values

        if self.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        if self.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)

        return adv, returns

    def on_start(self):
        self.logger.on_start()

    def on_finish(self):
        self.logger.on_finish()

    def reset(self):
        """
        Introduced for HRL with multiple subenvs trained in sequence

        So need to reset some auxiliary logging book-keeping information
        """

        MemoryAgent.__init__(self, obs_spec=self.obs_spec, act_spec=self.act_spec, traj_len=self.traj_len, batch_sz=self.batch_sz)
        self.logger.reset()

    def make_minimize_ops(self, subenv_id=None):

        if subenv_id is None:
            ops = [self.loss_terms, self.grads_norm]
            if self.sess_mgr.training_enabled:
                ops.append(self.train_op)
            # appending extra model update ops (e.g. running stats)
            # note: this will most likely break if model.compile() is used
            ops.extend(self.model.get_updates_for(None))
            return ops

        else:
            assert self.subenv_dict, "self.subenv_dict is None or empty"
            loss_terms = self.subenv_dict['loss_terms'][subenv_id]
            grads_norm = self.subenv_dict['grads_norms'][subenv_id]
            ops = [loss_terms, grads_norm]
            if self.sess_mgr.training_enabled:
                ops.append(self.subenv_dict['train_ops'][subenv_id])
            return ops

    @staticmethod
    def discounted_cumsum(x, discount):
        y = np.zeros_like(x)
        y[-1] = x[-1]
        for t in range(x.shape[0]-2, -1, -1):
            y[t] = x[t] + discount[t] * y[t+1]
        return y

    @abstractmethod
    def loss_fn(self): ...

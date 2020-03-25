import gin.tf
import tensorflow.compat.v1 as tf

from reaver.envs.base import Spec
from reaver.utils import StreamLogger
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType
from reaver.agents.base import SyncRunningAgent, ActorCriticAgent, DEFAULTS
from .a2c import HAIActorCriticAgent


@gin.configurable('HAIA2CAgent')
class HAIA2CAgent(SyncRunningAgent, HAIActorCriticAgent):
    """
    HAI: an extension of the existing A2C agent class to
    include human command interaction
    hence it is a HAI-A2C class
    """

    def __init__(
        self,
        obs_spec: Spec,
        act_spec: Spec,
        # need to specify a list of subagents: trained agents
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
        SyncRunningAgent.__init__(self, n_envs)

        assert len(subagents_checkpoints) == n_subagents, "The number of checkpoints \
        is not equal to the number of subagents."
        # currently assume all subagents share the model structure completely

        # the main model will be defined in parent class
        # call the init_subagent_models BEFORE the init of sess_mgr
        # to include the subagent's variables in the tf_graph
        self.subagent_models = self.init_subagent_models(model_fn, obs_spec, act_spec, 1)

        # sess_mgr.restore_or_init(n_subagents=1)
        if not sess_mgr:
            sess_mgr = SessionManager(subagent_checkpoints=subagents_checkpoints, n_subagents=n_subagents)

        HAIActorCriticAgent.__init__(obs_spec, act_spec, sess_mgr=sess_mgr)

        self.message_hub = []

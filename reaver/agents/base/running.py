import copy
from . import Agent
from reaver.envs.base import Env, MultiProcEnv


class RunningAgent(Agent):
    """
    Generic abstract class, defines API for interacting with an environment
    """

    def __init__(self):
        self.next_obs = None
        self.start_step = 0

    def run(self, env: Env, n_steps=1000000):
        env = self.wrap_env(env)
        env.start()
        try:
            self._run(env, n_steps)
        except KeyboardInterrupt:
            env.stop()
            self.on_finish()

    def _run(self, env, n_steps):
        self.on_start()
        obs, *_ = env.reset()
        obs = [o.copy() for o in obs]
        print("Printed from <RunningAgent> class: a total of {} parallel env(s)".format(str(len(env.envs))))
        for step in range(self.start_step, self.start_step + n_steps):
            print("getting the action-value prediction")
            action, value = self.get_action_and_value(obs)
            print("action-value prediction successful")
            self.next_obs, reward, done = env.step(action)
            print("environment step successful, obs-reward-done triplet is received")
            self.on_step(step, obs, action, reward, done, value)
            print("training on_step successful")
            obs = [o.copy() for o in self.next_obs]
            print("element-wise copying of observation successful")
        env.stop()
        self.on_finish()

    def get_action_and_value(self, obs):
        return self.get_action(obs), None

    def on_start(self): ...

    def on_step(self, step, obs, action, reward, done, value=None): ...

    def on_finish(self): ...

    def wrap_env(self, env: Env) -> Env:
        return env


def _get_chat_message(env):
    # Note the agent class does not store a copy of the env as an attribute
    # hence need to be passed in as argument
    reaver_sc2env = env.envs[0]._env
    try:
        message_received = reaver_sc2env._get_chat_message()
    except Exception as e:
        print("error in {}".format(str(e)))
        message_received = None

    return message_received


class SyncRunningAgent(RunningAgent):
    """
    Abstract class that handles synchronous multiprocessing via MultiProcEnv helper
    Not meant to be used directly, extending classes automatically get the feature
    """

    def __init__(self, n_envs):
        RunningAgent.__init__(self)
        self.n_envs = n_envs

    def wrap_env(self, env: Env) -> Env:
        render, env.render = env.render, False
        envs = [env] + [copy.deepcopy(env) for _ in range(self.n_envs-1)]
        env.render = render

        return MultiProcEnv(envs)


class RunningAgentWithHumanCommand(RunningAgent):
    """
    Generic abstract class, defines API for interacting with an environment
    This is used specifically for including:
        - human command
        - to switch to actions and value provided by other trained agents
            and/or scripts
    Currently does not support Synchronizing among multiple learning agents
    such as in a2c

    The chat interaction needs to be defined here, because it is where the
    agent and the environment actually interacts, i.e. where the agent can
    read and do something about the chat messages.
    """

    def __init__(self, subagents):
        RunningAgent.__init__()
        self.subagents = subagents  # represent trained agents or scripts
        self.message_hub = []  # to manage the command messages
        self.human_command_available = True

    def _run(self, env, n_steps):
        self.on_start()
        obs, *_ = env.reset()
        obs = [o.copy() for o in obs]
        for step in range(self.start_step, self.start_step + n_steps):
            action, value = self.get_action_and_value(obs)
            self.next_obs, reward, done = env.step(action)
            self.on_step(step, obs, action, reward, done, value)
            obs = [o.copy() for o in self.next_obs]
        env.stop()
        self.on_finish()

    def get_action_and_value(self, obs):
        if not self.message_hub:
            subagent_index = 0  # default use the first subagent
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
            print("invalid message, selecting default subagent")
            return 0

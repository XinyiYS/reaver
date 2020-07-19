import gin.tf
import copy
from . import Agent
from reaver.envs.base import Env, MultiProcEnv
from reaver.envs.sc2 import SC2Env

LOGGING_MSG_HEADER = "LOGGING FROM <reaver.reaver.agents.base.running> "
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
            self.on_start()
            self._run(env, n_steps)
        except KeyboardInterrupt:
            env.stop()
        self.on_finish()

    '''
    original code
        def _run(self, env, n_steps):
            # self.on_start()
            obs, *_ = env.reset()
            obs = [o.copy() for o in obs]
            print(LOGGING_MSG_HEADER + " : running {} parallel env(s)".format(str(len(env.envs))))
            for step in range(self.start_step, self.start_step + n_steps):
                action, value = self.get_action_and_value(obs)
                self.next_obs, reward, done = env.step(action)
                self.on_step(step, obs, action, reward, done, value)
                obs = [o.copy() for o in self.next_obs]
            env.stop()
            # self.on_finish()
    '''

    def _run(self, env, n_steps, terminating_threshold=None):
        obs, *_ = env.reset()
        obs = [o.copy() for o in obs]
        print(LOGGING_MSG_HEADER + " : running {} parallel env(s)".format(str(len(env.envs))))
        for step in range(self.start_step, self.start_step + n_steps):
            action, value = self.get_action_and_value(obs)
            self.next_obs, reward, done = env.step(action)
            # self.on_step(step, obs, action, reward, done, value)
            logs = self.on_step(step, obs, action, reward, done, value)
            obs = [o.copy() for o in self.next_obs]

            #check terminating_conditions
            # simple average
            if logs and terminating_threshold and logs['ep_rews_mean'] >= terminating_threshold:
                print("Successfully reached the stopping reward threshold {}".format(terminating_threshold))
                break

        env.stop()

    def get_action_and_value(self, obs):
        return self.get_action(obs), None

    def on_start(self): ...

    def on_step(self, step, obs, action, reward, done, value=None): ...

    def on_finish(self): ...

    def wrap_env(self, env: Env) -> Env:
        return env


class SyncRunningAgent(RunningAgent):
    """
    Abstract class that handles synchronous multiprocessing via MultiProcEnv helper
    Not meant to be used directly, extending classes automatically get the feature
    """

    def __init__(self, n_envs, args=None):
        RunningAgent.__init__(self)
        self.n_envs = n_envs
        self.args = args


    def run(self, env: Env, n_steps=1000000):

        if self.args.test or (not self.args.HRL) or (env.id not in SUB_ENV_DICT):
            # either testing or training without HRL at all
            # or the env selected does not have the subenvs
            env = self.wrap_env(env)
            env.start()
            try:
                self.on_start()
                self._run(env, n_steps)
            except KeyboardInterrupt:
                env.stop()

            self.on_finish()

        else:

            assert self.args.HRL in ['human', 'systematic', 'random']
            subenvs = SUB_ENV_DICT[env.id]
            print('Subenvs are: ', subenvs)
            subenv_steps = [n_steps//len(subenvs) for subenv in subenvs]
            thresholds = [None for subenv in subenvs ]

            if self.args.HRL == 'human':
                threshold = HRL_thredhold(env.id)
            elif self.args.HRL == 'random':
                import numpy as np
                np.random.seed(1234)
                indices = sorted(np.random.choice(n_steps, len(subenvs)-1, replace=False))
                indices = [0] + sorted(np.random.choice(n_steps, I-1, replace=False)) + [n_steps]
                subenv_steps = np.ediff1d(indices)
            elif self.args.HRL == 'systematic':
                pass
                #  subenv_steps already defined and initializied

            for subenv, subenv_step, threshold, in zip(subenvs, subenv_steps, threshold):
                env = SC2Env(subenv, env.render, max_ep_len=env.max_ep_len)
                print("Creating and Running subenv : {} with maximum {} steps, and reward threshold is {}.".format(env.id, subenv_step, threshold))
                env = self.wrap_env(env)
                env.start()
                try:
                    self.on_start()
                    self._run(env, subenv_step, threshold)
                except KeyboardInterrupt:
                    env.stop()
                    break
            self.on_finish()

        """
        terminating_conditions:
            simple thresholds for episodic average
            thredholds for running average
            thresholds for amortized average

        """

    def wrap_env(self, env: Env) -> Env:
          render, env.render = env.render, False
          envs = [env] + [copy.deepcopy(env) for _ in range(self.n_envs-1)]
          env.render = render

          return MultiProcEnv(envs)

'''
wrap_env and get_subenvs for concurrent and parallel subenvs
    # if not self.args.test and env.id in SUB_ENV_DICT:
    #     subenvs = SUB_ENV_DICT[env.id]
    #     print('Subenvs are: ', subenvs, "Running {} steps for each subenv".format(n_steps//len(subenvs)))
    #     for subenv in subenvs:
    #         env = SC2Env(subenv, env.render, max_ep_len=env.max_ep_len)
    #         print("Creating and Running {}".format(env.id))
    #         env = self.wrap_env(env)
    #         env.start()
    #         try:
    #             self.on_start()
    #             self._run(env, n_steps//len(subenvs))
    #         except KeyboardInterrupt:
    #             env.stop()
    #             # self.on_finish()
    #             break

    # else:
    #     env = self.wrap_env(env)
    #     env.start()
    #     try:
    #         self._run(env, n_steps)
    #     except KeyboardInterrupt:
    #         env.stop()

    # self.on_finish()

    def wrap_env(self, env: Env) -> Env:
        print("{} from SyncRunningAgent: wrapping {}:{} to create {} environments.".format(LOGGING_MSG_HEADER, env, env.id, self.n_envs))
        envs = self.get_subenvs(env)
        return MultiProcEnv(envs)
    def get_subenvs(self, env):
        assert isinstance(env, SC2Env), "Not implemented for non-SC2Envs"

        if self.args.test or env.id not in SUB_ENV_DICT:

            render, env.render = env.render, False
            envs = [env] + [copy.deepcopy(env) for _ in range(self.n_envs-1)]
            env.render = render

        else:
            print("{} from SyncRunningAgent: getting sub-envs for {}".format(LOGGING_MSG_HEADER, env.id))
            sub_envs = SUB_ENV_DICT[env.id]
            print("{} from SyncRunningAgent: {} sub-envs are {}".format(LOGGING_MSG_HEADER, len(sub_envs), sub_envs))

            envs = [SC2Env(subenv, env.render, max_ep_len=env.max_ep_len) for subenv in SUB_ENV_DICT[env.id] ]

            if self.n_envs % 8 == 0:
                duplicate_copy = self.n_envs // 4 # 4 because each has 4 sub-envs
                envs_ = []
                for env in envs:
                    for _ in range(duplicate_copy):
                        envs_.append(copy.deepcopy(env))
                envs = envs_
                del envs_
        return envs

'''
@gin.configurable('HRL_threshold')
def HRL_thredhold(envId, BM, CMAG):
    assert envId in ['BuildMarines', 'CollectMineralsAndGas']
    if envId == 'BuildMarines':
        return BM
    else:
        return CMAG


SUB_ENV_DICT = {"BuildMarines": 
                                [     
                                "BuildSupplyDepots",
                                "BuildBarracks",
                                "BuildMarinesWithBarracks",       
                                "BuildMarines_depotbarracks",
                                "BuildMarines",
                                ],
                "CollectMineralsAndGas":
                                [
                                "CollectMineralsAndGas",  # 420s
                                "BuildRefinery",
                                "CollectGasWithRefineries",
                                "BuildRefineryAndCollectGas",
                                "CollectMineralsAndGas",  # 420s
                                ],
                }
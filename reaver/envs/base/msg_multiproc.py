import numpy as np
from multiprocessing import Pipe, Process
from . import Env

# LISTEN only needed for Windows, potentially MacOS, since SC2 on linux is headless
START, STEP, LISTEN, RESET, STOP, DONE = range(6)


class MsgProcEnv(Env):
    def __init__(self, env):
        super().__init__(env.id)
        self._env = env
        self.conn = self.w_conn = self.proc = None

    def start(self):
        self.conn, self.w_conn = Pipe()
        self.proc = Process(target=self._run)
        self.proc.start()
        self.conn.send((START, None))

    def step(self, act):
        self.conn.send((STEP, act))

    def listen(self):
        self.conn.send((LISTEN, None))

    def reset(self):
        self.conn.send((RESET, None))

    def stop(self):
        self.conn.send((STOP, None))

    def wait(self):
        return self.conn.recv()

    def obs_spec(self):
        return self._env.obs_spec()

    def act_spec(self):
        return self._env.act_spec()

    def _run(self):
        while True:
            msg, data = self.w_conn.recv()
            if msg == START:
                self._env.start()
                self.w_conn.send(DONE)
            elif msg == STEP:
                obs, rew, done = self._env.step(data)
                self.w_conn.send((obs, rew, done))
            elif msg == LISTEN:
                received_msg = self._env.listen_to_chat_channel()
                self.w_conn.send((received_msg))
            elif msg == RESET:
                obs = self._env.reset()
                self.w_conn.send((obs, -1, -1))
            elif msg == STOP:
                self._env.stop()
                self.w_conn.close()
                break


class MsgMultiProcEnv(Env):
    """
    Parallel environments via multiprocessing + pipes
    """

    def __init__(self, envs):
        super().__init__(envs[0].id)
        self.envs = [MsgProcEnv(env) for env in envs]

    def start(self):
        for env in self.envs:
            env.start()
        self.wait()

    def step(self, actions):
        for idx, env in enumerate(self.envs):
            env.step([a[idx] for a in actions])
        return self._observe()

    def listen(self):
        for idx, env in enumerate(self.envs):
            env.listen()
        # only collect the non-empty messages
        received_message = [message for message in self.wait() if message]
        # TODO add in a better logic for filtering useless message
        # only return the latest
        if len(received_message) > 0:
            return received_message[0]

    def reset(self):
        for e in self.envs:
            e.reset()
        return self._observe()

    def _observe(self):
        obs, reward, done = zip(*self.wait())
        # n_envs x n_spaces -> n_spaces x n_envs
        obs = list(map(np.array, zip(*obs)))

        return obs, np.array(reward), np.array(done)

    def stop(self):
        for e in self.envs:
            e.stop()
        for e in self.envs:
            e.proc.join()

    def wait(self):
        return [e.wait() for e in self.envs]

    def obs_spec(self):
        return self.envs[0].obs_spec()

    def act_spec(self):
        return self.envs[0].act_spec()

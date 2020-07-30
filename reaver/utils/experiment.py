import os
import gin
import json
from datetime import datetime as dt


class Experiment:
    def __init__(self, results_dir, env_name, agent_name, name=None, restore=False, restore_mix=False, args=None):
        if not name:
            if restore:
                experiments = [e for e in os.listdir(results_dir) if env_name in e and agent_name in e]
                assert len(experiments) > 0, 'No experiment to restore'
                name = max(experiments, key=lambda p: os.path.getmtime(results_dir+'/'+p))
                name = '_'.join(name.split('_')[2:])

            else:
                name = dt.now().strftime("%y-%m-%d_%H-%M-%S")
        else:
            if restore_mix:
                assert name is not None, "Must give a experiment name in order to <restore_mix>."
                experiments = [e for e in os.listdir(results_dir) if name in e and agent_name in e]
                assert len(experiments) > 0, 'No experiment to restore'
                name = dt.now().strftime("%y-%m-%d_%H-%M-%S")


        self.name = name
        self.restore = restore
        self.env_name = env_name
        self.agent_name = agent_name
        self.results_dir = results_dir
        self.args = args

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + '/summaries', exist_ok=True)
        os.makedirs(self.results_dir + '/summaries', exist_ok=True)
        if not os.path.exists(self.summaries_path):
            os.symlink('../%s/summaries' % self.full_name, self.summaries_path)

    @property
    def full_name(self):
        if self.args.HRL and "HRL-{}".format(self.args.HRL) not in self.name:
            return '%s_%s_HRL-%s_%s' % (self.env_name, self.agent_name, self.args.HRL, self.name)
        else:
            return '%s_%s_%s' % (self.env_name, self.agent_name, self.name)


    @property
    def path(self):
        return '%s/%s' % (self.results_dir, self.full_name)

    @property
    def config_path(self):
        return '%s/%s' % (self.path, 'config.gin')

    @property
    def log_path(self):
        return '%s/%s' % (self.path, 'train.log')

    @property
    def checkpoints_path(self):
        return self.path + '/checkpoints'

    @property
    def summaries_path(self):
        return '%s/summaries/%s' % (self.results_dir, self.full_name)
    
    @property
    def experiment_config_path(self):
        return '%s/%s' % (self.path, 'experiment_config')
    
    def save_gin_config(self):
        with open(self.config_path, 'w') as cfg_file:
            cfg_file.write(gin.operative_config_str())
    
    def save_experiment_config(self):
        ex_config_dict = {"name": self.name, "restore": self.restore,
                            "env": self.env_name, "n_envs": self.args.n_envs,
                            "agent": self.agent_name,
                            "results_dir": self.results_dir, 
                            "time": dt.now().strftime("%y-%m-%d_%H-%M-%S"),
                            "HRL": self.args.HRL,
                            "max_ep_len": self.args.max_ep_len,
                            }
        with open(self.experiment_config_path, 'a') as ex_cfg_file:                
            ex_cfg_file.write(json.dumps(ex_config_dict))

    def save_model_summary(self, model, **kwargs):
        if model:
            with open(self.path + '/' + 'model_summary.txt', 'w') as fl:
                model.summary(print_fn=lambda line: print(line, file=fl))
        else:
            assert 'models' in kwargs and 'subenvs' in kwargs, "Missing models and subenvs"
            models = kwargs['models']
            subenvs = kwargs['subenvs']
            for model, subenv  in zip(models, subenvs):
                with open(self.path + '/' + subenv + '_model_summary.txt', 'w') as fl:
                    model.summary(print_fn=lambda line: print(line, file=fl))

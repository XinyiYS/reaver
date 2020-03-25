import os
import gin
import tensorflow.compat.v1 as tf
gin.external_configurable(tf.train.AdamOptimizer, module='tf.train')
gin.external_configurable(tf.train.RMSPropOptimizer, module='tf.train')
gin.external_configurable(tf.train.get_global_step, module='tf.train')
gin.external_configurable(tf.train.piecewise_constant, module='tf.train')
gin.external_configurable(tf.train.polynomial_decay, module='tf.train')
gin.external_configurable(tf.initializers.orthogonal,
                          'tf.initializers.orthogonal')


class SessionManager:
    def __init__(self, sess=None, base_path='results/', checkpoint_freq=100,
                 training_enabled=True, model_variable_scope=None,
                 subagents_dir='subagents/'):

        if not sess:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tf.keras.backend.set_session(sess)

        self.sess = sess
        self.saver = None
        self.base_path = base_path
        self.checkpoint_freq = checkpoint_freq
        self.training_enabled = training_enabled
        self.global_step = tf.train.get_or_create_global_step()
        self.summary_writer = tf.summary.FileWriter(self.summaries_path)

        self.model_variable_scope = model_variable_scope
        self.subagents_dir = subagents_dir

        self.get_subagent_variable_scopes()

    def restore_or_init(self):
        # main_model saver
        if not self.model_variable_scope:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_variable_scope))

        checkpoint = tf.train.latest_checkpoint(self.checkpoints_path)
        if checkpoint:
            self.saver.restore(self.sess, checkpoint)

            if self.training_enabled:
                # merge with previous summary session
                self.summary_writer.add_session_log(
                    tf.SessionLog(status=tf.SessionLog.START), self.sess.run(self.global_step))
        else:
            self.sess.run(tf.global_variables_initializer())

        if self.n_subagents != 0:
            self.restore_subagents()

        # this call locks the computational graph into read-only state,
        # as a safety measure against memory leaks caused by mistakingly adding new ops to it
        self.sess.graph.finalize()

    def load_subagents_from_checkpoints(self):
        # key : subagent_variable_scope
        # value : loaded checkpoint
        self.subagents = {}
        for subagent_dir in os.listdir(self.subagents_dir):
            subagent_variable_scope = '_'.join(subagent_dir.split('_')[:2]) 
            subagent_checkpoints_dir = os.path.join(self.subagents_dir, subagent_dir, 'checkpoints')
            self.subagents[subagent_variable_scope] = tf.train.latest_checkpoint(subagent_checkpoints_dir)
            # e.g. subagent_checkpoints_dir = "results/BuildMarinesWithBarracks_a2c_20-03-23_11-05-25/checkpoints"
        print("LOGGING from <utils.tensorflow> : loaded subagent checkpoints are ", self.subagents)

    def restore_subagents(self):
        """
        Initializes self.subagent_savers
        Restores the subagent models
        """
        # need to give proper names to the subagent models
        # for tf.variable_scope loading and saving
        self.load_subagents_from_checkpoints()
        self.subagent_savers = []
        for subagent_variable_scope, subagent_checkpoint in self.subagents.items():
            subagent_saver = tf.train.Saver(tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=subagent_variable_scope))

            subagent_saver.restore(self.sess, subagent_checkpoint)  # add the models of the subagents to the same session and tf_graph

            self.subagent_savers.append(subagent_saver)

    def run(self, tf_op, tf_inputs, inputs):
        return self.sess.run(tf_op, feed_dict=dict(zip(tf_inputs, inputs)))

    def on_update(self, step):
        if not self.checkpoint_freq or not self.training_enabled or step % self.checkpoint_freq:
            return

        self.saver.save(self.sess, self.checkpoints_path +
                        '/ckpt', global_step=step)

    def add_summaries(self, tags, values, prefix='', step=None):
        for tag, value in zip(tags, values):
            self.add_summary(tag, value, prefix, step)

    def add_summary(self, tag, value, prefix='', step=None):
        if not self.training_enabled:
            return
        summary = self.create_summary(prefix + '/' + tag, value)
        self.summary_writer.add_summary(summary, global_step=step)

    def get_subagent_variable_scopes(self):
        if os.path.isdir(self.subagents_dir):
            self.subagent_variable_scopes = ['_'.join(subagent_dir.split('_')[:2]) for subagent_dir in os.listdir(self.subagents_dir)]
            self.n_subagents = len(self.subagent_variable_scopes)
        else:
            self.n_subagents = 0
        print("LOGGING from <utils.tensorflow> Found a total of {} subagents".format(self.n_subagents))


    @staticmethod
    def create_summary(tag, value):
        return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])

    @property
    def start_step(self):
        if self.training_enabled:
            return self.global_step.eval(session=self.sess)
        return 0

    @property
    def summaries_path(self):
        return self.base_path + '/summaries'

    @property
    def checkpoints_path(self):
        return self.base_path + '/checkpoints'

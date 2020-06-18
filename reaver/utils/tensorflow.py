import os
import gin
import tensorflow.compat.v1 as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
gin.external_configurable(tf.train.AdamOptimizer, module='tf.train')
gin.external_configurable(tf.train.RMSPropOptimizer, module='tf.train')
gin.external_configurable(tf.train.get_global_step, module='tf.train')
gin.external_configurable(tf.train.piecewise_constant, module='tf.train')
gin.external_configurable(tf.train.polynomial_decay, module='tf.train')
gin.external_configurable(tf.initializers.orthogonal,
                          'tf.initializers.orthogonal')

LOGGING_MSG_HEADER = "LOGGING FROM <reaver.reaver.utils.tensorflow> "


class SessionManager:
    def __init__(self, sess=None, base_path='results/', checkpoint_freq=100,
                 training_enabled=True, model_variable_scope=None, restore_mix=False, env_name=None):
        if not sess:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # tf.keras.backend.set_session(sess)

        self.sess = sess
        self.saver = None
        self.base_path = base_path
        self.checkpoint_freq = checkpoint_freq
        self.training_enabled = training_enabled

        assert not restore_mix or env_name is not None, "Must specify which previously trained env to restore."
        self.restore_mix = restore_mix
        self.env_name = env_name

        with self.sess.graph.as_default():
            with tf.variable_scope(model_variable_scope) as main_tf_vs:
                self.global_step = tf.train.get_or_create_global_step()
                self.summary_writer = tf.summary.FileWriter(self.summaries_path)

        self.main_tf_vs = main_tf_vs  # a cleaner way to pass variable_scope
        self.model_variable_scope = model_variable_scope

    def restore_or_init(self):
        # main_model saver
        if not self.model_variable_scope:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_variable_scope))

        checkpoints = self.find_checkpoints(name=self.env_name if self.restore_mix else None)
        if checkpoints:
            for checkpoint in checkpoints:
                try:
                    self.saver.restore(self.sess, checkpoint)
                    print("{}: Restoring from a previous checkpoint: <{}>".format(LOGGING_MSG_HEADER, checkpoint))

                    new_summaries_path = os.path.join(  * (str(checkpoint).split("/", 2)[:2] + ['summaries']) )
                    self.summary_writer = tf.summary.FileWriter(new_summaries_path)
                    print("{}: Restoring summaries from previous: <{}>".format(LOGGING_MSG_HEADER, new_summaries_path))
                    if self.training_enabled:
                        # merge with previous summary session
                        self.summary_writer.add_session_log(
                            tf.SessionLog(status=tf.SessionLog.START), self.sess.run(self.global_step))
                    
                    break # if successful, break the loop
                except Exception as e:
                    print("Error {} when loading checkpoint {}".format(str(e), checkpoint))
        else:
            self.sess.run(tf.global_variables_initializer())

        print(LOGGING_MSG_HEADER + ": model for {} loaded and saver restored.".format(self.model_variable_scope))
        all_vars = tf.global_variables()
        # [print(var) for var in all_vars]
        main_model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_variable_scope)
        print("{} global variables ".format(len(all_vars)))
        print("{} variables under {} ".format(len(main_model_var_list), self.model_variable_scope))

        if self.restore_mix:
            """
            Renaming the restored variable scopes from the previous experiment (which is different), to this
            environment configuration.
            """
            pass


        # this call locks the computational graph into read-only state,
        # as a safety measure against memory leaks caused by mistakingly adding new ops to it
        self.sess.graph.finalize()

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
    
    def find_checkpoints(self, results_dir='results', name=None):
        """
        Given the current configuration, attempt to find all the checkpoints available.

        For the given name, list all the available checkpoints with the most recent in front.
        """
        name = name or self.base_path.split('/', 1)[1].split('_')[0]
        name = name + '_' # append an _ in the end to distinguish between maps with same prefix
        print("{}: env name is <{}>".format(LOGGING_MSG_HEADER,  name))
        print("{}: checking dir in this format <{}> for checkpoints".format(LOGGING_MSG_HEADER, os.path.join(results_dir, name)))

        # sorted according to descending dates, i.e. newest in front
        return sorted([tf.train.latest_checkpoint(os.path.join(results_dir, directory, 'checkpoints')) for directory in os.listdir(results_dir) if directory.startswith(name) and tf.train.latest_checkpoint(os.path.join(results_dir, directory, 'checkpoints'))], reverse=True)


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

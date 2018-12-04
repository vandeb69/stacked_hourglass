import os
import tensorflow as tf


class Logger:
    def __init__(self, config):
        self.log_dir = os.path.join(config.log_dir, config.name)
        self.scalars = config.log_scalars
        self.imgs = config.log_imgs
        self.hists = config.log_hists

        self.placeholders = {}
        self.ops = {}

        for scalar in self.scalars:
            self.placeholders[scalar] = tf.placeholder('float32', None, name=scalar)
            self.ops[scalar] = tf.summary.scalar(scalar, self.placeholders[scalar])
        for img in self.imgs:
            self.placeholders[img] = tf.placeholder('float32', None, name=img)
            self.ops[img] = tf.summary.image(img, self.placeholders[img], max_outputs=1)
        for hist in self.hists:
            self.placeholders[hist] = tf.placeholder('float32', None, name=hist)
            self.ops[hist] = tf.summary.histogram(hist, self.placeholders[hist])

        self.logger = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

    def log(self, sess, step, what):
        logs = sess.run([self.ops[tag] for tag in what.keys()],
                        {self.placeholders[tag]: value for tag, value in what.items()})
        for l in logs:
            self.logger.add_summary(l, step)

    def stop(self):
        self.logger.flush()

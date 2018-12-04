from utils.img import inp_outp_img
import numpy as np


class StackedHourglassTrainer:
    def __init__(self, config, sess, model, logger, loader):
        self.sess = sess
        self.loader = loader
        self.model = model
        self.logger = logger

        self.n_epochs = config.n_epochs
        self.batch_size = config.batch_size

        self.sess.run(self.model.init)
        self.model.load(self.sess)

        self.valid_generator = self.loader.generator(set='valid', batch_size=1)

    def train(self):
        # iterate over epochs
        while self.model.epoch.eval(self.sess) < self.n_epochs:
            print("Epoch {}/{}".format(self.model.epoch.eval(self.sess)+1, self.n_epochs))
            generator = self.loader.generator(set='train', batch_size=self.batch_size)

            batch = 0
            # iterate over mini-batches
            for img, gtmap, lengths in generator:
                inputs = {self.model.img: img,
                          self.model.gtmap: gtmap,
                          self.model.avg_lengths: lengths,
                          self.model.is_training: True}

                _, loss, prob_output = self.sess.run([self.model.train_op,
                                                 self.model.loss,
                                                 self.model.prob_output], feed_dict=inputs)

                print("\tBatch {} -- loss: {:.4f}".format(batch+1, loss))

                # log loss, img+gtmap, img+output
                log = {'training/loss': loss,
                       "training/img/earbases": inp_outp_img(img, gtmap, prob_output, 0, self.model.n_stacks-1, 0),
                       "training/img/eartips": inp_outp_img(img, gtmap, prob_output, 0, self.model.n_stacks-1, 1),
                       "training/img/spikelets": inp_outp_img(img, gtmap, prob_output, 0, self.model.n_stacks-1, 2),
                       "training/hist/earbases": prob_output[0, self.model.n_stacks-1, :, :, 0],
                       "training/hist/eartips": prob_output[0, self.model.n_stacks - 1, :, :, 1],
                       "training/hist/spikelets": prob_output[0, self.model.n_stacks - 1, :, :, 2]}
                self.logger.log(self.sess, step=self.model.step.eval(self.sess), what=log)

                batch += 1
                self.sess.run(self.model.increment_step)

            # log loss, img+gtmap, img+output for validation set input
            try:
                self.valid_img, self.valid_gtmap, self.valid_lengths = next(self.valid_generator)
            except StopIteration:
                self.valid_generator = self.loader.generator(set='valid', batch_size=1)
                self.valid_img, self.valid_gtmap, self.valid_lengths = next(self.valid_generator)

            valid_inputs = {self.model.img: self.valid_img,
                            self.model.gtmap: self.valid_gtmap,
                            self.model.avg_lengths: self.valid_lengths,
                            self.model.is_training: True}

            loss_val, prob_output_val = self.sess.run([self.model.loss, self.model.prob_output], feed_dict=valid_inputs)
            print('\tEpoch {}/{} -- validation loss: {:.4f}'.format(self.model.epoch.eval(self.sess)+1,
                                                                    self.n_epochs, loss_val))

            valid_log = {'validation/loss': loss_val,
                         "validation/img/earbases": inp_outp_img(self.valid_img, self.valid_gtmap, prob_output_val,
                                                                 0, self.model.n_stacks-1, 0),
                         "validation/img/eartips": inp_outp_img(self.valid_img, self.valid_gtmap, prob_output_val,
                                                                0, self.model.n_stacks-1, 1),
                         "validation/img/spikelets": inp_outp_img(self.valid_img, self.valid_gtmap, prob_output_val,
                                                                  0, self.model.n_stacks-1, 2),
                         "validation/hist/earbases": prob_output_val[0, self.model.n_stacks-1, :, :, 0],
                         "validation/hist/eartips": prob_output_val[0, self.model.n_stacks-1, :, :, 1],
                         "validation/hist/spikelets": prob_output_val[0, self.model.n_stacks-1, :, :, 2]}
            self.logger.log(self.sess, step=self.model.step.eval(self.sess), what=valid_log)

            self.sess.run(self.model.increment_epoch)

            min_validation_loss = self.model.min_validation_loss.eval(self.sess)
            print("\tCurrent minimal validation loss: {:.4f}".format(min_validation_loss))
            if loss_val < min_validation_loss:
                update = self.model.min_validation_loss.assign(loss_val)
                self.sess.run(update)
                self.model.save(self.sess)


if __name__ == "__main__":
    import tensorflow as tf
    import json
    from easydict import EasyDict

    from logger import Logger
    from loader import StackedHourglassLoader
    from model import StackedHourglassModel

    config_fname = 'config.json'

    with open(config_fname, 'r') as f:
        config = json.load(f)
    config = EasyDict(config)

    loader = StackedHourglassLoader(config)
    model = StackedHourglassModel(config)
    logger = Logger(config)

    sess = tf.Session()
    trainer = StackedHourglassTrainer(config, sess, model, logger, loader)
    trainer.train()

    logger.stop()

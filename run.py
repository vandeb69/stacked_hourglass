import os
import sys
import tensorflow as tf
import json
from easydict import EasyDict

from logger import Logger
from loader import StackedHourglassLoader
from model import StackedHourglassModel
from trainer import StackedHourglassTrainer
from utils.dirs import clear_dirs

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config_fname = str(sys.argv[1])
print(config_fname)
with open(config_fname, 'r') as f:
    config = json.load(f)
config = EasyDict(config)

if config.restart:
    clear_dirs((config.loader_dir, config.log_dir, config.cp_dir), name=config.name)

loader = StackedHourglassLoader(config)
model = StackedHourglassModel(config)
logger = Logger(config)

sess = tf.Session()
trainer = StackedHourglassTrainer(config, sess, model, logger, loader)
trainer.train()

logger.stop()

# nohup python - u run.py config.json > run1.out &

import logging
import tensorflow as tf
import numpy as np

def basic_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger()


def random_set_seed():
    tf.random.set_seed(1337)
    np.random.seed(1337)


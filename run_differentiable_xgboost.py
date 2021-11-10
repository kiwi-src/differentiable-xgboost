import tensorflow as tf
import numpy as np
from tensorflow import keras
import time

from trainer_config import TrainerConfig

from datasets.breast_cancer import Dataset
#from data.fraud import Dataset

from ensemble import Ensemble
import logging
import utils

utils.random_set_seed()
logger = utils.basic_logger()


"""
LEARNING_RATE = 0.001
NUM_EPOCHS = 2000
DEPTH = 7
BATCH_NORM = True
BATCH_NORM_TRAINABLE = True
BATCH_SIZE = 1024 * 2 #64 #128 #256*2*2
NUM_EXAMPLES = 100000
XGBOOST = False
NUM_CLASSES = 2
FEATURE_SELECTION_RATE = 0.95
NUM_MODELS = 10
"""


class Model(keras.Model):
    
    @staticmethod
    def _create_features_mask(num_features, num_selected_features):
        ones = tf.ones([num_selected_features], dtype=tf.int32)
        zeros = tf.zeros([num_features-num_selected_features], dtype=tf.int32)
        return tf.random.shuffle(tf.concat([ones, zeros], axis=0))
    
    def __init__(self, config, num_classes):
        super(Model, self).__init__()
        self.depth = config.depth
        self.num_leaves = 2 ** config.depth
        self.num_classes = num_classes
        self.config = config

        num_selected_features = int(config.num_features * config.feature_selection_rate)
        self.selected_features_mask = self._create_features_mask(config.num_features, num_selected_features)
        self.leaves_weights = tf.Variable(
            initial_value=tf.random_normal_initializer()(shape=[self.num_leaves, self.num_classes]),
            dtype=tf.float32,
            trainable=True
            )

        self.features_weights = tf.Variable(
            initial_value=tf.random_normal_initializer()(shape=[num_selected_features, self.num_leaves-1]),
            dtype=tf.float32, 
            trainable=True        
            )

        if config.xgboost:
            self.threshold = tf.Variable(
                initial_value=tf.random_normal_initializer()(shape=[self.num_leaves-1]),
                dtype=tf.float32, 
                trainable=True
                )

            self.temperature = tf.Variable(
                initial_value=tf.random_normal_initializer()(shape=[self.num_leaves-1]),
                dtype=tf.float32,
                trainable=True
                )

        if config.batch_norm:
            self.batch_norm = keras.layers.BatchNormalization(scale=config.batch_norm_trainable, center=config.batch_norm_trainable)


    def call(self, features, training):
        batch_size = tf.shape(features)[0]
        if self.config.batch_norm:
            features = self.batch_norm(features)

        # Select features used by this model
        # (batch_size, num_features) -> (batch_size, num_selected_features)
        features = tf.boolean_mask(features, self.selected_features_mask, axis=1)

        if self.config.xgboost:
            # (num_selected_features, num_leaves-1)
            feature_selection = tf.nn.softmax(self.features_weights, axis=0)
            
            # (batch_size, num_selected_features) x (num_selected_features, num_leaves-1) -> (batch_size, num_leaves-1)        
            selected_feature = tf.matmul(features, feature_selection)

            # (batch_size, num_leaves-1)   
            decisions = tf.sigmoid((-selected_feature + self.threshold) / tf.sigmoid(self.temperature))
        else:
            # (batch_size, num_selected_features) x (num_selected_features, num_leaves-1) -> (batch_size, num_leaves-1)
            decisions = tf.matmul(features, self.features_weights)

            # Convert decisions to probs
            decisions = tf.nn.sigmoid(decisions)

        # (batch_size, num_leaves-1) -> (batch_size, num_leaves-1, 1)
        decisions = tf.expand_dims(decisions, axis=2)

         # (batch_size, num_leaves-1) -> (batch_size, num_leaves-1, 2)
        decisions = keras.layers.concatenate([decisions, 1 - decisions], axis=2)
    
        nodes = tf.ones([batch_size, 1])
        start_index = 0
        end_index = 1
        num_level_leaves = 0
        for level in range(self.depth):
            # (batch_size, num_level_leaves, 2)
            level_decisions = decisions[:, start_index:end_index, :]

            # (batch_size, num_level_leaves, 2)
            nodes = tf.einsum('bi,bij->bij', nodes, level_decisions)

            num_level_leaves = 2 ** (level + 1)

            # Flatten: (batch_size, num_leaves-1)
            nodes = tf.reshape(nodes, [batch_size, num_level_leaves])

            start_index = end_index
            end_index = end_index + num_level_leaves

        if self.config.xgboost:
            outputs = tf.nn.softmax(tf.matmul(nodes,self.leaves_weights))
        else:
            # (num_leaves-1, num_classes)
            probs = tf.nn.softmax(self.leaves_weights)
            
            # (batch_size, num_leaves-1) * (num_leaves-1, num_classes) -> (batch_size, num_classes)
            outputs = tf.matmul(nodes, probs)
            
        return outputs

def main():
    NUM_EXAMPLES = None
    NUM_CLASSES = 2
    NUM_EPOCHS = 10000

    val_auc = tf.keras.metrics.AUC(name='val_auc', num_thresholds=200)
    train_ce_loss = tf.keras.metrics.Mean(name='train_ce_loss')
    val_ce_loss = tf.keras.metrics.Mean(name='eval_ce_loss')

    trainer_config = TrainerConfig(
        learning_rate=0.001,
        optimizer=tf.keras.optimizers.Adam,
        batch_norm=True,
        batch_norm_trainable=True,
        batch_size=128,
        depth=5,
        feature_selection_rate=1.0,
        xgboost=False,
        num_models=8
        )


    """
    LEARNING_RATE = 0.001
NUM_EPOCHS = 10000
DEPTH = 5
BATCH_NORM = True
BATCH_NORM_TRAINABLE = True
BATCH_SIZE = 128
NUM_EXAMPLES = None
XGBOOST = False
NUM_CLASSES = 2
NUM_MODELS = 8
FEATURE_SELECTION_RATE = 1.0
"""


    dataset = Dataset()
    train_dataset, test_dataset = dataset.load('tf', batch_size=trainer_config.batch_size, num_examples=NUM_EXAMPLES)
    trainer_config.num_features = len(dataset.feature_names)

    model = Ensemble(Model, config=trainer_config, num_models=trainer_config.num_models, num_classes=NUM_CLASSES)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer=keras.optimizers.Adam(learning_rate=trainer_config.learning_rate)

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            train_loss = loss_fn(labels, logits)
            grads = tape.gradient(train_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_ce_loss(train_loss)
    
    @tf.function
    def val_step(x, labels):
        logits = model(x, training=False)
        val_auc(y_true=tf.one_hot(labels, depth=2), y_pred=tf.nn.softmax(logits))
        val_loss = loss_fn(labels, logits)
        val_ce_loss(val_loss)

    best_val_auc = 0.0
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = 0
    start_time = time.time()
    
    for i in range(NUM_EPOCHS):
        val_auc.reset_states()
        train_ce_loss.reset_states()
        val_ce_loss.reset_states()

        for inputs, labels in test_dataset:
            val_step(inputs, labels)

        for inputs, labels in train_dataset:
            train_step(inputs, labels)

        val_loss = val_ce_loss.result()
        train_loss = train_ce_loss.result()
        
        if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_epoch = i
                best_val_auc = val_auc

        logging.info(
            f'Epoch {i+1} – train_loss {train_loss:.6f} '
            f'– val_loss {val_loss:.6f} '
            f'- val_auc {val_auc.result():.6f}')

        logging.info(
            f'Best epoch {best_epoch+1} '
            f'– train_loss {best_train_loss:.6f} '
            f'– val_loss {best_val_loss:.6f} '
            f'- val_auc {best_val_auc.result():.6f}')
        logging.info(f'{trainer_config.__dict__}')

    end_time = time.time()
    print(end_time-start_time)

if __name__ == '__main__':
    main()
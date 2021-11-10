import logging
import tensorflow as tf
from trainer_config import TrainerConfig
import time
from datasets.breast_cancer import Dataset
import utils
utils.random_set_seed()
logger = utils.basic_logger()

class Model(tf.keras.Model):
    def __init__(self, config, num_classes):
        super(Model, self).__init__()
        self._config = config

        if config.batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()
            self.batch_norm_layer2 = tf.keras.layers.BatchNormalization()
            self.batch_norm_layer3 = tf.keras.layers.BatchNormalization()

        self.layer_1 = tf.keras.layers.Dense(units=config.num_units,
                                             kernel_regularizer=tf.keras.regularizers.l2(
                                                 config.l2),
                                             activation=tf.keras.activations.relu)

        if config.hidden:
            self.layer_2 = tf.keras.layers.Dense(units=8,
                                                 kernel_regularizer=tf.keras.regularizers.l2(
                                                     config.l2),
                                                 activation=tf.keras.activations.relu)

        self.layer_3 = tf.keras.layers.Dense(
            units=num_classes, activation=None)

    def call(self, features):
        if self._config.batch_norm:
            layer = self.layer_1(self.batch_norm_layer(features))
        else:
            layer = self.layer_1(features)

        layer = self.batch_norm_layer2(layer)

        if self._config.hidden:
            layer = self.layer_2(layer)
            if self._config.batch_norm:
                layer = self.batch_norm_layer3(layer)

        return self.layer_3(layer)

    def compute_regularizer_loss(self):
        loss = 0
        for layer in self.layers:
            if hasattr(layer, 'layers') and layer.layers:
                raise NotImplementedError
            if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer:
                loss += layer.kernel_regularizer(layer.kernel)
            if hasattr(layer, 'bias_regularizer') and layer.bias_regularizer:
                loss += layer.bias_regularizer(layer.bias)
        return loss


if __name__ == '__main__':
    eval_auc = tf.keras.metrics.AUC(name='eval_auc', num_thresholds=200)

    # Compute mean cross entropy loss on train set for one epoch
    train_ce_loss = tf.keras.metrics.Mean(name='train_ce_loss')

    # Compute mean cross entropy loss on eval set for one epoch
    eval_ce_loss = tf.keras.metrics.Mean(name='eval_ce_loss')

    NUM_EPOCHS = 10000
    config = TrainerConfig(
        learning_rate=1e-3,
        regularization=True,
        l2=0.01,
        num_units=256,
        optimizer=tf.keras.optimizers.Adam,
        batch_norm=True,
        batch_size=128,
        num_examples=None,
        hidden=True
    )

    dataset = Dataset()
    cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    optimizer = config.optimizer(learning_rate=config.learning_rate)
    model = Model(config, num_classes=2)

    @tf.function()
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            mean_cross_entropy_loss = cross_entropy_loss(
                tf.expand_dims(labels, axis=1), logits)
            mean_loss = mean_cross_entropy_loss + model.compute_regularizer_loss()

        if config.regularization:
            gradients = tape.gradient(mean_loss, model.trainable_variables)
        else:
            gradients = tape.gradient(
                mean_cross_entropy_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_ce_loss(mean_cross_entropy_loss)

    @tf.function
    def eval_step(inputs, labels):
        logits = model(inputs, training=False)
        eval_auc(y_true=tf.one_hot(labels, depth=2),
                 y_pred=tf.nn.softmax(logits))
        mean_cross_entropy_loss = cross_entropy_loss(
            tf.expand_dims(labels, axis=1), logits)
        eval_ce_loss(mean_cross_entropy_loss)

    train_examples, eval_examples = dataset.load(
        'tf', batch_size=config.batch_size, num_examples=config.num_examples)
    start = time.time()

    best_eval_auc = 0.0
    best_eval_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        eval_auc.reset_states()
        train_ce_loss.reset_states()
        eval_ce_loss.reset_states()

        # training
        for input, label in train_examples:
            train_step(input, label)

        # evaluation
        for input, label in eval_examples:
            eval_step(input, label)

        log = f"Epoch {epoch} " \
            f"- {train_ce_loss.name} {train_ce_loss.result():.6f} " \
            f"- {eval_ce_loss.name} {eval_ce_loss.result():.6f} " \
            f"- {eval_auc.name} {eval_auc.result():.6f} "
        logger.info(f'{log}')

        if eval_ce_loss.result() < best_eval_loss:
            best_train_loss = train_ce_loss.result()
            best_eval_loss = eval_ce_loss.result()
            best_eval_auc = eval_auc.result()
            best_epoch = epoch

        logger.info(
            f'Best epoch {best_epoch} - train_loss {best_train_loss:.6f} - val_loss {best_eval_loss:.6f} - val_auc {best_eval_auc:.6f}')
        logger.info(f'{config.__dict__}')

    end = time.time()
    print(f'time {end - start}')

import tensorflow as tf
from tensorflow import keras
import numpy as np


class BaseNet(keras.Model):

    def __init__(self, model, learning_rate, discount_factor):
        super(BaseNet, self).__init__()
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.discount_factor = discount_factor

    def load(self, model_folder):
        self.model = keras.models.load_model("{}/model.h5".format(model_folder))

    def save(self, model_folder):
        self.model.save(model_folder)

    def call(self, state):
        return self.model.predict(state)

    def save_checkpoits(self, checkpoint_folder):
        self.model.save_weights(checkpoint_folder)

    def load_checkpoits(self, checkpoint_folder):
        self.model.load_weights(checkpoint_folder)

    @tf.function
    def get_single_best_action(self, state):
        s = tf.reshape(state, [1, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model(s)[0])

    def get_batch_best_action(self, batch, state):
        s = state.reshape([batch, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model(s), axis=1)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DQN(keras.Model):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._create_network()
        self.discount_factor = 0.99

    def load(self, model_folder):
        self.model = keras.models.load_model(model_folder)

    def save(self, model_folder):
        self.model.save(model_folder)

    def _create_network(self):
        dropout_prob = 0.3

        inputs = keras.Input(shape=self.input_dim, name='frame')
        x = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(inputs)
        x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
        x = layers.Dropout(dropout_prob)(x)
        q = layers.Dense(self.output_dim, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation=None)(x)

        model = keras.Model(inputs=inputs, outputs=q, name='vizdoom_agent_model')

        model.compile(optimizer='adam', loss='mse')

        return model

    def call(self, state):
        return self.model.predict(state)

    def train_step(self, memory):
        batch_size = 32
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(self.model.predict(s2), axis=1)
        target_q = self.model.predict(s1)

        target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2
        loss = self.model.train_on_batch(x=s1, y=target_q)
        return loss

    def get_best_action(self, state):
        s = state.reshape([1, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model.predict(s)[0])


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DQN(keras.Model):

    def __init__(self, input_dim, output_dim, action_size):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._create_network()
        self.discount_factor = 0.99
        self.action_size = action_size

    def load(self, model_folder):
        self.model = keras.models.load_model(model_folder)

    def save(self, model_folder):
        self.model.save(model_folder)

    def _create_network(self):
        dropout_prob = 0.3

        inputs = keras.Input(shape=self.input_dim, name='frame')
        x = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same',
                          kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
                          activation='relu')(inputs)
        x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
        x = layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
        flatten = layers.Flatten()(x)

        # Calcular values
        value = layers.Dense(512, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(flatten)
        value = layers.Dense(1, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1))(value)

        # Calcular advantage
        advantage = layers.Dense(512, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
                             activation='relu')(flatten)
        advantage = layers.Dense(self.action_size, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1))(
            advantage)

        # Agregating layer
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        output = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))

        model = keras.Model(inputs=inputs, outputs=output, name='vizdoom_agent_model')

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

    def get_single_best_action(self, state):
        s = state.reshape([1, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model.predict(s)[0])

    def get_batch_best_action(self, batch, state):
        s = state.reshape([batch, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model.predict(s), axis=1)

    @tf.function
    def compute_loss(self, state, target_Q):
        output = self.model.predict(state)

        Q = tf.reduce_sum(tf.multiply(output, actions_), axis=1)
        absolute_errors = tf.abs(target_Q - Q)  # for updating Sumtree
        loss = tf.reduce_mean(self.model.trainable_weights * tf.squared_difference(target_Q, Q))
        return loss, absolute_errors

    @tf.function
    def run_train(self, state, target_Q,  optimizer):
        # if isinstance(data, tuple):
        #     data = data[0]
        with tf.GradientTape() as tape:
            loss, absolute_errors = self.compute_loss(state, target_Q)

        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # print("Train loss: {}".format(kl_loss))



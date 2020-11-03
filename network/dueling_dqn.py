import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Dueling_DQN(keras.Model):

    def __init__(self, input_dim, output_dim):
        super(Dueling_DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._create_network()
        self.discount_factor = 0.99
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def load(self, model_folder):
        self.model = keras.models.load_model(model_folder)

    def save(self, model_folder):
        self.model.save(model_folder)
    #
    # def _create_network(self):
    #     dropout_prob = 0.3
    #
    #     inputs = keras.Input(shape=self.input_dim, name='frame')
    #     x = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                       kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
    #                       activation='relu')(inputs)
    #     x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                       kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
    #                       activation='relu')(x)
    #     x = layers.Flatten()(x)
    #     x = layers.Dense(512, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
    #                      activation='relu')(x)
    #     x = layers.Dropout(dropout_prob)(x)
    #     q = layers.Dense(self.output_dim, kernel_initializer='glorot_normal',
    #                      bias_initializer=tf.constant_initializer(0.1), activation=None)(x)
    #
    #     model = keras.Model(inputs=inputs, outputs=q, name='vizdoom_agent_model')
    #
    #     # model.compile(optimizer='adam', loss='mse')
    #
    #     return model
    #
    def _create_network(self):

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
        advantage = layers.Dense(self.output_dim, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1))(
            advantage)

        # Agregating layer
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        output = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))

        model = keras.Model(inputs=inputs, outputs=output, name='vizdoom_agent_model')

        return model

    def call(self, state):
        return self.model.predict(state)

    # @tf.function
    def train_step(self, memory, batch_size):

        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        return self.computer_loss(s1, a, s2, isterminal, r)

    # @tf.function
    def computer_loss(self, s1, a, s2, isterminal, r):
        with tf.GradientTape() as tape:
            Q_next = tf.stop_gradient(tf.reduce_max(self.model(s2), axis=1))
            Q_pred = tf.reduce_sum(tf.multiply(tf.one_hot(a, self.output_dim), self.model(s1)), axis=1)
            loss = tf.reduce_mean(0.5*(r + (1-isterminal)*self.discount_factor*Q_next - Q_pred)**2)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def mse(self, q_values, reference_q):
        return tf.reduce_mean(tf.pow(q_values - reference_q, 2))

    # @tf.function
    def get_single_best_action(self, state):
        # s = tf.reshape(state, [1, self.input_dim[0], self.input_dim[1], 1])

        s = state.reshape([1, self.input_dim[0], self.input_dim[1], 1])
        # a = tf.argmax(self.model(s)[0])
        # print("Action: {}".format(a))
        return tf.argmax(self.model(s)[0])

    def get_batch_best_action(self, batch, state):
        s = state.reshape([batch, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model(s), axis=1)

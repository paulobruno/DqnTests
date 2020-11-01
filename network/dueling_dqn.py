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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    def load(self, model_folder):
        self.model = keras.models.load_model(model_folder)

    def save(self, model_folder):
        self.model.save(model_folder)

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


    def train_step(self, memory):
        batch_size = 32
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        return self.computer_loss(s1, a, s2, isterminal, r)


    def computer_loss(self, s1, a, s2, isterminal, r):
        with tf.GradientTape() as tape:
            target_q = self.model(s1)
            next_q_values = np.max(self.model(s2), axis=1)

            action_q_values = tf.reduce_sum(tf.multiply(tf.one_hot(a, 8), target_q), axis=1)
            next_state_values_target = tf.reduce_max(next_q_values, axis=-1)

            # target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * next_state_values_target
            reference_q = r + self.discount_factor * (1 - isterminal) * next_state_values_target
            absolute_errors = (action_q_values - reference_q)**2
            loss = tf.reduce_mean(absolute_errors)


        grads = tape.gradient(loss, self.weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss

    def get_single_best_action(self, state):
        s = state.reshape([1, self.input_dim[0], self.input_dim[1], 1])
        # a = tf.argmax(self.model(s)[0])
        # print("Action: {}".format(a))
        return tf.argmax(self.model(s)[0])

    def get_batch_best_action(self, batch, state):
        s = state.reshape([batch, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model(s), axis=1)

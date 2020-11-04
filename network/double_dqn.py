import tensorflow as tf
from tensorflow import keras
from network.networks import create_dueling_dqn
import numpy as np
from network.base_network import BaseNet


class DDQN(BaseNet):

    def __init__(self, input_dim, output_dim):
        super(DDQN, self).__init__(create_dueling_dqn(input_dim, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.discount_factor = 0.99
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.target_model = create_dueling_dqn(input_dim, output_dim)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    # @tf.function
    def train_step(self, memory, batch_size):

        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        loss = self.computer_loss(s1, a, s2, isterminal, r)
        self.update_target()
        return loss

    @tf.function
    def computer_loss(self, s1, a, s2, isterminal, r):
        # target = self.target_model(s1).numpy()

        # q_next_pred = tf.reduce_max(self.target_model(s2), axis=1).numpy()
        # target[np.arange(target.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q_next_pred

        # tf.reduce_mean(0.5 * (r + (1 - isterminal) * self.discount_factor * Q_next - Q_pred) ** 2)
        # target = r + (1 - isterminal) * self.discount_factor * tf.reduce_max(self.target_model(s2), axis=1)

        with tf.GradientTape() as tape:
            # predict Q(s,a) given the batch of states
            # q_pred = tf.reduce_sum(tf.multiply(tf.one_hot(a, self.output_dim), self.model(s1)), axis=1)
            # predict Q(s',a') from the evaluation network
            q_next_pred = tf.stop_gradient(tf.reduce_max(self.target_model(s2), axis=1))
            # Q_next = tf.stop_gradient(tf.reduce_max(self.target_network(s2), axis=1))
            # Q_next = tf.stop_gradient(tf.reduce_max(self.target_network(s2), axis=1))
            # Q_pred = self.model(s1)
            # loss = tf.losses.mean_squared_error(target, Q_pred)
            Q_pred = tf.reduce_sum(tf.multiply(tf.one_hot(a, self.output_dim), self.model(s1)), axis=1)
            loss = tf.reduce_mean(0.5 * (r + (1 - isterminal) * self.discount_factor * q_next_pred - Q_pred) ** 2)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss

    # @tf.function
    def get_single_best_action(self, state):
        s = state.reshape([1, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model(s)[0])

    def get_batch_best_action(self, batch, state):
        s = state.reshape([batch, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model(s), axis=1)

import tensorflow as tf
from network.networks import create_dense_reinforce, create_vanila_dqn
from network.base_network import BaseNet
import numpy as np
import tensorflow_probability as tfp


class Reinforce(BaseNet):

    def __init__(self, input_dim, output_dim):
        super(Reinforce, self).__init__(create_vanila_dqn(input_dim, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.discount_factor = 0.99
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def train_step(self, experiences):
        # for s1, a, s2, isdone, r in zip(*experiences):
        self.computer_loss(experiences[0], experiences[4], experiences[1])

    @tf.function
    def computer_loss(self, s, r, a):
        with tf.GradientTape() as tape:
            p = self.model(s)
            log_prob = tfp.distributions.Categorical(probs=p, dtype=tf.float32).log_prob(a)
            loss = -log_prob * r

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss


    # @tf.function
    def get_single_best_action(self, state):
        prob = self.model(np.array([state]))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

import tensorflow as tf
from network.networks import create_dueling_dqn
from network.base_network import BaseNet


class Dueling_DQN(BaseNet):

    def __init__(self, input_dim, output_dim):
        super(Dueling_DQN, self).__init__(create_dueling_dqn(input_dim, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = create_dueling_dqn(input_dim, output_dim)
        self.discount_factor = 0.99
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


    def train_step(self, memory, batch_size):

        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        return self.computer_loss(s1, a, s2, isterminal, r)

    @tf.function
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
        s = state.reshape([1, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model(s)[0])

    def get_batch_best_action(self, batch, state):
        s = state.reshape([batch, self.input_dim[0], self.input_dim[1], 1])
        return tf.argmax(self.model(s), axis=1)
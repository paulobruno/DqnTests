import tensorflow as tf
from network.networks import create_dueling_dqn, create_vanila_dqn
from network.base_network import BaseNet


class FixedDDQN(BaseNet):

    def __init__(self, input_dim, output_dim, learning_rate, discount_factor, is_duel=True):
        model = create_dueling_dqn(input_dim, output_dim) if is_duel else create_vanila_dqn(input_dim, output_dim)
        super(FixedDDQN, self).__init__(model, learning_rate, discount_factor)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.discount_factor = 0.99
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.target_model = create_dueling_dqn(input_dim, output_dim)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_step(self, data):
        loss = self.computer_loss(*data)
        self.update_target()
        return loss

    @tf.function
    def computer_loss(self, s1, a, s2, isterminal, r):
        with tf.GradientTape() as tape:
            q_next_pred = tf.stop_gradient(tf.reduce_max(self.target_model(s2), axis=1))
            Q_pred = tf.reduce_sum(tf.multiply(tf.one_hot(a, self.output_dim), self.model(s1)), axis=1)
            absolute_errors = tf.stop_gradient(tf.abs(q_next_pred - Q_pred))
            loss = tf.reduce_mean(0.5 * (r + (1 - isterminal) * self.discount_factor * q_next_pred - Q_pred) ** 2)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss, absolute_errors

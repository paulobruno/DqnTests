import tensorflow as tf
from network.networks import create_dueling_dqn, create_vanila_dqn
from network.base_network import BaseNet


class DDQN(BaseNet):

    def __init__(self, input_dim, output_dim, learning_rate, discount_factor, is_duel=True):
        model = create_dueling_dqn(input_dim, output_dim) if is_duel else create_vanila_dqn(input_dim, output_dim)
        super(DDQN, self).__init__(model, learning_rate, discount_factor)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def train_step(self, data):
        return self.computer_loss(*data)

    @tf.function
    def computer_loss(self, s1, a, s2, isterminal, r):
        with tf.GradientTape() as tape:
            Q_next = tf.stop_gradient(tf.reduce_max(self.model(s2), axis=1))
            Q_pred = tf.reduce_sum(tf.multiply(tf.one_hot(a, self.output_dim), self.model(s1)), axis=1)
            loss = tf.reduce_mean(0.5 * (r + (1 - isterminal) * self.discount_factor * Q_next - Q_pred) ** 2)
            absolute_errors = tf.stop_gradient(tf.abs(Q_next - Q_pred))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(Q_next, Q_pred)
        return self.loss_tracker.result(), absolute_errors

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]

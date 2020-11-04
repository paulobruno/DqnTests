import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout


def create_dueling_dqn(input_dim, output_dim):

    inputs = Input(shape=input_dim, name='frame')
    x = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same',
                      kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
                      activation='relu')(inputs)
    x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
    flatten = Flatten()(x)

    # Calcular values
    value = Dense(512, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(flatten)
    value = Dense(1, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1))(value)

    # Calcular advantage
    advantage = Dense(512, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
                             activation='relu')(flatten)
    advantage = Dense(output_dim, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1))(
        advantage)

    # Agregating layer
    # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
    output = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))

    model = Model(inputs=inputs, outputs=output, name='vizdoom_agent_model')

    return model


def create_vanila_dqn(input_dim, output_dim):
        dropout_prob = 0.3

        inputs = Input(shape=input_dim, name='frame')
        x = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same',
                          kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
                          activation='relu')(inputs)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same',
                          kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
                          activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1),
                         activation='relu')(x)
        x = Dropout(dropout_prob)(x)
        q = Dense(output_dim, kernel_initializer='glorot_normal',
                         bias_initializer=tf.constant_initializer(0.1), activation=None)(x)

        model = Model(inputs=inputs, outputs=q, name='vizdoom_agent_model')

        # model.compile(optimizer='adam', loss='mse')

        return model
import tensorflow as tf
from tensorflow import keras


class BaseNet(keras.Model):

    def __init__(self, model):
        super(BaseNet, self).__init__()
        self.model = model

    def load(self, model_folder):
        self.model = keras.models.load_model("{}/model.h5".format(model_folder))

    def save(self, model_folder):
        self.model.save(model_folder)

    def call(self, state):
        data = tf.convert_to_tensor(state)
        return self.model.predict(data)
        # return self.model.predict(state)

    def save_checkpoits(self, checkpoint_folder):
        self.model.save_weights(checkpoint_folder)

    def load_checkpoits(self, checkpoint_folder):
        self.model.load_weights(checkpoint_folder)


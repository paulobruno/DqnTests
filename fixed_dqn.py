from __future__ import absolute_import, print_function, division, unicode_literals


# parse command line arguments
import argparse

parser = argparse.ArgumentParser(
    description="Train an agent in a given scenario. The agent can be trained from scratch or load a trained model. Be careful: if loading a previously trained agent the scenario given should be the same, if not it could break or run as usual but with a poor performance.")

parser.add_argument(
    "config_file",
    help="CFG file with settings of the ViZDoom scenario.")
parser.add_argument(
    "-l", "--load-model",
    action="store_true",
    help="Load a model from '--model-folder' if it is given and exists.")
parser.add_argument(
    "-s", "--save-model",
    action="store_true",
    help="Save a model in '--model-folder' if it is given. If no '--model-folder' is given, a folder called 'temp_model' will be created.")
parser.add_argument(
    "-m", "--model-folder",
    metavar="FOLDERNAME",
    default="temp_model",
    help="Path to the folder containing the model to be saved or loaded.")
parser.add_argument(
    "-i", "--save-model-interval",
    type=int,
    metavar="N",
    default=5,
    help="The model will be saved every N epochs. [default=5]")
parser.add_argument(
    "-show", "--show-model",
    metavar="FILENAME",
    help="Print the model architecture on screen and save a PNG image.",
    default="")
parser.add_argument(
    "-v", "--enable-training-view",
    action="store_true",
    help="Enable rendering of the game during agent training.")
parser.add_argument(
    "-log", "--log-file",
    metavar="FILENAME",
    help="Path to a file to save the results.",
    default="temp_log_file.txt")

args = parser.parse_args()

import vizdoom
import itertools as it
import numpy as np
import cv2
import os
import argparse

from random import sample, randint, random
from time import time, sleep
from tqdm import trange

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import test_maps


# limit gpu usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# deep q-learning settings
learning_rate = 0.0001
discount_factor = 0.99
replay_memory_size = 10000
frame_repeat = 8

# network settings
batch_size = 64
dropout_prob = 0.3

# training regime
epochs = 10
episodes_to_train = 100
episodes_to_test = 100
episodes_to_watch = 5
target_update_interval = 6

# frame size
FRAME_WIDTH = 64
FRAME_HEIGHT = 48
FRAME_CHANNELS = 1

epsilon_per_epoch = np.zeros(epochs, dtype=np.float32)

for i in range(epochs):
    epsilon_per_epoch = 


# down-sample the input image
def preprocess(img):
    img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
    img = img.astype(np.float32)
    img = img / 255.0
    return img


class ReplayMemory:
    def __init__(self, capacity):
        self.size = 0
        self.oldest_index = 0
        self.capacity = capacity

        self.old_state = np.zeros((capacity, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.int32)
        self.new_state = np.zeros((capacity, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.float32)

    def store_sample(self, old_state, action, reward, isterminal, new_state):
        self.old_state[self.oldest_index] = old_state
        self.action[self.oldest_index] = action
        self.reward[self.oldest_index] = reward
        self.isterminal[self.oldest_index] = isterminal
        self.new_state[self.oldest_index] = new_state
        
        self.oldest_index = (self.oldest_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_samples(self, num_samples):
        i = random.sample(range(self.size), num_samples)
        return self.old_state[i], self.action[i], self.reward[i], self.isterminal[i], self.new_state[i]

        
def create_network(num_available_actions):

    inputs = keras.Input(shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), name='frame')
    x = layers.Conv2D(filters=8, kernel_size=(6, 6), strides=(3, 3), kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(inputs)
    x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
    q = layers.Dense(num_available_actions, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation=None)(x)
    
    model = keras.Model(inputs=inputs, outputs=q, name='vizdoom_agent_model')

    model.compile(optimizer='adam', loss='mse')
    
    return model
    
    
def learn(model, state, target_q):
    loss = model.train_on_batch(x=state, y=target_q)
    return loss


def get_q_values(model, state):
    return model.predict(state)


def get_best_action(model, state):
    s = state.reshape([1, FRAME_WIDTH, FRAME_WIDTH, FRAME_CHANNELS])
    return tf.argmax(get_q_values(model, s)[0])
            

def learn_from_memory(dqn_model, target_model):
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(target_model, s2), axis=1)
        target_q = get_q_values(dqn_model, s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2        
        learn(dqn_model, s1, target_q)


def perform_learning_step(dqn_model, target_model, epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # linear decay
            return start_eps - (((epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs)) * (start_eps - end_eps))
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(dqn_model, s1)
        
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory(dqn_model, target_model)


def initialize_vizdoom(args):
    print("[1.] Initializing ViZDoom...")
    game = vizdoom.DoomGame()
    game.load_config(args.config_file)
    game.set_window_visible(args.enable_training_view)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.init()
    print("[.1] ... ViZDoom initialized.")
    return game


if __name__ == '__main__':

    # parse command line arguments
    argparser = argparse.ArgumentParser(
        description="Train an agent in a given scenario. The agent can be trained from scratch or load a trained model. Be careful: if loading a previously trained agent the scenario given should be the same, if not it could break or run as usual but with a poor performance.")

    argparser.add_argument(
        "config_file",
        help="CFG file with settings of the ViZDoom scenario.")
    argparser.add_argument(
        "-l", "--load-model",
        action="store_true",
        help="Load a model from '--model-folder' if it is given and exists.")
    argparser.add_argument(
        "-s", "--save-model",
        action="store_true",
        help="Save a model in '--model-folder' if it is given. If no '--model-folder' is given, a folder called 'temp_model' will be created.")
    argparser.add_argument(
        "-m", "--model-folder",
        metavar="FOLDERNAME",
        help="Path to the folder containing the model to be saved or loaded.",
        default="temp_model")
    argparser.add_argument(
        "-i", "--save-model-interval",
        type=int,
        metavar="N",
        default=5,
        help="The model will be saved every N epochs. [default=5]")
    argparser.add_argument(
        "-show", "--show-model",
        action="store_true",
        help="Print the model architecture on screen.")
    argparser.add_argument(
        "-v", "--enable-training-view",
        action="store_true",
        help="Enable rendering of the game during agent training.")
    argparser.add_argument(
        "-tr", "--training-log-file",
        metavar="FILENAME",
        help="Path to a file to save the results of training.",
        default="")
    argparser.add_argument(
        "-te", "--testing-log-file",
        metavar="FILENAME",
        help="Path to a file to save the results of testing.",
        default="")

    args = argparser.parse_args()
                

    # create vizdoom game
    game = initialize_vizdoom(args)

    num_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]

    memory = ReplayMemory(capacity=replay_memory_size)

    
    # load or create new model
    if (args.load_model and os.path.isdir(args.model_folder)):
        print("Loading model from " + args.model_folder + ".")
        dqn_model = keras.models.load_model(args.model_folder + str("/dqn_model"))
        target_model = keras.models.load_model(args.model_folder + str("/target_model"))
    else:
        if not os.path.isdir(args.model_folder):
            print("No folder was found in " + args.model_folder + ".")
        print("Creating a new model from scratch.")
        dqn_model = create_network(len(actions))
        target_model = create_network(len(actions))
        target_model.set_weights(dqn_model.get_weights())
        
    if args.show_model:
        dqn_model.summary()        
        

    # open file to save results
    if args.training_log_file:
        training_log_file = open(args.training_log_file, "w", buffering=1)
        
    if args.testing_log_file:
        testing_log_file = open(args.testing_log_file, "w", buffering=1)
        

    # training
    time_start = time()
    
    for epoch in range(epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        train_scores = []

        print("Training...")
        start_time = time()
        
        for episode in trange(episodes_per_epoch, leave=False):
            game.new_episode()
            
            while not game.is_episode_finished():
                perform_learning_step(dqn_model, target_model, epoch)
        
            score = game.get_total_reward()
            train_scores.append(score)

        # in seconds
        elapsed_time = time() - start_time

        train_scores = np.array(train_scores)

        print("Results: mean: %.1f±%.1f," %(train_scores.mean(), train_scores.std()), \
                "min: %.1f," %train_scores.min(), "max: %.1f," %train_scores.max())
                
        if args.training_log_file:
            print("{:.2f},{},{},{},{:.2f}".format(
                elapsed_time,
                train_scores.mean(),
                train_scores.min(),
                train_scores.max(),
                train_scores.std()),
                file=training_log_file)
                
        if ((epoch+1) % update_target_interval == 0):
            print("Updating target network")
            target_model.set_weights(dqn_model.get_weights())
            
        # save model
        if (args.save_model and ((epoch+1) % args.save_model_interval == 0)):
            print("Saving model to folder", args.model_folder)
            dqn_model.save(args.model_folder + str("/dqn_model"))
            target_model.save(args.model_folder + str("/target_model"))


        print("\nTesting...")
        test_episode = []
        test_scores = []
        
        start_time = time()
        
        for test_episode in trange(test_episodes_per_epoch, leave=False):
            game.new_episode()
            
            while not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer)
                best_action_index = get_best_action(dqn_model, state)
                game.make_action(actions[best_action_index], frame_repeat)
                
            r = game.get_total_reward()
            test_scores.append(r)

        # in seconds
        elapsed_time = time() - start_time
        
        test_scores = np.array(test_scores)
        
        print("Results: mean: %.1f±%.1f," % (test_scores.mean(), test_scores.std()), \
                "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())
        
        if args.testing_log_file:
            print("{:.2f},{},{},{},{:.2f}".format(
                elapsed_time,
                test_scores.mean(),
                test_scores.min(),
                test_scores.max(),
                test_scores.std()),
                file=testing_log_file)                

        print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    print("======================================")
    print("Training finished. It's time to watch!")

    # close file with results
    if args.training_log_file:
        training_log_file.close()
        
    if args.testing_log_file:
        testing_log_file.close()

    game.close()

    
    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(vizdoom.Mode.ASYNC_PLAYER)
    game.init()

    for i in range(episodes_to_watch):
        game.new_episode()
        
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(dqn_model, state)
            
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()
        
        # Sleep between episodes        
        sleep(1.0)
        score = game.get_total_reward()
        print("Score ep. " + str(i) + ": " + "{:2.0f}".format(score))
                
    game.close()

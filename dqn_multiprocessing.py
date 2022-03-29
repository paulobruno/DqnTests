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
    help="Path to the folder containing the model to be saved or loaded.",
    default="temp_model")
parser.add_argument(
    "-e", "--num-epochs",
    type=int,
    metavar="N",
    help="Num of epochs to train. [default=20]",
    default=20)
parser.add_argument(
    "-i", "--save-model-interval",
    type=int,
    metavar="N",
    help="The model will be saved every N epochs. [default=5]",
    default=5)
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

from cv2 import resize
from tqdm import trange

import itertools as it
import numpy as np
import os # to check if a model folder exists
import multiprocess

from random import sample, randint, random
from time import time, sleep
from multiprocessing import Process

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import test_maps


# Q-learning settings
learning_rate = 0.0001
discount_factor = 0.99
replay_memory_size = 10000

# NN learning settings
batch_size = 64
dropout_prob = 0.3

# Training regime
episodes_per_epoch = 200
test_episodes_per_epoch = 200

# Other parameters
resolution = (48, 64)
frame_repeat = 8
episodes_to_watch = 5


# TODO: virar args
num_network_updates = 10000


# Converts and down-samples the input image
def preprocess(img):
    img = resize(img, (resolution[1], resolution[0]))
    img = img.astype(np.float32)
    img = img / 255.0
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, reward, isterminal):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.r[self.pos] = reward
        self.isterminal[self.pos] = isterminal

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.r[i], self.isterminal[i]
        
        
def create_network(available_actions_count):

    inputs = keras.Input(shape=(resolution[0], resolution[1], 1), name='frame')
    x = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(inputs)
    x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
    x = layers.Dropout(dropout_prob)(x)
    q = layers.Dense(available_actions_count, kernel_initializer='glorot_normal', bias_initializer=tf.constant_initializer(0.1), activation=None)(x)
    
    model = keras.Model(inputs=inputs, outputs=q, name='vizdoom_agent_model')
    model.compile(optimizer='adam', loss='mse')
    
    return model
    
    
def learn(model, state, target_q):
    loss = model.train_on_batch(x=state, y=target_q)
    return loss


def get_q_values(model, state):
    return model.predict(state)


def get_best_action(model, state):
    s = state.reshape([1, resolution[0], resolution[1], 1])
    return tf.argmax(get_q_values(model, s)[0])
            

def learn_from_memory(q_network, memory):
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    for _ in (num_network_updates):
    
        # Get a random minibatch from the replay memory and learns from it.
        if memory.size > batch_size:
            s1, a, s2, r, isterminal = memory.get_sample(batch_size)

            q2 = np.max(get_q_values(model, s2), axis=1)
            target_q = get_q_values(model, s1)
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2        
            learn(model, s1, target_q)


def choose_action_from_state(state, model, epoch)
    ''' Take an action according to a policy and a Q-Learning model.
    The resulting trasition (s, a, r, s') is stores in a Replay Memory.'''

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * args.num_epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * args.num_epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(model, state)

    return a


def generate_memory(epoch, memory, q_network):

    # create vizdoom game
    game = initialize_vizdoom(args)
    game.new_episode()
    #episodes_starte.put(
    
    while True:
    
        # generate transition and store in memory
        s_old = preprocess(game.get_state().screen_buffer)
        
        a = choose_action_from_state(s_old, q_network, epoch)

        r = game.make_action(actions[a], frame_repeat)

        if game.is_episode_finished():
            s_new = None
            score = game.get_total_reward()
            train_scores.put(score) # train_scores is a Queue
            
            if train_scores.qsize() < num_episodes:
                game.new_episode()
                #episodes_started.put(
        else:
            s_new = preprocess(game.get_state().screen_buffer)

        memory.add_transition(s_old, a, s_new, r, isterminal)

      


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

    # shared replay memory
    BaseManager.register('ReplayMemory', ReplayMemory)
    manager = ReplayMemory()
    manager.start()    
    memory = manager.ReplayMemory(capacity=replay_memory_size)

    # create number of possible actions
    game = initialize_vizdoom(args)
    num_actions = game.get_available_buttons_size()
    game.close()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]

    # TODO: q network tbm tem q ser atualizada
    q_network = create_network(len(actions))
    #target_network = create_network(len(actions))
    
    
    for epoch in args.num_epochs:    
        print("\nEpoch %d\n-------" % (epoch + 1))        

        print("Training...")
        train_scores = []
        
        processes = []
        episode = args.num_episodes
    
        # processes that generate memory
        for i in range(num_processes):
            episode = episode - 1
            p = Process(target=play_episode, args=(epoch, memory, q_network))
            processes.append(p)
            p.start()
            
        # process that updates the networks
        p = Process(target=learn_from_memory, args(q_network, memory))
        processes.append(p)
        p.start()
            
        for p in processes:
            p.join()
    
        
        print("\nTesting...")
        test_scores = []
        
        for test_episode in trange(test_episodes_per_epoch, leave=False):
            game.set_seed(test_maps.TEST_MAPS[test_episode])
            game.new_episode()
            
            while not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer)
                best_action_index = get_best_action(q_network, state)
                game.make_action(actions[best_action_index], frame_repeat)
                
            r = game.get_total_reward()
            test_scores.append(r)
                    
        print("Results: mean: %.1f±%.1f," % (test_scores.mean(), test_scores.std()), \
              "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())
            
    '''
    # load or create new model
    if (args.load_model and os.path.isdir(args.model_folder)):
        print("Loading model from " + args.model_folder + ".")
        model = keras.models.load_model(args.model_folder)
    else:
        if not os.path.isdir(args.model_folder):
            print("No folder was found in " + args.model_folder + ".")
        print("Creating a new model from scratch.")
        model = create_network(len(actions))
        
    if args.show_model:
        model.summary()        
        keras.utils.plot_model(model, args.show_model + ".png", show_shapes=True)
    '''    
    '''
    # open file to save resultsMap: line
    if args.log_file:
        log_file = open(args.log_file, "w", buffering=1)
        print("Map,{}".format(args.config_file), file=log_file)
        print("Resolution,{}".format(resolution), file=log_file)
        print("Frame Repeat,{}".format(frame_repeat), file=log_file)
        print("Learning Rate,{}".format(learning_rate), file=log_file)
        print("Discount,{}".format(discount_factor), file=log_file)
        print("Replay Memory,{}".format(replay_memory_size), file=log_file)
        print("Batch Size,{}".format(batch_size), file=log_file)
        print("Dropout,{}".format(dropout_prob), file=log_file)
        print("Epochs,{}".format(args.num_epochs), file=log_file)
        print("Training Episodes,{}".format(episodes_per_epoch), file=log_file)
        print("Testing Episodes,{}".format(test_episodes_per_epoch), file=log_file)
        print("Training time,Training min,Training max,Training mean,Training std,Testing time,Testing min,Testing max,Testing mean,Testing std", file=log_file)
        

    # training
    time_start = time()
    
    for epoch in range(args.num_epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        
        train_scores = []


        print("Training...")
        start_time = time()
        
        for episode in trange(episodes_per_epoch, leave=False):
            game.new_episode()
            
            while not game.is_episode_finished():
                perform_learning_step(model, epoch)
        
            score = game.get_total_reward()
            train_scores.append(score)
            
        # in seconds
        train_elapsed_time = time() - start_time

        train_scores = np.array(train_scores)

        print("Results: mean: %.1f±%.1f," %(train_scores.mean(), train_scores.std()), \
                "min: %.1f," %train_scores.min(), "max: %.1f," %train_scores.max())
                
        # save model
        if (args.save_model and ((epoch+1) % args.save_model_interval == 0)):
            print("Saving model to folder", args.model_folder)
            model.save(args.model_folder)


        print("\nTesting...")
        test_scores = []
        
        start_time = time()
        
        for test_episode in trange(test_episodes_per_epoch, leave=False):
            game.set_seed(test_maps.TEST_MAPS[test_episode])
            game.new_episode()
            
            while not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer)
                best_action_index = get_best_action(model, state)
                game.make_action(actions[best_action_index], frame_repeat)
                
            r = game.get_total_reward()
            test_scores.append(r)

        # in seconds
        test_elapsed_time = time() - start_time
        
        test_scores = np.array(test_scores)
        
        print("Results: mean: %.1f±%.1f," % (test_scores.mean(), test_scores.std()), \
                "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())
        
        if args.log_file:
            print("{:.2f},{},{},{},{:.2f},{:.2f},{},{},{},{:.2f}".format(
                train_elapsed_time, train_scores.min(), train_scores.max(), train_scores.mean(), train_scores.std(),
                test_elapsed_time, test_scores.min(), test_scores.max(), test_scores.mean(), test_scores.std()),
                file=log_file)

        print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))


    print("======================================")
    print("Training finished. It's time to watch!")

    # close file with results
    if args.log_file:
        log_file.close()

    game.close()

    
    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(vizdoom.Mode.ASYNC_PLAYER)
    game.init()

    for i in range(episodes_to_watch):
        game.new_episode()
        
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(model, state)
            
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()
        
        # Sleep between episodes        
        sleep(1.0)
        score = game.get_total_reward()
        print("Score ep. " + str(i) + ": " + "{:2.0f}".format(score))
                
    game.close()
    '''

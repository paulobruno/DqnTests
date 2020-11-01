import vizdoom

from cv2 import resize
from tqdm import trange

import itertools as it
import numpy as np
import os # to check if a model folder exists

from random import sample, randint, random
from time import time, sleep
from multiprocessing import Process, Manager, current_process
from multiprocessing.managers import BaseManager

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
num_epochs = 2
num_episodes = 200
config_file = 'scenarios/basic/basic/basic.cfg'
num_processes = 2


# limit gpu usage
# by default, tensorflow allocates all available memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    

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

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]
    
    def get_size(self):
        return self.size

        
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
            

def learn_from_memory(model):
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(model, s2), axis=1)
        target_q = get_q_values(model, s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2        
        learn(model, s1, target_q)


def choose_action_from_state(state, model, epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * num_epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * num_epochs  # 60% of learning time

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
    game = initialize_vizdoom()
    game.new_episode()
    #episodes_starte.put(

    episode_finished = False

    while not episode_finished:
        
        # generate transition and store in memory
        s_old = preprocess(game.get_state().screen_buffer)
        
        #a = choose_action_from_state(s_old, q_network, epoch)
        a = randint(0, len(actions) - 1)
        
        r = game.make_action(actions[a], frame_repeat)

        episode_finished = game.is_episode_finished()
        s_new = preprocess(game.get_state().screen_buffer) if not episode_finished else None

        print('%s.: %s' % (current_process().name, memory.get_size()))
        memory.add_transition(s_old, a, s_new, r, episode_finished)
        print('.%s: %s' % (current_process().name, memory.get_size()))
    
    game.close()
    print('end')
    
    #score = game.get_total_reward()
    #train_scores.put(score) # train_scores is a Queue
    
    #learn_from_memory(model)


def initialize_vizdoom():
    print("[1.] Initializing ViZDoom...")
    game = vizdoom.DoomGame()
    game.load_config(config_file)
    game.set_window_visible(False)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.init()
    print("[.1] ... ViZDoom initialized.")
    return game


if __name__ == '__main__':
    
    # shared replay memory
    BaseManager.register('ReplayMemory', ReplayMemory)
    manager = BaseManager()
    manager.start()    
    memory = manager.ReplayMemory(capacity=replay_memory_size)

    # create vizdoom game
    game = initialize_vizdoom()
    num_actions = game.get_available_buttons_size()
    game.close()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]

    # TODO: q network tbm tem q ser atualizada
    q_network = create_network(len(actions))
    #target_network = create_network(len(actions))


    for epoch in range(num_epochs):    
        print("\nEpoch %d\n-------" % (epoch + 1))        

        print("Training...")
        train_scores = []
        
        processes = []
        episode = num_episodes
    
        # processes that generate memory
        for i in range(num_processes):
            p = Process(target=generate_memory, args=(epoch, memory, q_network))
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
    

import vizdoom
from random import randint, random
import itertools as it
import numpy as np
from cv2 import resize
from time import sleep
from utils.decorator import timeit

from network.dqn import DQN
from network.relay_memory import ReplayMemory
import tensorflow as tf
from tqdm import trange

import test_maps


class Agent:

    def __init__(self, num_epochs, batch_size, game, resolution, replay_memory_size, should_save_model,
                 frame_repeat=8, episodes_to_watch=10):
        self.log_on_tensorboard = True
        self.should_save_model = should_save_model
        self.num_epochs = num_epochs
        self.channel = 1
        self.eps = 1
        self.train_episodes_per_epoch = 200
        self.test_episodes_per_epoch = 200
        self.resolution = resolution
        self.game = game
        self.actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]
        self.frame_repeat = frame_repeat
        self.batch_size = batch_size
        self.episodes_to_watch = episodes_to_watch

        input_shape = (resolution[0], resolution[1], self.channel)
        self.dqn = DQN(input_shape, len(self.actions))

        state_shape = (replay_memory_size, resolution[0], resolution[1], self.channel)
        self.memory = ReplayMemory(state_shape)

        self.writer = tf.summary.create_file_writer('tensorboard')

    def preprocess(self, img):
        img = resize(img, (self.resolution[1], self.resolution[0]))
        img = img.astype(np.float32)
        img = img / 255.0
        return img

    def calc_epsilon(self, epocj):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * self.num_epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * self.num_epochs  # 60% of learning time

        if epocj < const_eps_epochs:
            return start_eps
        elif epocj < eps_decay_epochs:
            # Linear decay
            return start_eps - (epocj - const_eps_epochs) / \
                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    def get_action(self, state):
        if random() <= self.eps:
            return randint(0, len(self.actions) - 1)
        else:
            # Choose the best action according to the network.
            return self.dqn.get_best_action(state)

    @timeit
    def run_step(self):
        s1 = self.preprocess(self.game.get_state().screen_buffer)

        # With probability eps make a random action.
        a = self.get_action(s1)

        reward = self.game.make_action(self.actions[a], self.frame_repeat)

        isterminal = self.game.is_episode_finished()
        s2 = self.preprocess(self.game.get_state().screen_buffer) if not isterminal else None

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, isterminal, reward)

        if self.memory.size > self.batch_size:
            self.dqn.train_step(self.memory)

    def run_episode(self):
        self.game.new_episode()

        while not self.game.is_episode_finished():
            self.run_step()

        score = self.game.get_total_reward()
        # tf.summary.scalar('train_episore_score', score, step=episode)
        return score

    def run_episode_test(self, epoch):
        self.game.set_seed(test_maps.TEST_MAPS[epoch])
        self.game.new_episode()

        while not self.game.is_episode_finished():
            state = self.preprocess(self.game.get_state().screen_buffer)
            best_action_index = self.dqn.get_best_action(state)
            self.game.make_action(self.actions[best_action_index], self.frame_repeat)

        score = self.game.get_total_reward()
        return score

    def run_epoch_train(self):
        scores = [self.run_episode() for _ in trange(self.train_episodes_per_epoch, leave=False)]
        return np.mean(scores), np.max(scores), np.min(scores), np.std(scores)

    def run_epoch_test(self):
        scores = [self.run_episode_test(episode) for episode in trange(self.test_episodes_per_epoch, leave=False)]
        return np.mean(scores), np.max(scores), np.min(scores), np.std(scores)

    def run(self, save_interval=None, save_folder=None):
        with self.writer.as_default():
            with tf.summary.record_if(self.log_on_tensorboard):
                for epoch in range(self.num_epochs):
                    self.eps = self.calc_epsilon(epoch)
                    _mean, _max, _min, _std = self.run_epoch_train()
                    tf.summary.scalar('mean', _mean, step=epoch)
                    tf.summary.scalar('std', _std, step=epoch)
                    # SAVE MODEL HEREEEEE

                    if self.should_save_model and save_interval is not None and save_folder is not None \
                            and save_interval % epoch == 0:
                        self.dqn.save(save_folder)

                    self.run_episode_test(epoch)

        self.game.close()

    def eval(self):
        # Reinitialize the game with window visible
        self.game.set_window_visible(True)
        self.game.set_mode(vizdoom.Mode.ASYNC_PLAYER)
        self.game.init()

        for episode in range(self.episodes_to_watch):
            self.game.new_episode()

            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                best_action_index = self.dqn.get_best_action(state)

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(self.actions[best_action_index])
                for _ in range(self.frame_repeat):
                    self.game.advance_action()

            # Sleep between episodes
            sleep(1.0)
            score = self.game.get_total_reward()
            tf.summary.scalar('eval', score, step=episode)
            self.game.close()

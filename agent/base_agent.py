from random import randint, random
import numpy as np
from cv2 import resize
from time import sleep

# from network.double_dqn import DDQN as model
# from network.fixed_dqn import DDQN as model
from network.ddqn import DDQN
from network.fixed_dqn import FixedDDQN
# from memory.relay_memory import ReplayMemory as memory
from memory.memory import Memory as memory
import tensorflow as tf
from tqdm import trange

import test_maps


class BaseAgent:

    def __init__(self, params, model='fixed-dqn'):

        self.log_on_tensorboard = True
        self.should_save_model = params['should_save_model']
        self.num_epochs = params['num_epochs']
        self.channel = 1
        self.eps = 1
        self.train_episodes_per_epoch = params['train_episodes_per_epoch']
        self.test_episodes_per_epoch = params['test_episodes_per_epoch']
        self.resolution = params['resolution']
        self.game = params['game']
        self.actions = self.game.action_space()
        self.frame_repeat = params['frame_repeat']
        self.batch_size = params['batch_size']
        self.episodes_to_watch = params['episodes_to_watch']

        input_shape = (self.resolution[0], self.resolution[1], self.channel)

        if model == 'fixed-dqn':
            self.net = FixedDDQN(input_shape, len(self.actions), params['learning_rate'], params['discount_factor'])
        else:
            self.net = DDQN(input_shape, len(self.actions), params['learning_rate'], params['discount_factor'])

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
            return self.net.get_single_best_action(state)

    def run_step(self):
        pass

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
            state = self.preprocess(self.game.get_state())
            best_action_index = self.net.get_single_best_action(state)
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

                    if self.should_save_model and save_interval % (epoch+1) == 0:
                        self.net.save("{}/models/model.h5".format(save_folder))
                        self.net.save_weights("{}/checkpoints/".format(save_folder))

                    self.run_episode_test(epoch)

        self.game.close()

    def eval(self):
        # Reinitialize the game with window visible
        self.game.make_visible()
        self.game.init()

        for episode in range(self.episodes_to_watch):
            self.game.new_episode()

            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state())
                best_action_index = self.net.get_single_best_action(state)

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.play(self.actions[best_action_index], self.frame_repeat)

            # Sleep between episodes
            sleep(1.0)
            score = self.game.get_total_reward()
            tf.summary.scalar('eval', score, step=episode)
        self.game.close()

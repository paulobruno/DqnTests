import abc


class Game(abc.ABC):

    @abc.abstractmethod
    def get_state(self):
        pass

    @abc.abstractmethod
    def make_action(self, action, repeat=None):
        pass

    @abc.abstractmethod
    def get_state(self):
        pass

    @abc.abstractmethod
    def is_episode_finished(self):
        pass

    @abc.abstractmethod
    def get_total_reward(self):
        pass

    @abc.abstractmethod
    def set_seed(self, seed):
        pass

    @abc.abstractmethod
    def new_episode(self):
        pass

    @abc.abstractmethod
    def action_space(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def make_visible(self):
        pass





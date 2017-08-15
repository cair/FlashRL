import os
import time
import random
import numpy as np
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.models import load_model
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop
from tensorflow.contrib.keras.python.keras.engine import Input, Model
from tensorflow.contrib.keras.python.keras.utils.vis_utils import plot_model
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from tensorflow.contrib.keras.python.keras.layers.core import Flatten, Dense, Lambda, Reshape

from rl.Memory import Memory
from rl.settings import settings


class DQN:
    def __init__(self, state_size, action_space):
        self.memory = Memory(settings["memory_size"])

        # Parameters
        self.LEARNING_RATE = settings["learning_rate"]
        self.BATCH_SIZE = settings["batch_size"]
        self.GAMMA = settings["discount_factor"]

        # Epsilon decent
        self.EPSILON_START = settings["epsilon_start"]
        self.EPSILON_END = settings["epsilon_end"]
        self.EPSILON_DECAY = (self.EPSILON_END - self.EPSILON_START) / settings["epsilon_steps"]
        self.epsilon = self.EPSILON_START

        # Exploration parameters (fully random play)
        self.EXPLORATION_WINS = settings["exploration_wins"]
        self.EXPLORATION_WINS_COUNTER = 0

        # Episode data
        self.episode = 0  # Episode Count
        self.episode_loss = 0  # Loss sum of a episode
        self.episode_reward = 0  # Reward sum of a episode
        self.frame = 0  # Frame counter
        self.loss_list = []

        # State data
        self.state = None
        self.state_size = state_size

        self.is_clean = True

        # Action data
        self.action_size = len(action_space)

        #self.model = load_model(last_checkpoint)
        #self.target_model = load_model(last_checkpoint)

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.model.summary()
        print("DQNRUnner inited!")
        print("State size is: %s,%s,%s" % self.state_size)
        print("Action size is: %s" % self.action_size)
        print("Batch size is: %s " % self.BATCH_SIZE)

    def reset(self, state):

        # Get new initial state
        self.state = state

        # Update target model
        if self.target_model:
            self.update_target_model()

            # Save target model
            model_name = "./save/dqn_p%s_%s.h5" % (self.player.id, int(time.time()))
            self.save(model_name)
        else:
            pass
            # Lost the round, delete memories
            # self.memory.remove_n(self.iteration)

        # Print output
        print("Episode: %s, Epsilon: %s, Reward: %s, Loss: %s, Memory: %s" % (
            self.episode, self.epsilon, self.episode_reward, self.episode_loss / (self.frame + 1), self.memory.count))

        self.frame = 0

        # Reset loss sum
        self.episode_loss = 0

        # Reset episode reward
        self.episode_reward = 0


        # Increase episode
        self.episode += 1

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        # Neural Net for Deep-Q learning Model

        # Image input
        input_layer = Input(shape=self.state_size, name='image_input')
        x = Conv2D(16, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform', trainable=True)(input_layer)
        x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform', trainable=True)(x)
        #x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform', trainable=True)(x)
        #x = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform')(x)
        #x = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform')(x)
        x = Reshape((int(x.shape[1]), int(x.shape[2]), int(x.shape[3])))(x)
        x = Flatten()(x)

        # Value Stream
        vs = Dense(512, activation="relu", kernel_initializer='uniform')(x)
        vs = Dense(1, kernel_initializer='uniform')(vs)

        # Advantage Stream
        ad = Dense(512, activation="relu", kernel_initializer='uniform')(x)
        ad = Dense(self.action_size, kernel_initializer='uniform')(ad)

        policy = Lambda(lambda w: w[0] - K.mean(w[0]) + w[1])([vs, ad])
        #policy = keras.layers.merge([vs, ad], mode=lambda x: x[0] - K.mean(x[0]) + x[1], output_shape=(self.action_size,))

        model = Model(inputs=[input_layer], outputs=[policy])
        optimizer = RMSprop(lr=self.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="mse")
        #plot_model(model, to_file='./output/model.png', show_shapes=True, show_layer_names=True)

        return model

    @staticmethod
    def huber(y_true, y_pred):
        cosh = lambda x: (K.exp(x) + K.exp(-x)) / 2
        return K.mean(K.log(cosh(y_pred - y_true)), axis=-1)

    def load(self, name):
        self.model = load_model(name)
        self.target_model = load_model(name)

    def save(self, name):
        self.target_model.save(name)

    def train(self):

        if self.memory.count < self.BATCH_SIZE:
            return

        batch_loss = 0
        memories = self.memory.get(self.BATCH_SIZE)
        for s, a, r, s1, terminal in memories:
            # Model = We train on
            # Target = Draw actions from

            target = r

            tar_s = self.target_model.predict(np.array([s]))
            if not terminal:
                tar_s1 = self.target_model.predict(np.array([s1]))
                target = r + self.GAMMA * np.amax(tar_s1[0])

            tar_s[0][a] = target
            loss = (r + (self.GAMMA * np.amax(tar_s1[0]) - np.amax(tar_s[0]))) ** 2

            history = self.model.fit(np.array([s]), tar_s, epochs=1, batch_size=1, callbacks=[], verbose=0)
            batch_loss += loss
            self.episode_loss += loss

    def act(self, state):
        if np.random.uniform() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploit Q-Knowledge
        act_values = self.target_model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def reward_fn(self):
        score = 0
        return score


    def update(self):
        self.is_clean = False

        # 1. Do action
        # 2. Observe
        # 3. Train
        # 4. set state+1 to state
        action = self.act()
        self.action_distribution[action] += 1
        s, a, s1, r, terminal, _ = self.state, action, *self.game.step(self.player, action, settings["grayscale"])

        reward = self.reward_fn()

        self.memory.add([s, a, reward, s1, terminal])

        self.frame += 1
        self.state = s1
        self.episode_reward += reward

        self.epsilon = max(0, self.epsilon + self.EPSILON_DECAY)


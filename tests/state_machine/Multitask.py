import threading
import numpy as np
import time

import pygame
from pyVNC.constants import K_LEFT, K_RIGHT
from PIL import Image
from rl.Main import DQN
from state.multitask_one.model import StateModel
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class Multitask:

    def state_debugger(self, s, s1, r):
        du = np.sum((s - s1)**2)
        label = self.pygame_font.render(str(r), 1, (255, 0, 0))
        label2 = self.pygame_font.render(str(du), 1, (255, 0, 0))
        label_surface = pygame.Surface((84, 84))
        label_surface.blit(label, (0, 0))
        label_surface.blit(label2, (0, 48))

        score = pygame.surfarray.array3d(label_surface).swapaxes(0, 1)
        concat_it = np.concatenate((s, score, s1), axis=1)

        im = Image.fromarray(concat_it.astype('uint8'))
        im.save("test.png")


    def create_training_data(self, clazz, s):
        pass


    def handle_menu(self):

        if not self.has_trained:
            print(self.episode_reward, self.action_distribution, self.q_model.memory.count)
            self.episode_reward = 0
            self.action_distribution = [0 for x in self.action_space]

            # Train for 10 epochs
            for i in range(100):
                self.q_model.train()

            # Set trained flag to true
            self.has_trained = True

        if not self.has_pressed_menu:
            self.py_vnc.send_key("q")
            self.stage_time_dy = None
            self.has_pressed_menu = True

    def on_frame(self):
        pass

    def __init__(self, py_vnc):
        self.py_vnc = py_vnc
        self.py_vnc.send_mouse("Left", (self.py_vnc.screen.size[0], 0))
        self.py_vnc.add_callback(1 / 10, self.on_frame) # 10 FPS
        self.model = StateModel(True)
        self.action_space = [K_LEFT, K_RIGHT, None]
        self.q_model = DQN((84, 84, 3), self.action_space)

        self.EPSILON_DECAY = (self.q_model.EPSILON_END - self.q_model.EPSILON_START) / 10000
        self.image_save_path = os.path.join(dir_path, "..", "state", "unlabeled")

        self.has_trained = False
        self.is_clean = True
        self.has_pressed_menu = False
        self.has_pressed_score = False
        self.been_terminal = False

        self.stage_time_counter = 0
        self.stage_time_dy = None
        self.stage_time_max = 120

        self.action_distribution = None
        self.episode_reward = 0

        # DEbugging stuff
        self.pygame_font = pygame.font.SysFont("monospace", 18)

        while True:
            raw_img = self.render()
            s = self.model.preprocess(raw_img)

            predicted = self.model.predict(s)
            print(predicted)

            if predicted == "menu":
                self.been_terminal = False
                self.handle_menu()

            elif predicted == "prompt":
                self.py_vnc.send_key("x")

            elif predicted in ["terminal_1", "terminal_2"]:
                self.been_terminal = True
                if not self.is_clean:
                    self.q_model.memory.buffer[self.q_model.memory.count - 1][2] = -1
                    #print("setting last in pair to negative")
                    time.sleep(3)
                    self.py_vnc.send_key("q")

                    self.is_clean = True
                    self.has_pressed_menu = False

            elif predicted == "stage" or True:
                if not self.been_terminal:
                    self.is_clean = False
                    self.has_trained = False
                    # 0. Observe (s)
                    # 1. Do action
                    # 2. Observe
                    # 3. Train
                    # 4. set state+1 to state

                    a_idx = self.q_model.act(np.array([s]))
                    a = self.action_space[a_idx]
                    self.action_distribution[a_idx] += 1
                    if a is not None:
                        self.py_vnc.send_key(a, duration=.1)

                    time.sleep(.5)

                    raw_img = self.render()
                    s1 = self.model.preprocess(raw_img)
                    r = 0.01
                    self.episode_reward += r

                    self.q_model.memory.add([s, a_idx, r, s1, False])


                    #Debug stuff
                    self.state_debugger(s, s1, r)

            time.sleep(.1)

    def render(self):
        img = self.py_vnc.screen.get_array()
        img = Image.fromarray(img)
        return img

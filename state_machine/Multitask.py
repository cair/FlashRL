import json
from io import BytesIO
import time
import numpy as np
import psutil
import pywinauto
from PIL import Image
from pywinauto import Application
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from rl.Main import DQN
from state.multitask_one.model import StateModel
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def dispatchKeyEvent(driver, name, options = {}):
    options["type"] = name
    body = json.dumps({'cmd': 'Input.dispatchKeyEvent', 'params': options})
    resource = "/session/%s/chromium/send_command" % driver.session_id
    url = driver.command_executor._url + resource
    driver.command_executor._request('POST', url, body)

def holdKeyW(driver, key, duration):
    endtime = time.time() + duration

    while True:
        dispatchKeyEvent(driver, "rawKeyDown", key)
        dispatchKeyEvent(driver, "char", key)

        if time.time() > endtime:
            dispatchKeyEvent(driver, "keyUp", key)
            break

        key["autoRepeat"] = True
        time.sleep(0.01)

class Multitask:

    def get_pid(self):
        if isinstance(self.driver, webdriver.Chrome):
            process = psutil.Process(self.driver.service.process.pid)
            return process.children()[0].pid              # process id of browser tab
        if isinstance(self.driver, webdriver.Firefox):
            return self.driver.service.process.pid       # process id of browser

    def __init__(self, driver):
        self.driver = driver
        self.model = StateModel(True)
        self.action_space = [Keys.LEFT, Keys.RIGHT]
        self.action_space_chrome = [{
            "code": "KeyLeft",
            "key": "left",
            "text": "left",
            "unmodifiedText": "left",
            "nativeVirtualKeyCode": 37,
            "windowsVirtualKeyCode": 37
        },
            {
                "code": "KeyRight",
                "key": "left",
                "text": "left",
                "unmodifiedText": "left",
                "nativeVirtualKeyCode": 39,
                "windowsVirtualKeyCode": 39
            }]


        self.q_model = DQN((150, 150, 3), self.action_space)
        self.area = self.get_screen_area()
        self.element = self.driver.find_element_by_tag_name('embed')
        self.image_save_path = os.path.join(dir_path, "..", "state", "multitask_one", "images")

        chrome_window_app = Application()
        chrome_window_app.connect(process=self.get_pid())  # connect to browser

        has_trained = False
        is_clean = True
        state_pairs = []

        while True:
            raw_img = self.render()
            s = self.model.preprocess(raw_img)
            predicted = self.model.predict(s)

            if predicted == "menu":
                if not has_trained:
                    # Train xD
                    # Add to exp replay
                    for pair in state_pairs:
                        self.q_model.memory.add(pair)
                    state_pairs = []

                    for i in range(10):
                        self.q_model.train()

                    has_trained = True

                pywinauto.mouse.click(button='left', coords=(300, 700))
            elif predicted == "stage_1_prompt":
                self.element.send_keys(Keys.ENTER)
            elif predicted == "score":
                pywinauto.mouse.click(button='left', coords=(300, 800))
                pywinauto.mouse.click(button='left', coords=(350, 800))
            elif predicted == "terminal":
                if not is_clean:
                    state_pairs[len(state_pairs) - 1][2] = -1
                    print("setting last in pair to negative")
                    is_clean = True

            elif predicted == "stage_1":
                is_clean = False
                has_trained = False
                # 0. Observe (s)
                # 1. Do action
                # 2. Observe
                # 3. Train
                # 4. set state+1 to state
                a_idx = self.q_model.act(np.array([s]))
                a = self.action_space[a_idx]

                print(a)
                holdKeyW(self.driver, self.action_space_chrome[a_idx], .05)


                raw_img = self.render()
                s1 = self.model.preprocess(raw_img)
                r = 0.01

                state_pairs.append([s, a_idx, r, s1, False])




                #print(predicted)
                #raw_img.save(os.path.join(self.image_save_path, "state_%s_%s.png" % (predicted, int(time.time()))))

    def get_screen_area(self):
        element = self.driver.find_element_by_tag_name('embed')
        location = element.location
        size = element.size
        return location["x"], location["y"], size["height"], size["width"]

    def render(self):
        img = self.driver.get_screenshot_as_png()
        img = Image.open(BytesIO(img))
        img = img.crop(self.area)
        return img
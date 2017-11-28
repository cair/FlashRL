import importlib
import os
import threading
import time
import numpy as np
from PIL import Image
from easyprocess import EasyProcess
from pyvirtualdisplay import Display
from pyVNC.Client import Client
from keras.models import load_model
dir_path = os.path.dirname(os.path.realpath(__file__))

class Environment:
    def __init__(self, env_name):
        self.env_name = env_name
        self.path = os.path.join(dir_path, "environments", self.env_name)
        self.env_config = self.load_config()
        self.swf = self.env_config["swf"]
        self.model_path = os.path.join(self.path ,self.env_config["model"])
        self.dataset = self.env_config["dataset"]
        self.action_space = self.env_config["action_space"]
        self.action_names = self.env_config["action_names"]
        self.state_space = self.env_config["state_space"]
        self.model = load_model(self.model_path)

    def load_config(self):
        spec = importlib.util.spec_from_file_location("module.define", os.path.join(self.path, "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.define

    def setup(self, vnc):
        self.vnc = vnc
        self.vnc.send_mouse("Left", (self.vnc.screen.size[0], 0))
        self.vnc.add_callback(1 / 10, self.on_frame) # 10 FPS

    def preprocess(self, pil_image):
        img = pil_image.resize((self.state_space[0], self.state_space[1]), Image.ANTIALIAS)
        img = img.convert('RGB')
        return np.array(img)

    def render(self):
        img = self.vnc.screen.get_array()
        img = Image.fromarray(img)
        arr_img = self.preprocess(img)
        return np.array([arr_img])

    def on_frame(self):
        s = self.render()
        state_type = self.action_names[np.argmax(self.model.predict(s))]
        print(state_type)

        



class GameEnvironment(threading.Thread):
    def __init__(self, display, env):
        threading.Thread.__init__(self)
        #super(self)
        self.display = display
        self.env = env

    def run(self):
        self.vnc(self.display)

    def vnc(self, vnc_display):
        os.environ["DISPLAY"] = vnc_display
        with Display(backend='xvnc', rfbport=5902, size=(223, 150)) as disp:
            with EasyProcess(' '.join(['gnash', os.path.join(self.env.path, self.env.swf), "--width", "150", "--height", "150"])) as proc:
                proc.wait()


class Handler:
    def __init__(self):
        original_display = os.environ["DISPLAY"]
        vnc_display = ":98"
        print("Initialize xVNC")
        print("---------------")
        print("Display: %s" % original_display)
        print("VNC Display: %s" % vnc_display)

        environment = "multitask"

        env = Environment(environment)

        x_vnc = GameEnvironment(vnc_display, env)
        x_vnc.start()

        time.sleep(1)

        os.environ["DISPLAY"] = original_display
        py_vnc = Client(host="127.0.0.1", port=5902, gui=True, array=True)
        py_vnc.start()

        time.sleep(1)

        env.setup(py_vnc)

import os
import numpy as np
from PIL import Image
import importlib
from keras.models import load_model
dir_path = os.path.dirname(os.path.realpath(__file__))

class Environment:
    def __init__(self, env_name, fps=10, frame_callback=None, grayscale=False, normalized=False):
        self.fps = fps
        self.grayscale = grayscale
        self.normalized = normalized
        self.frame_callback = frame_callback
        self.env_name = env_name
        self.path = os.path.join(dir_path, "..", "contrib", "environments", self.env_name)

        if not os.path.isdir(self.path):
            self.path = os.path.join("contrib", "environments", self.env_name)

            if not os.path.isdir(self.path):
                raise FileExistsError("The specified environment \"%s\" could not be found." % self.env_name)

        self.env_config = self.load_config()
        self.swf = self.env_config["swf"]
        self.model_path = os.path.join(self.path ,self.env_config["model"])
        self.dataset = self.env_config["dataset"]
        self.action_space = self.env_config["action_space"]
        self.action_names = self.env_config["action_names"]
        self.state_space = self.env_config["state_space"]

        try:
            self.model = load_model(self.model_path)
        except OSError as e:
            print("No state prediction!")
            self.model = None
            """# Missing model, prompt for collecting training data
            ynq = None
            while ynq not in ["y", "n", "q"]:
                ynq = input("State prediction model is missing. Collect training data? (Q for quit): ").lower()
            
            if ynq == "n" or ynq == "q":
                print("Exiting!")
                exit(0)
            else:
                print("Starting Game Mode, Collecting unlabeled images")
            """


    def load_config(self):
        spec = importlib.util.spec_from_file_location("module.define", os.path.join(self.path, "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.define

    def setup(self, vnc):
        self.vnc = vnc
        self.vnc.send_mouse("Left", (self.vnc.screen.size[0], 0))
        self.vnc.add_callback(1 / self.fps, self.on_frame)

    def preprocess(self, pil_image):
        img = pil_image.resize((self.state_space[0], self.state_space[1]), Image.ANTIALIAS)
        if self.grayscale:
            img = img.convert("L")
        else:
            img = img.convert('RGB')
        data = np.array(img)

        if self.normalized:
            data = data / 255

        return data

        # NN-Tr XD lets go

    def render(self):
        img = self.vnc.screen.get_array()
        img = Image.fromarray(img)
        arr_img = self.preprocess(img)
        return np.array([arr_img])

    def on_frame(self):
        state = self.render()
        state_type = None
        if self.model:
            state_type = self.action_names[np.argmax(self.model.predict(state))]

        if self.frame_callback:
            self.frame_callback(state, state_type, self.vnc)
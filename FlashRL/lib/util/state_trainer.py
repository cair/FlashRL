import glob
import os
import importlib.util
import pickle
from model import Model
dir_path = os.path.dirname(os.path.realpath(__file__))

class Trainer:

    def __init__(self, environment_path):
        self.env_path = environment_path
        self.module_path = os.path.join(self.env_path, "__init__.py")

        self.env_config = None
        self.env_training_data = None

        self.load_environment_config()
        self.load_training_data()

        self.env_model_path = os.path.join(self.env_path, self.env_config["model"])

    def load_environment_config(self):
        spec = importlib.util.spec_from_file_location("module.define", self.module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.env_config = mod.define

        if self.env_config is None:
            raise RuntimeError("Environment configuration is empty!")

    def load_training_data(self):
        self.env_training_data = pickle.load(open(os.path.join(self.env_path, self.env_config["dataset"]), "rb"))

    def train(self):
        m = Model(self.env_training_data, self.env_model_path)
        m.train()
        

if __name__ == "__main__": 

    for env in glob.glob(os.path.join(dir_path, "..", "environments/*")):
        trainer = Trainer(env)
        trainer.train()
        
import pickle
from state.multitask_one.model import StateModel
import os
if __name__ == "__main__":
    model = StateModel()

    data = pickle.load(open(os.path.join(os.getcwd(), "dataset.p"), "rb"))

    model.train_epoch(data[0], data[1])





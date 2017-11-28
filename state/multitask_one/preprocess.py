from PIL import Image
import numpy as np
import pickle
from state.multitask_one.model import StateModel
import os
if __name__ == "__main__":
    model = StateModel()

    image_class_path = os.path.join(os.getcwd(), "images", "training")
    image_classes = [x for x in os.listdir(image_class_path)]
    image_classes_path = [os.path.join(image_class_path, x) for x in image_classes]
    n_classes = len(image_classes)
    print(n_classes)
    X = []
    Y = []
    for i, class_path in enumerate(image_classes_path):
        files = [os.path.join(class_path, x) for x in os.listdir(class_path)]

        for image_path in files:
            img = Image.open(image_path)

            # Preprocess image
            np_img = model.preprocess(img)

            # Create class label
            y = np.zeros(shape=(n_classes, ))
            y[i] = 1

            # Add to dataset
            X.append(np_img)
            Y.append(y)

    pickle.dump((np.array(X), np.array(Y)), open(os.path.join(os.getcwd(), "dataset.p"), "wb"))




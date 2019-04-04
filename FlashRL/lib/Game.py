import os
import time
from pyVNC.Client import Client
from .GameEnvironment import GameEnvironment
from .Environment import Environment

class Game:
    def __init__(self, environment_name, fps=10, frame_callback=None, grayscale=False, normalized=False):
        original_display = os.environ["DISPLAY"]
        vnc_display = ":98"
        print("Initialize xVNC")
        print("---------------")
        print("Display: %s" % original_display)
        print("VNC Display: %s" % vnc_display)

        env = Environment(environment_name, fps=fps, frame_callback=frame_callback, grayscale=grayscale, normalized=normalized)

        x_vnc = GameEnvironment(vnc_display, env)
        x_vnc.start()

        time.sleep(1)

        os.environ["DISPLAY"] = original_display
        py_vnc = Client(host="127.0.0.1", port=5902, gui=True, array=True)
        py_vnc.start()

        time.sleep(1)

        env.setup(py_vnc)

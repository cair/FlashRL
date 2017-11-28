import os
import threading
import time
from easyprocess import EasyProcess
from pyVNC.Client import Client
from pyvirtualdisplay import Display

from state_machine.Multitask import Multitask

dir_path = os.path.dirname(os.path.realpath(__file__))


def vnc(vnc_display):
    os.environ["DISPLAY"] = vnc_display
    with Display(backend='xvnc', rfbport=5902, size=(223, 150)) as disp:
        with EasyProcess(' '.join(['gnash', os.path.join(dir_path, "environments", "multitaskgame.swf"), "--width", "150", "--height", "150"])) as proc:
            proc.wait()


class Handler:
    def __init__(self):
        original_display = os.environ["DISPLAY"]
        vnc_display = ":98"
        print("Initialize xVNC")
        print("---------------")
        print("Display: %s" % original_display)
        print("VNC Display: %s" % vnc_display)
        x_vnc = threading.Thread(target=vnc, args=(vnc_display, ))
        x_vnc.start()

        time.sleep(1)

        os.environ["DISPLAY"] = original_display
        py_vnc = Client(host="127.0.0.1", port=5902, gui=False, array=True)
        py_vnc.start()

        time.sleep(1)

        Multitask(py_vnc)


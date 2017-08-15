import threading

from handler import Handler
from server import app
import sys


if __name__ == "__main__":
    #t1 = threading.Thread(target=app.run)
    #t1.start()

    sys.setrecursionlimit(999999999)
    h = Handler()



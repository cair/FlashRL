import threading

from handler import Handler
import sys

from server import serve

if __name__ == "__main__":
    print("Starting Webserver")
    web_thread = threading.Thread(target=serve)
    web_thread.daemon = False
    web_thread.start()
    try:
        h = Handler()
    except Exception as e:
        print(e)
    web_thread.join()



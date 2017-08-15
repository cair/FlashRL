import os

from selenium import webdriver
from selenium.webdriver.opera.options import Options
from selenium.webdriver.opera.options import DesiredCapabilities
from selenium.webdriver.

from settings import settings
from state_machine.Multitask import Multitask
dir_path = os.path.dirname(os.path.realpath(__file__))


class Handler:
    def __init__(self):
        # sudo pip3 install selenium==3.3.3
        # sudo apt-get install firefox=50.1.0+build2-0ubuntu1
        # geckodriver 16.1
        print("Initialize PhantomJS")

        self.driver = webdriver.Chrome()
        self.driver.get("http://localhost:8082/environments/multitaskgame.html")

        #self.driver.get(settings["multitask_1-v0"]["url"])

        Multitask(self.driver)

    def exit(self):
        self.driver.close()
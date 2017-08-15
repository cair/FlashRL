import json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


import os

from settings import settings
from state_machine.Multitask import Multitask

dir_path = os.path.dirname(os.path.realpath(__file__))


class Handler:
    def __init__(self):
        print("Initialize PhantomJS")
        chrome_path = "chromedriver.exe"
        self.driver = webdriver.Chrome(executable_path=chrome_path)
        self.driver.get(settings["multitask_1-v0"]["url"])

        Multitask(self.driver)

    def exit(self):
        self.driver.close()
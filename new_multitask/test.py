import time
from io import BytesIO

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys



from selenium.webdriver.firefox.options import Options

options = Options()
options.set_preference("dom.ipc.plugins.enabled.libflashplayer.so","true")
options.set_preference("plugin.state.flash", 2)

driver = webdriver.Firefox(firefox_options=options)
driver.get("http://localhost:8000/swf.html")

element = driver.find_element_by_tag_name('embed')
rect = element.rect
points = [rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height']]

while True:
    start = time.time()
    img = Image.open(BytesIO(driver.get_screenshot_as_png()))
    img = img.crop(points)

    img.save("test.png")
    end = time.time()
    print("PNG: %s s" % ((end - start) * 1000))

    """start = time.time()
    driver.get_screenshot_as_png()
    end = time.time()
    print("PNG %s s" % (end - start))"""






driver.close()
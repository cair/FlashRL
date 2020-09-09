# FlashRL - Flash Platform for Reinforcement Learning
**Note: This version of FlashRL has been forked to experiment with non-Keras ML frameworks**

For the original version of FlashRL, go to [this link](https://github.com/cair/FlashRL).


This repository is forked from the original FlashRL repository.
The goal of this is to repository is to support our efforts in Clemson University's MATH 4990 Creative Inquiry class during Fall Semester 2020.
Our goal is to use FlashRL to run our own Flash games and train reinforcement learning algorithms by using the Palmetto Cluster and NVIDIA DGX-2.

# TODO List
* Fix pyVNC issue. Currently pyVNC fails to start a VNC server for the game to run on. We need to solve this issue in order to run our games in headless mode.
* Begin developing custom environments.
* Begin developing Docker containers for our code to run in. Preferably, create a Dockerfile that can be used to run custom environments without the need for the local machine to have all the dependencies installed.

# Prerequisites
* Linux based operating system (Ubuntu 17.04 and 17.10 are tested)
* Python 3.x.x (3.5 and 3.6 are tested)
* gnash
* xvfb

# Installation
For our testing, we have been working in a python virtual environment.
```python
pip install git+https://github.com/cair/pyVNC
pip install git+https://github.com/JDaniel41/FlashRL
```

# Deploy new environment
Developers are able to import custom environments through ```project/contrib/environments/```

A typical custom implementation looks like this:
```python
- project
    - __init__.py
    - main.py
    - contrib
        - environments
            - env_name
                - __init__.py
                - dataset.p
                - model.h5
                - env.swf

```
in the following section, we demonstrate how to implement the flash game Mujaffa as an environment for FlashRL.

## Mujaffa-1.6
### Prerequisites
* SWF Game File
* Python 3x
* Keras

###
*  Create directory structure ```mkdir -p contrib/environments/mujaffa-v1.6```
*  Create Configuration file:  
```python
echo "define = {
    "swf": "mujaffa.swf",
    "model": "model.h5",
    "dataset": "dataset.p",
    "scenes": [],
    "state_space": (84, 84, 3)
}" > contrib/environments/mujaffa-v1.6/__init__.py
```

* Add swf "mujaffa.swf" to ```contrib/environments/mujaffa-v1.6/```
* Create file ```main.py in project root``` with following template

```
from FlashRL import Game

def on_frame(state, type, vnc):
    # vnc.send_key("a") # Sends the key "a"
    # vnc.send_mouse("Left", (200, 200)) # Left Clicks at x=200, y=200
    # vnc.send_mouse("Right", (200, 200)) # Right Clicks at x=200, y=200
    pass

g = Game("mujaffa-v1.6", fps=10, frame_callback=on_frame, grayscale=True, normalized=True)
```


# Licence
Copyright 2017/2018 Per-Arne Andersen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

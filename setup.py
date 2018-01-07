from setuptools import setup
import os



setup(
    name='FlashRL',    # This is the name of your PyPI-package.
    version='1.0',                          # Update the version number for new releases
    install_requires=['numpy', 'pygame', "pyVNC", "pillow", "scipy", "easyprocess", "pyvirtualdisplay", "keras", "h5py"],
    packages=["FlashRL.lib", "FlashRL.lib.util", "FlashRL.contrib", "FlashRL.contrib.environments", "FlashRL.contrib.environments.multitask],
    scripts=[],
    dependency_links=[
        "git+ssh://git@github.com:UIA-CAIR/pyVNC.git"
    ]
)

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup

setup(
   name = "treelearn",
   version = "0.0.4",
   #package_dir = { '' : 'treelearn' },
   packages = ['treelearn'],
   install_requires = [ 'scikit-learn' ],
   license = "LGPL",
   keywords = "machine learning tree forest random",
   url = "https://github.com/capitalk/treelearn",
   classifiers=[
     "Development Status :: 3 - Alpha",
     "Topic :: Utilities",
     "License :: OSI Approved :: LGPL License",
   ],
)

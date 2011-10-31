from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup

setup(
   name = "treelearn",
   version = "0.0.3",
   package_dir = { '' : 'treelearn' },
   packages = ['treelearn'],
   install_requires = [ 'scikit-learn' ],
   license = "LGPL",
   keywords = "machine learning tree forest random",
   url = "https://github.com/capitalk/treelearn",
   long_description=read('README.md'),
   classifiers=[
     "Development Status :: 3 - Alpha",
     "Topic :: Utilities",
     "License :: OSI Approved :: LGPL License",
   ],
)

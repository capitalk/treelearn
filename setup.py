import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "treelearn",
    version = "0.0.1",
    author = "Alex Rubinsteyn",
    author_email = "alex.rubinsteyn[~at~]gmail[~dot~]com",
    description = ("Machine learning with trees and forests"), 
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

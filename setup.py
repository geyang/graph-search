from os import path
from setuptools import setup, find_packages

with open(path.join(path.abspath(path.dirname(__file__)), 'VERSION'), encoding='utf-8') as f:
    version = f.read()

setup(name='graph_search',
      packages=find_packages(),
      install_requires=[
          "jupyter",
          "matplotlib",
          "ml-logger",
          "networkx",
          "numpy",
          "pandas",
          "waterbear",
      ],
      description='graph-search',
      author='Ge Yang<ge.ike.yang@gmail.com>',
      url='https://github.com/geyang/graph_search',
      author_email='ge.ike.yang@gmail.com',
      version=version)

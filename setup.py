from setuptools import setup, find_packages

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
      author='Ge Yang<yangge1987@gmail.com>',
      url='https://github.com/episodeyang/graph_search',
      author_email='yangge1987@gmail.com',
      version='0.0.0')

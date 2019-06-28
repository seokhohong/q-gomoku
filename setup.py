import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name='q-gomoku',
	version='0.0.2',
	packages=['qgomoku','qgomoku.core','qgomoku.learner', 'qgomoku.util']
)

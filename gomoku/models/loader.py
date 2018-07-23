import os

def get_file(filename):
	return os.path.join(os.path.dirname(__file__), filename)

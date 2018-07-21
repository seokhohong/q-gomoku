import os
 
from setuptools import setup, find_packages 
 
 
setup(
    name="gomoku",
    author="Seokho Hong",
    author_email="hseokho@cisco.com",
    url="https://wwwin-github.cisco.com/GVS-CS-DSX/package_repo",
    python_requires=">3",
    install_requires=[],
    package_directory={"", "src"},
    include_package_data=True,
    packages=['gomoku'],
)

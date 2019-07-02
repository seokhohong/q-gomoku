from setuptools import setup

setup(
    name="gomoku2",
    python_requires=">3",
    install_requires=[],
    include_package_data=True,
    packages=["core", 'main', 'learner', 'util'],
)

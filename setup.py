from setuptools import setup, find_namespace_packages

setup(name='gym_molecule',
      version='0.0.1',
      description="A package for RL based drug molecule generation",
      author="Soo Kyung Ahn",
      author_email="sahnsookyung@gmail.com",
      packages=['gym_molecule'],
      install_requires=['gym', 'numpy', 'rdkit']
      )

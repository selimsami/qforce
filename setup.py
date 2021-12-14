from setuptools import setup, find_packages

setup(
      packages = find_packages(),
      package_data={'qforce': ['data/*']},
      entry_points = {
        'console_scripts': ['qforce=qforce.main:run',]
      },
     )

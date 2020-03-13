from setuptools import setup

setup(
      packages = ['qforce', 'qforce/molecule', 'qforce/modified_seminario', 'colt'],
      package_data = {'qforce': ['data/*']},
      scripts = ['bin/qforce'], 
     )


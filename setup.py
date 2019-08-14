from setuptools import setup

setup(
      packages = ['qforce', 'qforce/modified_seminario'] ,
      package_data = {'qforce': ['data/*']},
      scripts = ['bin/qforce'], 
     )


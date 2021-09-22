from setuptools import setup, find_packages

setup(
      packages = find_packages(),
      package_data={'qforce': ['data/*']},
      scripts=['bin/qforce', 'bin/morse_to_morse_mp'],
     )


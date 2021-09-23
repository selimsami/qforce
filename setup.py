from setuptools import setup, find_packages

setup(
      packages = find_packages(),
      package_data={'qforce': ['data/*']},
      scripts=[
          'bin/qforce',
          'bin/morse_to_morse_mp',
          'bin/morse_to_morse_mp2',
          'bin/morse_mp_to_morse_mp2',
          'bin/morse_mp2_to_morse_mp',
      ],
     )


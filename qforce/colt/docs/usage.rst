=====
Usage
=====

The core motivation for colt was to provide an simple
way to read (and write) input files, while validating
the user input for type correctness.
This enables several unique features:

1. Simple generation of commandline interfaces
2. Initialize classes from commandline/input files
3. Build complex input files directly from a class hierarchy


Questions
---------

The key component for colt are its question syntax, which
are in type annotated config-file::

  # This is an integer larger equal 2
  integer = :: int :: >2
  # This is a general float
  float = :: float
  # This is a string, with default value "Hello World"
  string = Hello World :: str
  # List of strings
  list = :: list

The syntax hereby is::

  default :: typ :: selection 

comments describing the elements can be added on top of the values 
using '#' as comment character. Multiple comments can be used.

Example: Commandline Parser
---------------------------

One simple example to use colt for is to create simple commandline 
interfaces using the `FromCommandline`-decorator.
An example code that prints values between `xstart` and `xstop` 
using optionally `step` as 

.. literalinclude:: ../examples/commandline_xrange.py
   :linenos:

which generates the following commandline interface if its called with arguments.
**Note:** The function is only called regulary if its called with arguments,
in case of a function that uses no arguments, **always** the argparser will be called!
   
::

   usage: commandline_xrange.py [-h] [-step step] xstart xstop

   positional arguments:
      xstart      int, Range(>0)
                  start of the range
      xstop       int, Range(>1)
                  end of the range

   optional arguments:
      -h, --help  show this help message and exit
      -step step  int,
                  step size

The decorated function can still be used as it is:

.. code-block:: python

   >>> from commandline_xrange import x_range
   >>> x_range(10, 100, 10)
   ... 10
   ... 20
   ... 30
   ... 40
   ... 50
   ... 60
   ... 70
   ... 80
   ... 90


Colt: 
-----

Python is primary an object-orientated programming language,
and colt provides a class to initialize objects directly from
an input file or from the commandline.

.. literalinclude:: ../examples/basic_colt.py
   :linenos:

::

  usage: basic_colt.py [-h] [-options::do_force do_force] natoms nstates

  positional arguments:
      natoms                int, Range(>1)
      nstates               int, Range(>1)

  optional arguments:
      -h, --help            show this help message and exit
      -options::do_force do_force
                            bool, 


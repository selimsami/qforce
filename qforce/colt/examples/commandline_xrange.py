# examples/commandline_xrange.py
from colt import from_commandline


@from_commandline("""
# start of the range
xstart = :: int :: >0
# end of the range
xstop = :: int :: >1
# step size
step = 1 :: int 
""")
def x_range(xstart, xstop, step):
    for i in range(xstart, xstop, step):
        print(i)

if __name__ == '__main__':
    x_range()

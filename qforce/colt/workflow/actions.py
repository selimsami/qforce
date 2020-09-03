class Action:
    """Basic Action object to wrap functions and add types"""

    __slots__ = ('_func', 'arg_types', 'kwarg_types', 'return_typ', 'nargs')

    def __init__(self, func, arg_types, kwarg_types, return_typ, need_self=False):
        #
        if need_self is False:
            func = with_self(func)
        self._func = func
        self.arg_types = tuple(arg_types)
        self.kwarg_types = kwarg_types
        self.nargs = len(arg_types)
        self.return_typ = return_typ

    def __call__(self, workflow, args, kwargs):
        return self._func(workflow, *args, **kwargs)


class IteratorAction(Action):
    """Loop over an iterator"""

    __slots__ = ('iterator_id', 'use_progress_bar')

    def __init__(self, func, arg_types, kwarg_types, return_typ, need_self=False,
                 iterator_id=0, use_progress_bar=False):
        super().__init__(func, arg_types, kwarg_types, return_typ, need_self=need_self)
        self.use_progress_bar = use_progress_bar
        self.iterator_id = iterator_id

    def __call__(self, workflow, args, kwargs):
        out = {}
        #
        if self.use_progress_bar:
            iterator = ProgressBar(args[self.iterator_id], len(args[self.iterator_id]))
        else:
            iterator = args[self.iterator_id]
        #
        for ele in iterator:
            out[ele] = self._func(workflow, *args[:self.iterator_id], ele,
                                  *args[self.iterator_id+1:], **kwargs)
        return out


class ProgressBar:
    """Basic Class to handle progress"""

    def __init__(self, iterator, nele, width=80):
        self.iterator = iterator
        self.nele = nele
        self.width = width

    def progress_bar_string(self, i):
        icurrent = int(i/self.nele * self.width)
        _bar = "="*icurrent + ' '*(self.width - icurrent)
        return f'Progress: [{_bar}] {round(icurrent*100/self.width, 2)}%'

    def __iter__(self):
        try:
            for i, ele in enumerate(self.iterator):
                print(f'\r{self.progress_bar_string(i)}', end='')
                yield ele
            print(f'\r{self.progress_bar_string(self.nele)}', end='')
        finally:
            print()


def with_self(func):
    def _wrapper(self, *args, **kwargs):
        return func(*args, **kwargs)
    return _wrapper

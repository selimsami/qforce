class TermSelector:

    def __init__(self, options, offterm):
        if 'off' not in options:
            options['off'] = offterm
        self._options = options

    def get_factory(self, value):
        factory = self._options.get(value, None)
        if factory is not None:
            return factory
        raise ValueError(f"Do not know term factory '{value}'")


class SingleTermSelector:

    def __init__(self, term):
        self._term = term

    def get_factory(self, value):
        if value == 'on':
            return self._term
        raise ValueError(f"Do not know term factory '{value}'")


def to_selector(dct, offterm):
    """Convert the dictory to a dict of term selectors"""
    out = {}

    for name, options in dct.items():
        # probably should be mapping
        if isinstance(options, dict):
            out[name] = TermSelector(options, offterm)
        else:
            out[name] = SingleTermSelector(options)
    return out

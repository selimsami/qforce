class ForcefieldSettings:

    _always_on_terms = []

    _optional_terms = {}

    _term_types = {}

    @classmethod
    def get_questions(cls):
        tpl = '# Turn {key} FF term on or off\n{key} = {default} :: bool\n\n'
        tplset = '# The {key} FF term is always on\n{key} = {default} :: bool :: [True]\n\n'
        tpltypes = '# \n{key}_type = {default} :: str :: [{options}] \n\n'
        #
        questions = ''
        for name, (default, options) in cls._term_types.items():
            questions += tpltypes.format(key=name, default=default, options=', '.join(options))
        for name in cls._always_on_terms:
            questions += tplset.format(key=name, default='True')
        for name, default in cls._optional_terms.items():
            questions += tpl.format(key=name, default=default)

        return questions

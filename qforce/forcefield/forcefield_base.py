class ForcefieldSettings:

    _always_on_terms = []

    _optional_terms = {}

    @classmethod
    def get_questions(cls):
        tpl = '# Turn {key} FF term on or off\n{key} = {default} :: bool\n\n'
        tplset = '# The {key} FF term is alwayson\n{key} = {default} :: bool :: [True]\n\n'
        #
        questions = ''
        for name in cls._always_on_terms:
            questions += tplset.format(key=name, default='True')
        for name, default in cls._optional_terms.items():
            questions += tpl.format(key=name, default=default)
        return questions

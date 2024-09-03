from colt.validator import Validator


def boolean(string):
    string = string.lower()
    if string in ['off', 'no', 'n', 'false', 'f']:
        return 'off'
    if string in ['on', 'yes', 'y', 'true', 't']:
        return 'on'
    raise ValueError(f"do not know boolean value {string}")


def options(string):
    string = string.lower()
    if string in ['off', 'no', 'false', 'f']:
        return 'off'
    return string


Validator.add_validator("boolean", boolean)
Validator.add_validator("options", options)


class ForcefieldSettings:

    _always_on_terms = {}

    _optional_terms = {}

    @classmethod
    def terms(cls):
        return list(cls._always_on_terms) + list(cls._optional_terms)

    @classmethod
    def get_questions(cls):

        def select_options(name, options, always_on=False):

            def join(options):
                if isinstance(options, str):
                    options = [options]
                return ', '.join(str(val) for val in options)

            if always_on is True:
                if isinstance(options, bool):
                    return '# Turn {key} FF term is always turned on\n{key} = True :: boolean :: [True,]\n\n'.format(key=name)
                if isinstance(options, str):
                    return '# FF {key} is always set to {default}\n{key} = {default} :: options :: {options}\n\n'.format(key=name, default=options, options=join(options))
                # remove False from options
                options = [key for key in options if key is not False]
                if len(options) == 0:
                    return '# Turn {key} FF term is always turned on\n{key} = True :: boolean :: [True,]\n\n'.format(key=name)
                return '# FF {key} \n{key} = {default} :: options :: {options}\n\n'.format(key=name, default=str(options[0]), options=join(options))

            if isinstance(options, bool):
                return '# Turn {key} FF term on or off\n{key} = {default} :: boolean\n\n'.format(key=name, default=options)
            if isinstance(options, str):
                options = [options] + [False]
            else:
                if False not in options:
                    options = list(options) + [False]
            return '# FF {key} can be off or one of the following options\n{key} = {default} :: options :: {options}\n\n'.format(key=name, default=str(options[0]), options=join(options))
        #
        questions = ''
        for name, options in cls._always_on_terms.items():
            questions += select_options(name, options, always_on=True)
        for name, options in cls._optional_terms.items():
            questions += select_options(name, options, always_on=False)

        return questions

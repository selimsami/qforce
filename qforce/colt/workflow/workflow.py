import inspect
from collections import UserDict
#
from ..validator import Validator
from ..commandline import get_config_from_commandline
#
from .language import Assignment, Parser, Type, Variable
from .actions import Action, IteratorAction


class ActionContainer(UserDict):
    """Container to store actions to prevent overwriting of them!"""

    def __init__(self, dct=None):
        super().__init__()
        if dct is None:
            dct = {}
        self.data = dct

    def __setitem__(self, key, value):
        if key in self:
            raise Exception("Should not overwrite action!")
        self.data[key] = value


class WorkflowGenerator:
    """Workflow engine, stores all actions"""

    def __init__(self):
        self.actions = ActionContainer()

    def register_action(self, *args, need_self=False, iterator_id=None, progress_bar=False):
        #
        func, need_self, iterator_id, progress_bar = self._parse_args(args, need_self,
                                                                      iterator_id, progress_bar)

        def _wrapper(func):
            name = func.__name__
            #
            arg_types, kwarg_types, return_typ = get_signiture(func)
            if iterator_id is None:
                self.actions[name] = Action(func, arg_types, kwarg_types, return_typ,
                                            need_self=need_self)
            else:
                self.actions[name] = IteratorAction(func, arg_types, kwarg_types, return_typ,
                                                    iterator_id=iterator_id, need_self=need_self,
                                                    use_progress_bar=progress_bar)
            return func

        if func is not None:
            return _wrapper(func)
        return _wrapper

    @staticmethod
    def _parse_args(args, need_self, iterator_id, progress_bar):
        if len(args) > 3:
            raise Exception(f"function takes 3 arguments {len(args)} provided")

        if len(args) == 1:
            if inspect.isfunction(args[0]):
                return args[0], need_self, iterator_id, progress_bar
            need_self = args[0]
        elif len(args) == 2:
            need_self, iterator_id = args
        elif len(args) == 3:
            need_self, iterator_id, progress_bar = args
        return None, need_self, iterator_id, progress_bar

    def create_workflow(self, name, nodes, add_workflow=False, output=None):
        workflw = Workflow(name, nodes, self.actions)
        if add_workflow is True:
            func, arg_types, return_typ = workflw.get_function(output=output)
            self.actions[name] = Action(func, arg_types, {}, return_typ)
        return workflw

    @staticmethod
    def generate_workflow_file(filename, name, workflow, module, engine):
        with open(filename, 'w') as fhandle:
            fhandle.write(generate_workflow(name, workflow, module, engine))

    @staticmethod
    def add_subtypes(parent, subtypes):
        Type.add_subtypes(parent, subtypes)


def generate_workflow(name, workflow, module, engine):
    workflow = f'"""{workflow}"""'
    return f"""
from {module} import {engine}

workflow = {engine}.create_workflow('{name}', {workflow})

if __name__ == '__main__':
    workflow.run()
"""


class WorkflowExit(Exception):
    pass


class Workflow:
    """Workflow object used"""

    def __init__(self, name, string, actions):
        self.name = name
        self.parser = Parser(actions)
        self.nodes = self._parse_string(string)
        self.input_nodes, self.types = self._check_types()

    def _check_types(self):
        types = {}
        input_nodes = {}
        for node in self.nodes:
            node.check_types(types, input_nodes)
        for var, typ in input_nodes.items():
            if typ.typ.typ not in Validator.parsers:
                raise Exception(f"cannot get type '{typ}' of variable {var} from commandline")
        return input_nodes, types

    def _input_questions(self, data):
        txt = ""
        for name, value in self.input_nodes.items():
            if name in data:
                continue
            if value.value is None:
                _value = ''
            else:
                _value = str(value.value)
            if value.comment is not None:
                txt += f"#{value.comment}\n"
            txt += f"{name} = {_value} :: {value.typ.typ}\n"
        return txt

    def _parse_string(self, string):
        out = []
        for inum, line in self.string_itr(string):
            res, _ = self.parser.match_line(line)
            if res is None:
                raise Exception(f"Cannot parse line {inum}:\n{line}")
            out.append(res)
        return out

    @staticmethod
    def error(msg):
        raise WorkflowExit

    @staticmethod
    def debug(msg):
        print("Debug: ", msg)

    @staticmethod
    def info(msg):
        print("Workflow: ", msg)

    @staticmethod
    def string_itr(string, comment='#'):
        for i, line in enumerate(string.splitlines()):
            line = line.partition(comment)[0].strip()
            if line != '':
                yield i, line

    def get_function(self, output=None):
        # name of the keys
        keys = list(self.input_nodes.keys())
        if output is not None:
            return_typ = self.types[output]
        else:
            return_typ = None
        #
        arg_types = tuple(value.typ for value in self.input_nodes.values())

        def _func(*args):
            if len(args) != len(keys):
                raise Exception("")
            data = {key: value for key, value in zip(keys, args)}
            data = self._run(data)
            if output is not None:
                return data[output]
        return _func, arg_types, return_typ

    def run(self, data=None, description=None):
        """execute workflow"""
        if data is None:
            data = {}
        questions = self._input_questions(data)
        if questions != '':
            data.update(get_config_from_commandline(questions, description=description))
        return self._run(data)

    def _run(self, data=None):
        """execute workflow"""
        if data is None:
            data = {}
        #
        for node in self.nodes:
            try:
                if isinstance(node, Assignment):
                    data[node.name] = node.call(self, data)
                else:
                    node.call(self, data)
            except WorkflowExit:
                break
        return data


def get_signiture(func):
    position_arguments = []
    keyword_arguments = {}
    #
    if not hasattr(func, '__annotations__'):
        raise Exception("Type annotations need to be set")
    #
    return_typ = check_typ(func.__annotations__.get('return', None))
    #
    for name, value in inspect.signature(func).parameters.items():
        if name == 'self':
            continue
        if value.annotation is value.empty:
            raise Exception("All types need to be annotated")
        typ = check_typ(value.annotation)
        #
        if value.default is value.empty:
            position_arguments.append(typ)
        else:
            keyword_arguments[name] = Variable(value.default, typ)
    return position_arguments, keyword_arguments, return_typ


def check_typ(typ):
    if not isinstance(typ, str) and typ is not None:
        raise Exception("Type annotations need to be strings")
    return Type(typ)


# Primitives
Type('str')
Type('bool')
Type('int')
Type('float')
# Numbers
Type('number')
Type.add_subtypes('number', ['int', 'float'])
# Lists
Type('list')
Type('ilist', alias=['ilist_np'])
Type('flist', alias=['flist_np'])
#
Type.add_subtypes('list', ['ilist', 'flist'])
# Files
Type('file')
Type('existing_file')
Type('non_existing_file')
Type.add_subtypes('file', ['existing_file', 'non_existing_file'])
# Folder
Type('folder')
Type('existing_folder')
Type('non_existing_folder')
Type.add_subtypes('folder', ['existing_folder', 'non_existing_folder'])

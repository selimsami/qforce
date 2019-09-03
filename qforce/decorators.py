from functools import wraps
import types
import time

_total_times = {}

def log(func):

    name = func.__name__

    @wraps(func)
    def _wrapper(*args, **kwargs):
        print("Entering Function '%s'" % name)
        result = func(*args, **kwargs)
        print("Leaving Function '%s'" % name)
        return result
    return _wrapper



def timeit(func):

    global _total_times

    name = func.__name__
    _total_times[name] = [0.0, 0]

    @wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        # count total costs
        _total_times[name][0] += end-start
        # count times called
        _total_times[name][1] += 1
        return result
    return _wrapper


def decorate_all_members(*decorators):

    def _cls_wrapper(cls):
        # save names and pointer to function
        function_names = [] 
        functions = []
        # 
        for name in dir(cls): 
            attribute = getattr(cls, name)
            # get function and no dunder methods!
            if isinstance(attribute, types.FunctionType):
                # add class to function name
                attribute.__name__ = cls.__name__ + "_" + attribute.__name__
                # create decorated function
                for decorator in decorators:
                    attribute = decorator(attribute)
                # save function and function_name
                functions.append(attribute)
                function_names.append(name)

        # modify the attributes of the class
        for i, name in enumerate(function_names):
            setattr(cls, name, functions[i])
        # return the class
        return cls
    return _cls_wrapper


def print_timelog():
    global _total_times

    print("Function call log:")
    form = "%20s: time = %12.8f ncall = %d " 

    for name, (time, ncalled) in _total_times.items():
        print(form % (name, time, ncalled))


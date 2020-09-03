"""Load Plugins from a given folder, basically calls `import module` for all modules
in the plugin folder, that are not explicitly ignored"""
import importlib.util
import os
import re
import sys


class PluginLoader:
    """Plugin Loader for Colt's Plugin system"""
    __slots__ = ('ignore',)

    def __init__(self, folder, *, ignorefile=None):
        """Setup a Plugin Loader for Colt's Plugin system
        Parameters
        ----------
        folder: str
            name of the plugin folder to be loaded

        ignorefile: str, optional
            if given, load pattern from ignorefile to load only specific files

        Examples
        --------
        A basic `ignorefile` can look like the following, the syntax is adopted
        from gitignore files. Only a minimum of pattern matching is supported!

        ----------------------------------------
        unimportant_*.py # Ignore all python files that start with unimportant_
        **/ignore        # Ignore all folders that are named ignore
        **/ignore.py     # Ignore `ignore.py` in every folder that is loaded
        *                # Ignore everything
        !important.py    # Do not ignore `important.py` files!
        ----------------------------------------
        """
        # setup self ignore
        if ignorefile is None:
            self.ignore = lambda x: False
        else:
            self.ignore = IgnorePattern(folder, ignorefile=ignorefile)
        #
        if os.path.isdir(folder):
            self._import_folder(folder)

    def _import_folder(self, folder):
        """Import all files/folders in the folder, if the folder
        contains an `__init__.py` file it is considered a
        module and will be only _import_module will be called"""
        if self.ignore(folder) is True:
            return None
        #
        files = tuple(os.listdir(folder))
        #
        if '__init__.py' in files:
            return self._import_module(folder)
        #
        files = tuple(os.path.join(folder, filename) for filename in files)
        #
        for filename in files:
            if filename.endswith('__pycache__'):
                continue
            if self.ignore(filename) is True:
                continue
            if filename.endswith('.py'):
                self._import_file(filename)
            elif os.path.isdir(filename):
                self._import_folder(filename)
        return None

    @staticmethod
    def _import_module(folderpath):
        """import a module"""
        path, name = os.path.split(folderpath)
        with AddFolderToPath(path):
            try:
                importlib.import_module(name)
                return
            except Exception:
                pass
        raise ImportError(f"Could not load module: {folderpath}")

    @staticmethod
    def _import_file(filepath):
        """Import a file"""
        _, name = os.path.split(filepath)
        # rm .py
        name = name[:-3]
        spec = importlib.util.spec_from_file_location(name, filepath)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            return module
        except Exception:
            pass
        filepath = os.path.abspath(filepath)
        raise ImportError(f"Could not load file: {filepath}")


def get_matcher(pattern):
    """Returns the matcher for a given pattern
    Parameters
    ----------
    pattern: str
        path to be matched

    Returns
    -------
    PathMatcher
        the corresponding matcher for the given pattern
    """
    parent, path = os.path.split(pattern)
    paths = [path]
    while parent != '':
        parent, path = os.path.split(parent)
        paths.append(path)
    #
    paths.reverse()
    #
    if len(paths) == 1:
        path = paths[0]
        if '*' in path:
            path = path.replace('*', r'.*')
        return PathMatcherGlobal(path)
    if paths[0] == '**':
        return PathMatcherGlobal(_replace_placeholder(paths[1:]), nlevel=len(paths)-1)
    return PathMatcher(_replace_placeholder(paths))


def _replace_placeholder(paths):
    """Replaces each placeholder in the values in the path and then
    returns the corresponding pattern """
    pattern = ''
    for path in paths:
        if '*' in path:
            path = path.replace('*', r'[\w\d-]*')
        pattern = os.path.join(pattern, path)
    return r'^' + pattern + r'$'


class PathMatcher:
    """A `PathMatcher`, matches anything that represents the given pattern"""

    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def match(self, path):
        """Check if path matches pattern"""
        if path == '':
            return False
        return self.pattern.match(path) is not None


class PathMatcherGlobal(PathMatcher):
    """A Global `PathMatcher`, matches anything that ends on the given pattern"""

    def __init__(self, pattern, *, nlevel=1):
        """
        Parameters
        ----------
        pattern: str
            Used pattern to match

        nlevel: int, default=1
            number of levels in the pattern
        """
        self.nlevel = nlevel
        super().__init__(pattern)

    def match(self, path):
        """Check if path matches pattern"""
        path = self._get_path(path)
        if path == '':
            return False
        return self.pattern.match(path) is not None

    def _get_path(self, path):
        """Get only the part of the path that should be matched!"""
        if self.nlevel == 1:
            return os.path.basename(path)
        paths = []
        for _ in range(self.nlevel):
            if path == '':
                return path
            paths.append(os.path.basename(path))
            path = os.path.dirname(path)
        paths.reverse()
        #
        out_path = ''
        for _path in paths:
            out_path = os.path.join(out_path, _path)
        return out_path


class IgnorePattern:
    """Load pattern in ignore file and setup an ignore function
    that will decide if a given filepath should be ignored or not"""

    def __init__(self, folder, ignorefile):
        """Setup `IgnorePattern`
        Parameters
        ----------
        folder: str
            name of the plugin folder, where the ignorefile should be inside

        ignorefile: str
            name of the ignorefile
        """

        if folder in ('.', ''):
            self.nignore = 0
        else:
            self.nignore = len(folder) + 1
        #
        self.folder = folder
        patterns = self._load_ignorefile(os.path.join(folder, ignorefile))
        self.matchers = [get_matcher(pattern) for pattern in patterns
                         if not pattern.startswith('!')]
        self.non_matchers = [get_matcher(pattern[1:]) for pattern in patterns
                             if pattern.startswith('!')]

    @staticmethod
    def _load_ignorefile(filename):
        """Load the patterns in the ignore file"""
        if not os.path.isfile(filename):
            return []
        #
        out = []
        with open(filename, 'r') as fhandle:
            for line in fhandle:
                line, _, _ = line.partition('#')
                line = line.rstrip()
                if line == '':
                    continue
                out.append(line)
        return out

    def __call__(self, filename):
        """Loop over all matchers and see if something should be ignored"""
        filename = filename[self.nignore:]
        # if entry in non_matcher -> do not ignore the entry
        for matcher in self.non_matchers:
            if matcher.match(filename) is True:
                return False
        # if entry in matcher -> do ignore the entry
        for matcher in self.matchers:
            if matcher.match(filename) is True:
                return True
        # do not ignore the entry
        return False


class AddFolderToPath:
    """Context Manager to add a folder to the pythonpath for import"""

    __slots__ = ('folder', 'idx')

    def __init__(self, folder):
        """save the folder name"""
        if folder in ('', None):
            folder = '.'
        self.folder = folder
        self.idx = None

    def __enter__(self, *args, **kwargs):
        """Add the folder to your path"""
        sys.path.append(self.folder)

    def __exit__(self, *args, **kwargs):
        """Remove the folder from the path"""
        try:
            sys.path.remove(self.folder)
        except Exception:
            pass

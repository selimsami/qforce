from collections.abc import MutableMapping
import re


class FileIterable:
    """Basic Iterator over the file, analouge to open()"""

    def __init__(self, filename, options="r"):
        self.closed = True
        self._previous = 0
        self._status = None
        self._fileiter = self._open(filename, options)

    def _open(self, filename, options):
        fhandle = open(filename, options)
        self.closed = False
        return fhandle

    def _read(self):
        while True:
            line = self._fileiter.readline()
            if not line:
                break
            yield line

    def __iter__(self):
        self._status = self._read()
        return self

    def __next__(self):
        return next(self._status)

    def __del__(self):
        if not self.closed:
            self._fileiter.close()


class ConfigParser(MutableMapping):
    """"Basic logic to parse own INI files"""

    comment = "#"
    base = "DEFAULTS"
    _is_header = re.compile(r"\s*\[\s*(?P<header>.*)\]\s*")
    _is_entry = re.compile(r"(?P<key>.*)=(?P<value>.*)")

    def __init__(self, config, literals):
        self._config = config
        self.literals = {name: None for name in literals}

    @classmethod
    def from_string(cls, filename, literals):
        config, literals = cls.read(filename, literals)
        return cls(config, literals)

    @classmethod
    def _header(cls, line):
        match = cls._is_header.match(line)
        if match is None:
            return None
        return match['header'].strip()

    @classmethod
    def _entry(cls, line):
        match = cls._is_entry.match(line)
        if match is None:
            return None, None
        return match['key'].strip(), match['value'].strip()

    @classmethod
    def _parse_literals(cls, currentheader, fileiter):
        string = []
        for line in fileiter:
            header = cls._header(line)
            if header is not None:
                if header != currentheader:
                    return "".join(string), header
                continue
            string.append(line)
        return "".join(string), None

    @classmethod
    def get_literals(cls, header, literals, fileiter):
        literals[header], header = cls._parse_literals(header, fileiter)
        # if next block is also a literalblock continue
        if header in literals:
            return cls.get_literals(header, literals, fileiter)
        # return next block
        return header

    @classmethod
    def read(cls, filename, literals):
        #
        literals = {name: None for name in literals}
        #
        entries = {}
        configs = {cls.base: entries}
        #
        fileiter = FileIterable(filename)
        for line in fileiter:
            header = cls._header(line)
            # check for literalblocks
            if header is not None:
                if header in configs:
                    raise ValueError(f"{header} defined twice in config")
                # handle literal block and find next header
                if header in literals:
                    header = cls.get_literals(header, literals, fileiter)
                    if header is None:
                        break
                # reset entries to nothing
                entries = {}
                configs[header] = entries
                continue
            #
            line = line.strip()
            if line == "" or line.startswith(cls.comment):
                continue
            #
            key, value = cls._entry(line)
            if key is not None:
                entries[key] = value
                continue
            #
            raise ValueError(f"Line in config unknown: '{line}' ")
        return configs, literals

    # all logic for MutableMapping

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value):
        self._config[key] = value

    def __delitem__(self, key):
        del self._config[key]

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)

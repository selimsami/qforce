"""Basic Logging for QForce"""
import sys
from datetime import datetime
from io import StringIO


class LoggerExit(SystemExit):
    """Exit QForce with e.g. missing files, etc. no real Error
    does not stop run_complete
    """


class LoggerError(SystemExit):
    """Exit QForce with an error/inconsistancy, stops run_complete!"""


class Timer:

    def __init__(self, name, logger, *, start_msg=None, end_msg=None):
        self.name = name
        self.logger = logger
        self.start_msg = start_msg
        self.end_msg = end_msg
        self._start = None
        self._stop = None

    def __call__(self, func):
        """Timer class can also be used as a Function decorator!"""
        def _f(*args, **kwargs):
            with self:
                func(*args, **kwargs)
        return _f

    def __enter__(self):
        self.logger.info(self.start())

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.info(self.stop())

    def start(self):
        """Start the timer, returns the start msg as a string"""
        self._start = datetime.now()
        return self._startmsg()

    def stop(self):
        """Stop the timer, returns the stop msg as a string"""
        self._stop = datetime.now()
        return self._stopmsg()

    def _timedelta(self):
        difference = self._stop - self._start
        total_seconds = difference.total_seconds()
        hours = difference // 3600
        remainder = total_seconds - hours*3600
        minutes = remainder // 60
        seconds = remainder - minutes*60
        return "{int(hours):02d}:{int(minutes):02d}:{seconds:-12.8f}"

    def _startmsg(self):
        msg = '' if self.start_msg is None else self.start_msg
        return f"""{self.name}: Started at {self.logger.formatdatetime(self._start)}\n{msg}\n"""

    def _stopmsg(self):
        msg = '' if self.start_msg is None else self.start_msg
        return (f"{self.name}: Ended at {self.logger.formatdatetime(self._start)} "
                f"after {self._timedelta()}\n{msg}\n")


class QForceLogger:

    def __init__(self, filename=None, *, mode='w'):
        self.isstdout = False
        if filename is None:
            self._f = sys.stdout
            self.isstdout = True
        elif isinstance(filename, StringIO):
            self._f = filename
        else:
            self._f = open(filename, mode=mode)

    def __del__(self):
        if self._f is not sys.stdout or isinstance(self._f, StringIO):
            self._f.close()

    def info(self, msg):
        """Standard printout for QForceLogger"""
        self.__write(f"{msg}\n")

    def write(self, msg):
        """Same as info"""
        self.__write(f"{msg}\n")

    def timestamp(self):
        """get the current time as a string"""
        return self.formatdatetime(datetime.now())

    def warning(self, msg):
        """write a warning"""
        self.__write(f"WARNING: {msg}\n")

    def note(self, msg):
        """write a note"""
        self.__write(f"NOTE: {msg}\n")

    def timeit(self, name, *, start_msg=None, end_msg=None):
        """Time a certain event"""
        return Timer(name, self, start_msg=start_msg, end_msg=end_msg)

    def exit(self, msg):
        """raise an LoggerExit, this will NOT end run_complete"""
        raise LoggerExit(msg)

    def error(self, msg):
        """raise an LoggerError, this will END run_complete"""
        raise LoggerError(msg)

    def __write(self, txt, flush=True):
        self._f.write(txt)
        if flush is True:
            self._f.flush()

    @staticmethod
    def formatdatetime(time):
        return time.strftime('%Y-%m-%d %H:%M:%S')

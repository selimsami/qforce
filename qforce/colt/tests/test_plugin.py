import pytest
#
from collections import namedtuple
from colt import Plugin


@pytest.fixture
def plugin():

    class ExamplePlugin(Plugin):

        _plugins_storage = '_methods'
        _is_plugin_factory = True

    class PluginOne(ExamplePlugin):
        pass

    class PluginTwo(ExamplePlugin):
        pass

    class PluginThree(ExamplePlugin):
        pass

    plugins = namedtuple("plugins", ("one", "two", "three"))

    return ExamplePlugin, plugins(PluginOne, PluginTwo, PluginThree)


def test_plugin_basics(plugin):
    #
    base, plugins = plugin

    class PluginFour(base):
        pass

    assert base._methods.get("one", None) is None
    assert base._methods.get("PluginOne", None) == plugins.one
    assert base._methods.get("PluginTwo", None) == plugins.two
    assert base._methods.get("PluginThree", None) == plugins.three
    assert base._methods.get("PluginFour", None) == PluginFour

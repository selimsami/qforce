from .colt import Colt, ColtMeta
from .colt import add_defaults_to_dict, delete_inherited_keys


def plugin_meta_setup(clsdict):
    plugin_defaults = {
        '_register_plugin': True,
        '_is_plugin_factory': False,
        '_is_plugin_specialisation': False,
        '_plugins_storage': 'inherited'
    }

    add_defaults_to_dict(clsdict, plugin_defaults)
    delete_inherited_keys(['_plugins_storage'], clsdict)


class PluginMeta(ColtMeta):
    """Meta class to keep subclasshooks to handle plugins in a very simple manner
       (It also supports Colts question routines)
    """

    def __new__(cls, name, bases, clsdict):
        plugin_meta_setup(clsdict)
        return ColtMeta.__new__(cls, name, bases, clsdict)

    def __init__(cls, name, bases, clsdict):
        #
        cls._plugin_storage = getattr(cls, '_plugins_storage', '_plugins')
        if cls._plugins_storage == 'plugins':
            cls._plugins_storage = '_plugins'
        #
        cls.__store_subclass(name)
        #
        ColtMeta.__init__(cls, name, bases, clsdict)

    def __store_subclass(cls, name):
        """main routine to store the current class, that has been already created with __new__,
           in one of the `plugin_storage_classes` this class inherites from

           Args:
                cls (object): Current Class
                name (str): Name of the class

        """
        plugin_storage_classes, idx = cls.__get_storage_classes()
        # case current class is a storage class!
        if idx == 0:
            cls.__new_plugin_storage()
            return
        # store hook for current class in all selected plugin_storage classes
        cls.__store_plugin(name, plugin_storage_classes)

    def __store_plugin(cls, name, plugin_storage_classes):
        """store plugin in the plugin_storage_classes it inherits from"""
        if getattr(cls, '_register_plugin', True) is False:
            return
        # store plugin in all stoarge classes
        for storage_class in plugin_storage_classes:
            storage_class.add_plugin(name, cls)

    def __get_storage_classes(cls):
        """return all relevant storage classes"""
        mro = cls.mro()
        #
        storage_classes = []
        idx = []
        #
        for i, plugin_class in enumerate(mro):
            if getattr(plugin_class, '_is_plugin_factory', False) is True:
                storage_classes.append(plugin_class)
                idx.append(i)
                # stop if it is just a plugin specialisation
                if getattr(plugin_class, '_is_plugin_specialisation', False) is not True:
                    break
        #
        if idx == []:
            idx = -1
        else:
            idx = idx[0]
        return storage_classes, idx

    def __new_plugin_storage(cls):
        """create new plugin storage"""
        setattr(cls, cls._plugins_storage, {})


class PluginStorageDescriptor:
    """Simple way to reference to the plugin storage without using an property"""

    def __get__(self, obj, typ):
        return getattr(typ, getattr(typ, '_plugins_storage'))


class Plugin(Colt, metaclass=PluginMeta):
    """Base class for the construction of PluginFactories"""

    _plugins_storage = '_plugins'
    _is_plugin_factory = False
    _register_plugin = False
    _is_plugin_specialisation = False
    # none-value descriptor to store plugins
    plugins = PluginStorageDescriptor()

    @classmethod
    def add_plugin(cls, name, clsobj):
        """Register a plugin"""
        cls.plugins[name] = clsobj

    @classmethod
    def plugin_from_config(cls, config, *args, **kwargs):
        """has to be the correct setting"""
        return cls.plugins[config.value].from_config(config, *args, **kwargs)

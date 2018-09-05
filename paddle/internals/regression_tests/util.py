# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

import urllib
import os
import urlparse
import copy
import subprocess
import tarfile

class ConfigurableObject(object):
    """
    The object can be parsed from JSON/XML or other config format.

    Just invoke .parse to parse object.
    """

    def __init__(self, x_path, inner_obj):
        """
        Constructor
        :param x_path: The unique path for this object value in config file.
        Could be xpath for xml and json
        :type x_path: str
        :param inner_obj: The object which save the parse result. Invoke it
        to set value.
        :type inner_obj: callable
        """
        self.XPath = x_path
        self.obj = inner_obj

    def parse(self, obj):
        """
        Parse config object into value.
        :param obj: The config object, could be json.parse from json string,
        or xml object.
        :return: Nothing.
        :rtype: None
        """
        raise NotImplementedError()


class JsonConfigurableObject(ConfigurableObject):
    """
    The object can be parsed from json
    """

    def __init__(self, p, o):
        ConfigurableObject.__init__(self, p, o)
        self.__xpath__ = None

    def parse(self, json_obj):
        if self.__xpath__ is None:
            import jsonpath_rw
            self.__xpath__ = jsonpath_rw.parse(self.XPath)

        for match in self.__xpath__.find(json_obj):
            try:
                self.obj(match.value)
            except AssertionError as e:
                print "Parse", self.XPath, "error"
                print json_obj
                raise e


def with_config(obj, p, config_type=JsonConfigurableObject):
    """
    Mark a settable value can be parsed from json/xml etc.
    :param obj: original object. which invoke this object is set value.
    :type obj: callable
    :param p: xpath for config file
    :type p: str
    :param config_type: Class for Config file.

    :note: It just append configurable object into obj.configs = [].
    """
    if not hasattr(obj, "configs"):
        obj.configs = []
    obj.configs.append(config_type(p, obj))
    return obj


def parse_json(obj, json_obj):
    """
    Parse json config for object obj.
    :param obj: the object that contains many attributes with
    ConfigurableObject.
    :param json_obj: the object return by json.parse
    :return: Nothing
    :rtype: None
    """
    for name in dir(obj):
        tmp = getattr(obj, name)
        if hasattr(tmp, "configs") and isinstance(tmp.configs, list):
            for each_config in tmp.configs:
                if isinstance(each_config, ConfigurableObject):
                    each_config.parse(json_obj)


class __NeedCopyWhenInit__(object):
    """
    Object Mark for Deepcopy from class scope to instance scope when __init__

    Just a object type mark internal uses.
    """
    pass


class SimpleConfigValue(__NeedCopyWhenInit__):
    """
    A simple configurable value. It will not serialize to anything.
    User can get value by obj.value
    """

    def __init__(self, default_value=None):
        self.value = default_value

    def __call__(self, value):
        self.value = value


class CommandFlag(SimpleConfigValue):
    """
    Base class of CommandFlag, Used for CommandExecutor.

    Use can set command flag by invoke this object.
    """

    def __init__(self, name, tp, default_value, value):
        """
        Constructor
        :param name: the flag name. For example, 'use_gpu' means the command
        has --use_gpu flag.
        :type name: str
        :param tp: the type of flag.
        :type tp: type class
        :param default_value: the default value of flag. It means not to
        print flag when default_value and value both be None.
        :param value: the value of flag.
        """
        self.name = name
        self.type = tp
        self.default_value = default_value

        assert isinstance(self.name, str)
        assert callable(self.type)
        SimpleConfigValue.__init__(self, value)

    def __repr__(self):
        """
        toString method for flag. If flag is not set, then return "",
        else will format flag to command line argument.
        :rtype: str
        """
        if self.value is None:
            self.value = self.default_value
        if self.value is not None:
            assert isinstance(self.value, self.type)
            return self.fmt_arg(self.name, self.value)
        else:
            return ""

    def __call__(self, v):
        """
        Set flag value
        :param v: value
        :return: Nothing
        :rtype: None
        :raise: AssertError
        """
        assert isinstance(v, self.type)
        self.value = v

    def fmt_arg(self, k, v):
        """
        Format key,value to command line argument
        :param k: the name for flag without prefix '--'
        :type k: str
        :param v: the value for flag. It is checked and not None.
        :return: string for command line argument
        :rtype: str
        """
        return "--" + "=".join([k, repr(v)])

    def __str__(self):
        """
        Same for repr
        """
        return repr(self)


class BoolCMDFlag(CommandFlag):
    """
    Bool command line flag. Such as --use_gpu=true --local=false etc.
    """

    def __init__(self, name, default_value=None, value=None):
        """
        Constructor
        :param name: flag name
        :type name: str
        :type default_value: bool or None
        :type value: bool or None
        """
        assert isinstance(default_value, bool)
        assert isinstance(value, bool) or value is None

        CommandFlag.__init__(self, name, bool, default_value, value)


class IntCMDFlag(CommandFlag):
    """
    Int command line flag. Such as --trainer_count=12 --gpu_id=2 etc.
    """

    def __init__(self, name, default_value=None, value=None):
        assert isinstance(default_value, int) or default_value is None
        assert isinstance(value, int) or value is None

        CommandFlag.__init__(self, name, int, default_value, value)


class StringCMDFlag(CommandFlag):
    """
    String command line flag. Such as --config=blah blah.config
    """

    def __init__(self, name, default_value=None, value=None):
        assert isinstance(default_value, basestring) or default_value is None
        assert isinstance(value, basestring) or value is None

        CommandFlag.__init__(self, name, basestring, default_value, value)

    def fmt_arg(self, k, v):
        return "--%s=%s" % (k, v)


class SimpleFlag(CommandFlag):
    """
    Simple flag is a bool flag. But not have '='.
    such as '-l' in 'ls -l -h'.
    """

    def __init__(self, name, default_value=False, value=None):
        assert isinstance(default_value, bool)
        assert isinstance(value, bool) or value is None
        CommandFlag.__init__(self, name, bool, default_value, value)

    def fmt_arg(self, k, v):
        if v:
            return "-" + k if len(k) == 1 else "--" + k
        else:
            return ""


class CommandEnv(SimpleConfigValue):
    """
    Command Environment Variable
    """

    def __init__(self, key, value=None, app=True):
        """
        Environment Variable For command.

        :param key: Environment Variable Name
        :type key: str
        :param value: Environment Variable Value
        :type value: str
        :param app: Append environment variable, or reset. True means append.
        :type app: bool
        """
        assert isinstance(key, str)
        assert value is None or isinstance(value, str)
        assert isinstance(app, bool)

        self.key = key
        self.app = app
        self.is_set = value is not None
        SimpleConfigValue.__init__(self, value)

    def __call__(self, val):
        """
        Set value
        """
        if isinstance(val, str):
            self.value = val
        elif isinstance(val, unicode):
            self.value = str(val)
        self.is_set = True

    def __repr__(self):
        """
        toString
        """
        if self.is_set:
            return "=".join([self.key, self.value])
        else:
            return ""

    def clean(self):
        """
        Clean Environment.
        """
        self.is_set = False
        self.value = None


class CommandExecutor(object):
    """
    Command line executor. Abstract for a runnable command.
    """

    def __init__(self, exec_name, print_stderr=False, cwd=None, clean_env=False):
        """
        Constructor
        :param exec_name: executable command name. Maybe lazy str value
        :type exec_name: str or callable ()=>str
        :param print_stderr: print stdout and stderr or not
        :type print_stderr: bool
        :param cwd: work directory
        :type cwd: str
        :param clean_env: remove all environment variable or not.
        :type clean_env: bool
        """
        # TODO(yuyang18): make sure exec_name is executable.
        assert isinstance(exec_name, basestring) or callable(exec_name)
        # TODO(yuyang18): make sure cwd is a path
        # TODO(yuyang18): should cwd be lazy?
        assert isinstance(cwd, basestring) or cwd is None

        self.exec_name = exec_name
        self.print_stderr = print_stderr
        self.cwd = cwd
        self.clean_env = clean_env

        self.flags = []
        self.envs = []

        # Copy __NeedCopyWhenInit__ into object.
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, __NeedCopyWhenInit__):
                new_obj = copy.deepcopy(obj)
                setattr(self, name, new_obj)
                if isinstance(obj, CommandFlag):
                    self.flags.append(new_obj)
                elif isinstance(obj, CommandEnv):
                    self.envs.append(new_obj)

    def __repr__(self):
        return " ".join(map(repr, filter(lambda x: x.is_set, self.envs)) +
                        [self.exec_name() if callable(self.exec_name) else
                         self.exec_name] +
                        filter(lambda x: len(x) != 0, map(repr, self.flags)))

    def exec_(self):
        """
        Execute command.
        :rtype: subprocess.Popen
        """
        args = self.exec_name().split() if callable(self.exec_name) else self.exec_name.split()
        args.extend(filter(lambda x: len(x) != 0, map(repr, self.flags)))
        debug = self.print_stderr() if callable(self.print_stderr) else self.print_stderr
        env = {}
        if not self.clean_env:
            env = copy.deepcopy(os.environ)

        for e in self.envs:
            assert isinstance(e, CommandEnv)
            if e.is_set:
                if e.app and env.has_key(e.key):
                    env[e.key] = ":".join([e.value, env[e.key]])
                else:
                    env[e.key] = e.value
        return subprocess.Popen(args=args,
                                bufsize = 0,
                                stdout=subprocess.PIPE if not debug else None,
                                stderr=subprocess.PIPE if not debug else None,
                                cwd=self.cwd,
                                env=env)

def download_file_list(baseurl, file_list, out_path, overwrite=False):
    """
    Download a list of files.
    baseurl: the base http url to download file.
    file_list: a list of the names of files to download.
    out_path: the path to store downloaded files.
    overwrite: whether to overwrite the existing files.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for f in file_list:
        url = urlparse.urljoin(baseurl, f)
        outname = os.path.join(out_path, f)
        download_file(url, outname, out_path, overwrite)


def download_file(url, outname, out_path, overwrite=False):
    """
    Download a file.
    url: the http url to download file.
    outname: the name to store the downloaded name.
    """
    if not overwrite and os.path.exists(outname):
        return
    down_file = urllib.URLopener()
    down_file.retrieve(url, outname)
    if outname.find("tar.gz")!= -1:
        untar_file(outname, out_path)

def untar_file(outname, out_path):
    """
    Untar a tar.gz file
    outname: the name of tar.gz file
    out_path: the path to store untared file.
    """
    f = tarfile.open(outname)
    f.extractall(path = out_path)

if __name__ == '__main__':
    class TestCommand(CommandExecutor):
        FLAGS_l = with_config(SimpleFlag('l'), "ls.l")
        FLAGS_a = SimpleFlag('a')
        FLAGS_h = SimpleFlag('h')

        CMD_NAME = with_config(SimpleConfigValue("ls"), "ls.name")

        WITH_GPU = CommandEnv('PADDLE_WITHGPU')

        def __init__(self):
            CommandExecutor.__init__(self, lambda : self.CMD_NAME.value)


    ls = TestCommand()

    parse_json(ls, {"ls": {'l': True, "name": "ls2"}})

    print ls

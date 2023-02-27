from pyhocon.config_tree import ConfigTree

def dict_to_configtree(conf_dict):
    conf = ConfigTree()
    for key in conf_dict:
        if type(conf_dict[key]) != dict:
            conf.put(key, conf_dict[key])
        else:
            conf.put(key, dict_to_configtree(conf_dict[key]))
    return conf

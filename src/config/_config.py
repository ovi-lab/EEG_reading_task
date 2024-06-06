import logging 
import os 

import yaml

logger = logging.Logger(__name__)

class Config:
    def __init__(self):
        fileRoot = os.getcwd() 

        rootDir = os.path.dirname(fileRoot)

        '''check if root directory actually exists '''

        if not (os.path.isdir(rootDir)):
            raise Exception (" Could not find the root directroy path on"
                             + fileRoot )
        else:
            root = rootDir

        self.__root =   root
        confog_path_from_root =  os.path.relpath(os.path.dirname(os.path.abspath(__file__)), start = self.__root)
        self.__config_path = os.path.join(rootDir,confog_path_from_root , 'config.yml')  


    def __getConfig(self):
        config = {}

        if not os.path.exists(self.__config_path):
            raise Exception (" Could not find the config file on path"
                             + self.__config_path )
        
        else:
            with open(self.__config_path, 'r') as file:
                contents = yaml.safe_load(file)

                if contents is not None:
                    config.update(contents)
                    config['root'] =  self.__root
                    
        return config           


    def getConfigSnapshot(self):
        return self.__getConfig()


# ConfigObj = Config()

import os
from typing import Callable
import  yaml
import pandas as pd
import mne 

from config import Config

configObj = Config()
configss = configObj.getConfigSnapshot()

def getOVStimCodes():# -> dict[str: int]
    ovStimCodes = {}
    ovStimListPath = os.path.join(configss['root'],
                                   configss['ov_stim_list_path'])
    with open(ovStimListPath) as file:
        lines = file.readlines()
        for line in lines:
            val = line.split() 
            ovStimCodes[val[0].strip()] = int(val[2].strip(), base =16)

   
    if not len(ovStimCodes) == len(lines):
        raise Exception (
            "The number of stimulation codes read from the stimulations " +
            "file is not equal to the number of lines in that file."
                         )    
        
    return ovStimCodes

def getChannelNamesEEGO(): # -> list[str]:
    path = os.path.join(configss['root'],
                         configss['eego_electrode_map_path'])
    with open(path, "r") as f:
        channelNames = [line.strip() for line in f]
    return channelNames


def getStimGroups():
    stimGroups = {}
    path = os.path.join(configss['root'], configss['oddball_stim_path'])

    with open(path, 'r') as file: 
        contents = yaml.safe_load(file)

        if contents is not None:
            stimGroups.update(contents)
    # to flatten the stim dictionary json_normalize is used
    df = pd.json_normalize(stimGroups, sep='/')
    return df.to_dict(orient='records')[0]


def getMontage():
    path = os.path.join(configss['root'], configss['electrode_layout_path'])
    montage = mne.channels.read_custom_montage(path)

# fig1 = neonatal_montage.plot(sphere=(0.00, -0.04, 0.00, 0.1255))   plotting 
    return montage

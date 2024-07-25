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


def segmentData():

    stimcodes =  getOVStimCodes()
    stimGroups = getStimGroups()

    inv_stimCodesMap =  { v: k for k, v in stimcodes.items()}
    inv_stimGroupsMap = { v: k for k, v in stimGroups.items() }


    # specify the participant numbers
    for pnum in (1, 2, 4):
        file_path =  "C:\\Users\\erang\\Desktop\\Reading_task\\ov_experiment\\scenarios\\data\\Pilot\\record_p{}.gdf".format(pnum)
        raw  = mne.io.read_raw_gdf(file_path)

        events_from_annot, event_dict =  mne.events_from_annotations(raw)

        # if check to make sure only get the relevant keys 
        modified_event_dict =  { inv_stimGroupsMap[inv_stimCodesMap[int(k)]] :
                                v for k,v in event_dict.items() 
                                if inv_stimCodesMap[int(k)] in inv_stimGroupsMap}

        timings ={}

        timings['timing/distractive/start'] = [event[0] / raw.info['sfreq'] for \
                                    event in events_from_annot if event[2] ==  \
                                modified_event_dict['timing/distractive/start']]
        timings['timing/distractive/stop'] = [event[0] / raw.info['sfreq'] for \
                                event in events_from_annot if event[2] == \
                                modified_event_dict['timing/distractive/stop']]
        timings['timing/attentive/start'] = [event[0] / raw.info['sfreq'] for \
                                event in events_from_annot if event[2] == \
                                modified_event_dict['timing/attentive/start']]
        timings['timing/attentive/stop'] = [event[0] / raw.info['sfreq'] for \
                                event in events_from_annot if event[2] == \
                                modified_event_dict['timing/attentive/stop']]
        
        num_passages_per_condition = len(timings['timing/distractive/start'])
        
        for i in range(0, num_passages_per_condition):
            raw_segment = raw.copy().crop(tmin=timings['timing/distractive/start'] [i] ,
                                    tmax=timings['timing/distractive/stop'] [i])
        
            path =  "C:\\Users\\erang\\Desktop\\Reading_task\\ov_experiment\\scenarios\\data\\Pilot\\P{}\\D{}.fif".format(pnum, i)


            raw_segment.save(fname = path, overwrite=True)


            raw_segment = raw.copy().crop(tmin=timings['timing/attentive/start'] [i] ,
                                    tmax=timings['timing/attentive/stop'] [i])
        
            path =  "C:\\Users\\erang\\Desktop\\Reading_task\\ov_experiment\\scenarios\\data\\Pilot\\P{}\\ND{}.fif".format(pnum, i)


            raw_segment.save(fname = path, overwrite=True)

def getPIDtoCondDict():

    conditions_data_path =  'psychopy_experiment\conditions\conditions.xlsx'
    path = os.path.join(configss['root'], conditions_data_path ) 
    print(path)

    df = pd.read_excel(path)
    # Function to limit to the first two occurrences of 'D' and 'ND'

    sorted_df = df.sort_values(by='Index').reset_index(drop=True)

    def limit_blocks(blocks):
        d_count = 0
        nd_count = 0
        limited_blocks = []
        for index, value in enumerate(blocks):
            if value == 'D' and d_count < 2:
                limited_blocks.append(f'{value}{d_count}')
                d_count += 1
            elif value == 'ND' and nd_count < 2:
                limited_blocks.append(f'{value}{nd_count}')
                nd_count += 1
        return limited_blocks

    # Create the dictionary with limited 'D' and 'ND' values
    return {pid: limit_blocks(list(sorted_df[sorted_df["PID"] == pid]['Block_type'])) for pid in sorted_df["PID"].unique() if pid > 0 }
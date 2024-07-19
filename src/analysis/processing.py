import mne
import pandas as pd
import tools.helpers
import os
import matplotlib.pyplot as plt
from mne.preprocessing import ICA 

from config import Config
configObj = Config()
configss = configObj.getConfigSnapshot()


# todo: add validations to check the path, file type, partipantId check
def loadData(partipantId):
    participant_name = 'P' + str(partipantId)
    partipant_data_path =  participant_name + '/' + participant_name +'.gdf'
    path = os.path.join(configss['root'], configss['data_dir'] , partipant_data_path ) 
    raw  = mne.io.read_raw_gdf(path)
    return raw

def loadSegmentedData(partipantId, block_number):

    participant_number = 'P' + str(partipantId)
    partipant_data_path =  'Pilot'+ '/'+ participant_number + '/' + block_number +'.fif'
    path = os.path.join(configss['root'], configss['data_dir'] , partipant_data_path ) 
    raw  = mne.io.read_raw_fif(path)
    return raw


def preprocessing(raw):

    raw = raw.resample(sfreq=250)

    # remove unnecessary channels
    data_channels = [x for x in raw.ch_names if x not in 
                     configss['non_data_channels']]
    if configss['target_channels'] is not None:
        data_channels = configss['target_channels']

    raw = raw.pick(data_channels)

    l_freq = configss['l_freq'] if configss['l_freq'] is not None else None 
    h_freq = configss['h_freq'] if configss['h_freq'] is not None else None 
    raw_filtered = raw.filter(l_freq=l_freq, h_freq=h_freq,
                                         picks = data_channels)

    notch_freqs = configss['notch_freqs'] if configss['notch_freqs'] is not \
        None else None 

    raw_filtered =  raw_filtered.notch_filter(freqs=notch_freqs,
                                             picks= data_channels)
    
    montage = tools.helpers.getMontage()

    raw_filtered.set_montage(montage)

    raw_artifact_corrected = applyICA(raw_filtered)

    return  raw_artifact_corrected


def eventEpochdata(raw):
    # get events and event dict 
    events_from_annot, event_dict =  mne.events_from_annotations(raw)

    stimcodes =  tools.helpers.getOVStimCodes()
    stimGroups = tools.helpers.getStimGroups()

    inv_stimCodesMap =  { v: k for k, v in stimcodes.items()}
    inv_stimGroupsMap = { v: k for k, v in stimGroups.items() }

    # if check to make sure only get the relevant keys 
    modified_event_dict =  { inv_stimGroupsMap[inv_stimCodesMap[int(k)]] :
                            v for k,v in event_dict.items() 
                            if inv_stimCodesMap[int(k)] in inv_stimGroupsMap}

    # only select events related to odd and the frequent stimuli
    epoch_event_dict =  {k:v for k, v in modified_event_dict.items() if 'freq' in k}     

    tmin = configss['epoch_tmin'] if configss['epoch_tmin'] is not None else None 
    tmax = configss['epoch_tmax'] if configss['epoch_tmax'] is not None else None

    # reject high amplitude signals, could be artifacts
    reject_criteria = dict(eeg=100e-6)  # 100 µV

    # baseline correction
    baseline_correction_l = configss['baseline_correction_l'] \
        if configss['baseline_correction_l'] is not None else None 
    
    baseline_correction_h = configss['baseline_correction_h'] \
        if configss['baseline_correction_h'] is not None else None

    epochs = mne.Epochs(raw, events_from_annot, 
                        event_id=epoch_event_dict, tmin=tmin, tmax=tmax,
                          preload=True, reject=reject_criteria, 
                        baseline=(
                            baseline_correction_l,
                              baseline_correction_h))

    return epochs, epoch_event_dict


def eventEpocshByBlocks(raw):
    # get events and event dict 
    events_from_annot, event_dict =  mne.events_from_annotations(raw)

    stimcodes =  tools.helpers.getOVStimCodes()
    stimGroups = tools.helpers.getStimGroups()

    inv_stimCodesMap =  { v: k for k, v in stimcodes.items()}
    inv_stimGroupsMap = { v: k for k, v in stimGroups.items() }

    # if check to make sure only get the relevant keys 
    modified_event_dict =  { inv_stimGroupsMap[inv_stimCodesMap[int(k)]] :
                            v for k,v in event_dict.items() 
                            if inv_stimCodesMap[int(k)] in inv_stimGroupsMap}

    tmin = configss['epoch_tmin'] if configss['epoch_tmin'] is not None else None 
    tmax = configss['epoch_tmax'] if configss['epoch_tmax'] is not None else None

    metadata_tmin, metadata_tmax = tmin, tmax

    row_events = [
    'auditory/onset/stimulus/non_freq',
    'auditory/onset/stimulus/freq',
    'visual/image_display/onset/instruction/set_1/instruction_1'
    ]


    metadata, meta_events, meta_event_id = mne.epochs.make_metadata(
    events=events_from_annot,
    event_id= modified_event_dict, 
    row_events = row_events,
    tmin=metadata_tmin,
    tmax=metadata_tmax,
    sfreq=raw.info["sfreq"],
    keep_first= ['auditory/onset/stimulus']
    )

    metadata["oddball"] = False
    metadata["control"] = False
    metadata.loc[ metadata['first_auditory/onset/stimulus'] == 'non_freq', "oddball" ] = True 
    metadata.loc[ metadata['first_auditory/onset/stimulus'] == 'freq', "control" ] = True 

    mask = metadata['visual/image_display/onset/instruction/set_1/instruction_1'] == 0.0

    relavent_indexes = metadata[mask].index

    #initialize column
    metadata.loc[:, "block"] = ""

    for i in range(len(relavent_indexes)):
        from_index =  relavent_indexes[i -1] if i > 0 else 0
        to_index = relavent_indexes[i]
        metadata.loc[from_index: to_index, "block"] = "b" + str(i)


    # reject high amplitude signals, could be artifacts
    reject_criteria = dict(eeg=100e-6)  # 100 µV


    # baseline correction
    baseline_correction_l = configss['baseline_correction_l'] \
        if configss['baseline_correction_l'] is not None else None 
    
    baseline_correction_h = configss['baseline_correction_h'] \
        if configss['baseline_correction_h'] is not None else None

    epochs = mne.Epochs(raw, meta_events, 
                        event_id= meta_event_id, tmin=tmin, tmax=tmax,
                          preload=True, reject=reject_criteria, 
                          metadata=  metadata,
                        baseline=(
                            baseline_correction_l,
                              baseline_correction_h))

    return epochs, meta_event_id


# CI: confidence interval
# roi: interested channels 
# evokeds : mne evok signals 
def getERP(evokeds, roi, ci, invert = False):
    #plot evokeds at 0.9 CI
    fig, ax = plt.subplots()
    mne.viz.plot_compare_evokeds(evokeds, picks= roi, ci = ci , 
                                styles = {"oddball": {"color" :'red'}, 
                                        "control":{"color": 'blue'}}, 
                                    show =False, axes = ax )
    if(invert): 
        ax.invert_yaxis()
    
    ax.set_ylabel("V")

    plt.show()


def getERPMontage(evokeds):
    mne.viz.plot_compare_evokeds(
    evokeds,
    picks="eeg",
    ci = 0.95,
    styles = {"oddball": {"color" :'red'}, 
                                        "control":{"color": 'blue'}}, 
    axes="topo",
    ylim = dict(eeg=[-10e-6, 10e-6]), 
)    
    


def applyICA(raw):
    ica = ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(raw)

    ica.exclude = []
    num_excl = 0
    max_ic = 2
    z_thresh = 3.5
    z_step = .05

    while num_excl < max_ic:
        eog_indices, eog_scores = ica.find_bads_eog(raw,
                                                ch_name=['1L', '1R', '2LC', '2RC'], 
                                                threshold=z_thresh
                                                )
        num_excl = len(eog_indices)
        z_thresh -= z_step # won't impact things if num_excl is ≥ n_max_eog 

# assign the bad EOG components to the ICA.exclude attribute so they can be removed later
    ica.exclude = eog_indices
    ica.apply(raw)

    return raw


def epochContinuousData(raw):
    return mne.make_fixed_length_epochs(raw, duration=1, overlap = 0.1, preload=False)

def segmentData(p_num_list, preprocess = True):

    stimcodes =  tools.helpers.getOVStimCodes()
    stimGroups = tools.helpers.getStimGroups()

    inv_stimCodesMap =  { v: k for k, v in stimcodes.items()}
    inv_stimGroupsMap = { v: k for k, v in stimGroups.items() }


    # specify the participant numbers
    for pnum in (p_num_list):
        
        participant_name = 'record_p' + str(pnum)
        partipant_data_path =  'Pilot'+ '/' + participant_name +'.gdf'
        path = os.path.join(configss['root'], configss['data_dir'] , partipant_data_path ) 

        raw  = mne.io.read_raw_gdf(path)

        if(preprocess):
            raw = preprocessing(raw)

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
            
            block_number = 'D' + str(i)
            participant_number = 'P' + str(pnum)
            partipant_data_path =  'Pilot'+ '/'+ participant_number + '/' + block_number +'.fif'
            path = os.path.join(configss['root'], configss['data_dir'] , partipant_data_path ) 
        
            raw_segment.save(fname = path, overwrite=True)


            raw_segment = raw.copy().crop(tmin=timings['timing/attentive/start'] [i] ,
                                    tmax=timings['timing/attentive/stop'] [i])
        
            block_number = 'ND' + str(i)
            participant_number = 'P' + str(pnum)
            partipant_data_path =  'Pilot'+ '/'+ participant_number + '/' + block_number +'.fif'
            path = os.path.join(configss['root'], configss['data_dir'] , partipant_data_path ) 


            raw_segment.save(fname = path, overwrite=True)



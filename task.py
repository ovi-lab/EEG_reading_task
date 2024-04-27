import threading
from psychopy import visual, event, core, data, gui, sound 
#at the start of your script
from psychopy.hardware import keyboard
import numpy as np
import pandas as pd


info = {} #a dictionary
#present dialog to collect info
info['participant'] = ''
dlg = gui.DlgFromDict(info) #(and from psychopy import gui at top of script)
if not dlg.OK:
    core.quit()
#add additional info after the dialog has gone
info['fixTime'] = 0.2 # seconds
info['dateStr'] = data.getDateStr() #will create str of current date/time
filename = "data/" + info['participant'] + "_" + info['dateStr']

sound_duration = 0.3
sound_iti = 0.7
volume = 1
sound_playing = False


# try this way, needs only one thread
# have a for loop over time
# if the where each time we would play a tone for 0.2 in every 0.5s
# iterate through  trials while inside that for loop 



def reading_task(passage_stim, fixation, win, kb, auds, is_bg_sound):

    np.random.seed(7)
    sound_ind = np.random.binomial(1, 0.2, 10000)

    full_text = []

    with open('Text0.txt') as f:
        for line in f:
            for word in line.split():
                full_text.append(word)
                
    sentences = []
    word_count = 50
    start = 0
    end = 0
    for i in range(0, int(len(full_text)/ word_count )):
        end = end + word_count if end + word_count < len(full_text) else len(full_text)
        text_range = full_text[start: end ]
        sentence = ' '.join(text_range)
        sentences.append(sentence)
        start =  end

    globalClock = core.Clock()  # to track the time since experiment started
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 
    
    count = 0

    continueRoutine = True
    # routineForceEnded = False
    number_of_pages = len(sentences)

    current_page_number = 0
    past_page_number = -1
    audio_iter = 0
    sound_playing = False

    globalClock.reset()
    kb.clearEvents()
    kb.clock.reset()

    while(continueRoutine) :

        if(is_bg_sound): 
            if np.mod(globalClock.getTime(), sound_iti) < sound_duration:
                if (not sound_playing):
                    auds[sound_ind[audio_iter - 1]].stop()
                    sound_type  = sound_ind[audio_iter]
                    auds[sound_type].play()

                    if (sound_type):
                        count += 1

                    sound_playing = True
                    audio_iter += 1

            
            else:
                sound_playing = False


        if (past_page_number <  current_page_number):
            thisText = sentences[current_page_number]
            passage_stim.setText(thisText)

            passage_stim.draw()
            win.flip()

            past_page_number += 1  

        # check and handle keyboard and mouse  
        keys = kb.getKeys(keyList = ['space','escape'], clear =True)

        if(keys):
            resp = keys[0].name #take first response
            rt = keys[0].rt

            if resp=='escape':
                continueRoutine = False

            kb.clearEvents()
            kb.clock.reset()  

            current_page_number += 1

        if current_page_number == number_of_pages:
            continueRoutine = False

    return count

def main():
    #then creating a window is another single line. We’ll use units of pixels for the window for simplicity. Then all our stimulus dimensions will be in pixels:
    win = visual.Window([1920,1080], fullscr=True, units='norm',  screen=1)
    win.mouseVisible = False

    # create a default keyboard (e.g. to check for escape)
    kb = keyboard.Keyboard(backend='ptb')

    fixation = visual.Circle(win, size = 0.01,
        lineColor = 'white', fillColor = 'lightGrey')


    passage_stim =  visual.TextStim(win, height = 0.1, pos= (0.0, 0.0))

    #Auditory stimuli
    # Initialize stimuli
    aud1 = sound.Sound("C", octave=5, sampleRate=44100, secs=sound_duration)
    aud1.setVolume(volume)
    aud2 = sound.Sound("D", octave=6, sampleRate=44100, secs=sound_duration)
    aud2.setVolume(volume)
    auds = [aud1, aud2]


    odd_sound_count = reading_task(passage_stim,fixation, win, kb, auds, is_bg_sound = True )


    print("count is " + str(odd_sound_count))



main()
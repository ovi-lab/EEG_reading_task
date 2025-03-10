from psychopy import prefs
prefs.hardware['audioLib'] = ['ptb']
from psychopy import visual, event, core, data, gui, sound 
from psychopy.hardware import keyboard
import numpy as np
import pandas as pd
import random
import logging
import datetime
from sklearn.utils import shuffle
import os
import Stimulations
from time import sleep

import psutil
import time

np.random.seed(7)

### tcp tagging stuff
import sys
import socket
from time import time, sleep

#ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

FORMAT =  '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT,  datefmt='%Y-%m-%d %H:%M:%S')
# that do not set a less verbose level.
# logging.root.setLevel(logging.NOTSET)
logging.warning('Watch out!')  # info, error

current_directory = os.path.dirname(__file__)


# Preliminary setup box and variable capture
info={'participant':'', 'practice': ["Y", "N"] }
infoDlg=gui.DlgFromDict(dictionary=info, title='Setup',order=['participant'])
if not infoDlg.OK: core.quit()

PID=info['participant']
practice = info['practice']
blocks= ['ND','D']

globalClock = core.Clock()
routineTimer = core.Clock() 

timeStamp=core.getAbsTime()
today= datetime.date.today().strftime('%Y/ %m/ %d') 

data_results_filename =  os.path.join(current_directory, 'data','reading_data'\
                                      , "data_" + PID + "_" + data.getDateStr())

tone_results_file_name =  os.path.join(current_directory, 'data','tones'\
                                      , "tones_" + PID + "_" + data.getDateStr())

qa_results_file_name =  os.path.join(current_directory, 'data','qa'\
                                      , "qa_" + PID + "_" + data.getDateStr())


comp_results_file_name =  os.path.join(current_directory, 'data','completion'\
                                      , "comp_" + PID + "_" + data.getDateStr())


df_data = pd.DataFrame(columns=['PID', 'Date','Timestamp', 'BlockNo', \
                                'BlockType', 'Paragraph_id', 'Reading_time'])

df_tones = pd.DataFrame(columns=['PID', 'Date','Timestamp', 'BlockNo',\
                                  'BlockType', 'Paragraph_id', 'KeyPressed', 'CorrectAns', \
                                    'Correct'])

df_qa = pd.DataFrame(columns=['PID', 'Date','Timestamp', 'BlockNo', \
                              'BlockType', 'Paragraph_id', 'Question_id',\
                                  'KeyPressed', 'CorrectAns', 'Correct'])

df_comp =  pd.DataFrame(columns=['PID', 'Date','Timestamp', 'BlockNo', \
                              'BlockType', 'Paragraph_id',  'completion'])


# monitor setup
win = visual.Window([1920,1080], fullscr=True, units='norm',  screen=1)

# setup keyboard
kb = keyboard.Keyboard(backend='ptb')

# define tones
sound_duration = 0.2 
sound_iti = 2 # originally 1.5
sound_playing = False

aud1 = sound.Sound("C", octave=5, sampleRate=44100, secs=sound_duration, \
                   volume= 1)
aud2 = sound.Sound("D", octave=6, sampleRate=44100, secs=sound_duration, \
                   volume= 1)
auds = [aud1, aud2]

# define text stims
word_count_per_frame = 80

passage_stim =  visual.TextBox(win, size = (1, 1.5), font_size = 32, \
                                   pos= (0.0, 0.0), \
                                    grid_vert_justification='center' \
                                   , font_color=[1,1,1])

question_tone = visual.TextStim(\
    win, text= 'How many times did you hear the "ODD" tone ', pos=[0, 0.3], \
        wrapWidth=1.6, color='black')


textbox = visual.TextBox2(
     win, text='', font='Open Sans',
     pos=(0, 0),     letterHeight=0.2,
     size=(None, None), borderWidth=2.0,
     color='white',
     alignment='center',
     editable=True,
     name='textbox',
     autoLog=True,
)


endButton = visual.TextStim(win, name='endButton',
    text='Click here to continue',
    pos=(0, -0.3), height=0.1, 
    color='black');

mouse = event.Mouse(win=win)
x, y = [None, None]
mouse.mouseClock = core.Clock()
mouse.x = []
mouse.y = []
mouse.leftButton = []
mouse.midButton = []
mouse.rightButton = []
mouse.time = []
mouse.clicked_name = []
gotValidClick = False 


question_stim =  visual.TextStim(win, wrapWidth=1.6, pos= (0.0, 0.4))
                                 

answers_stim =  visual.TextBox(win, size = (1, 1), font_size = 32, 
                                   pos= (0.0, - 0.1), \
                                    grid_vert_justification='center'\
                                   , font_color=[1,1,1])



slider_text = visual.TextStim(win=win, name='text',
        text='I think, I was able to finish " _______" % of the passage.',
        # font='Arial',
        pos=(0, 0.25),   wrapWidth=1.6,
        # height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);

slider = visual.Slider(win=win, name='slider',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=['Nothing (0%)',  'All (100%)'], ticks=(0, 100), granularity=0.0,
        style='slider', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)



# host and port of tcp tagging server
HOST = '127.0.0.1'
PORT = 15361

# Artificial delay (ms). 
# It may need to be increased if the time to send the tag is too long and causes tag loss.
DELAY=0

# transform a value into an array of byte values in little-endian order.
def to_byte(value, length):
    for x in range(length):
        yield value%256
        value//=256

# connect 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

def get_process_by_name(process_name):
    """Find a process by its name."""
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'] == process_name:
            return process
    return None

def check_cpu_usage(process_name, threshold=10, interval=1):
    """Check if CPU usage of a specific process exceeds the threshold."""
    process = get_process_by_name(process_name)
    if not process:
        print(f"No process found with name {process_name}")
        return

    print(f"Monitoring process {process_name} (PID: {process.pid})")
    try:
        cpu_usage = process.cpu_percent(interval=interval)
            
        if cpu_usage > threshold:
            return True # generally when recording CPU usage> 10%
            # print(f"CPU Usage Alert: {cpu_usage}% (exceeds {threshold}%)")
        else:
            # print(f"CPU Usage: {cpu_usage}%")
            return False
            
        time.sleep(interval)
    except psutil.NoSuchProcess:
        print(f"Process {process_name} has terminated.")
        return False
    except KeyboardInterrupt:
        print("Monitoring stopped.")
        return False



if (not check_cpu_usage("openvibe-designer.exe", threshold=10)):
    print ('DATA IS NOT RECORDING')
    win.close()
    core.quit()



def sendTcpTag(event):
    padding=[0]*8
    event_id=list(to_byte(event, 8))

    # timestamp can be either the posix time in ms,
    # or 0 to let the acquisition server timestamp the tag itself.
    # timestamp=list(to_byte(int(time()*1000)+DELAY, 8))
    timestamp=list(to_byte(int(0*1000)+DELAY, 8))

    # send tag and sleep
    s.sendall(bytearray(padding+event_id+timestamp))
    sleep(1)



# functions 

def myShuffle(array):
    return shuffle(array)


def instructionScreen(window, instructionText, advance):
    textToShow = visual.TextStim(win=window, text=instructionText, pos=[0, 0],\
                                  wrapWidth=1.6, color='black',\
                                      anchorHoriz = 'center')
    textToShow.draw()
    window.flip()
    # Advance on key press
    if advance=='anykey':
        key=event.waitKeys()
        return key
    else:
        while True:
            keypress=event.waitKeys()[0]
            if keypress==advance:
                break

def instructionScreenWithSounds(window, instructionText, advance, aud):
    textToShow = visual.TextStim(win=window, text=instructionText, pos=[0, 0],\
                                  wrapWidth=1.6, color='black', \
                                    anchorHoriz = 'center')
    textToShow.draw()
    window.flip()
    core.wait(1)
    aud.play()
    core.wait(1)
    aud.stop()
    # Advance on key press
    if advance=='anykey':
        key=event.waitKeys()
        return key
    else:
        while True:
            keypress=event.waitKeys()[0]
            if keypress==advance:
                break


def messageScreen(message):
    textToShow = visual.TextStim(win=win, text=message, pos=[0, 0],\
                                  wrapWidth=1.6, color='black')
    textToShow.draw()
    win.flip()
    #whenever a message screen is displayed the experiment can end early if x 
    #is pressed
    routineTimer.reset()
    while routineTimer.getTime()<2:
        keys=event.getKeys(keyList='x')
        core.wait(0.01)
        if 'x' in keys:
            # outputCSV.writerow(['Terminated early by experimenter'])
            print ('Quitting... experiment terminated early by experimenter')
            win.close()
            core.quit()


def distractionTaskInstructions(win):

    ins3 = 'Distraction Condition Task:\n\n' +  \
    'In the distraction condition, you will be asked to '+\
    'count the number of "ODD" sounds you hear. \n\n' +\
    ' \n\nPress "SPACE" for more instructions'

    ins3a = '"ODD" Tone:\n\n' +  \
    'The "ODD" tone sounds like this:\n\n' +\
    '\n\nPress "SPACE" for more instructions'

    ins3b = '"Frequent" Tone:\n\n' +  \
    'The "Frequent" tone sounds like this:\n\n' +\
    '\n\nPress "SPACE" to continue'


    instructionScreen(win, ins3, 'space')
    instructionScreenWithSounds(win, ins3a, 'space', aud2 )
    instructionScreenWithSounds(win, ins3b, 'space', aud1 )

def introScreen(win):

    intro1 = 'Press a key to begin experiment'

    ins1 = 'Welcome to the Reading Comprehension Task\n\n' +  \
    'Task: \n\n' +\
    '1. Read the passage. \n' + \
    '2. Select the correct answer. \n'+ \
    ' \n\nPress "SPACE" for more instructions'

    ins1a = 'Page Controls: \n\n' +  \
    '1. A page contains 9 lines. \n' + \
    '2. A line consists of 40 characters (about 4 - 8 words). \n'+ \
    '3. Each page lasts 12 seconds. \n'+ \
    '4. You can advance through pages by pressing "SPACE".\n'+ \
    ' \n\nPress "SPACE" for more instructions'
    
    ins2 = 'Study Conditions:\n\n' +  \
    'You will participate in two types of study conditions/blocks: \n\n' +\
    '1. Distraction Condition: You will hear distraction sounds. \n' + \
    '2. Quiet Condition: There will be no distractions. \n'+ \
    ' \n\nPress "SPACE" for more instructions'
        

    instructionScreen(win, intro1 , 'anykey')
    instructionScreen(win, ins1, 'space')
    instructionScreen(win, ins1a, 'space')
    instructionScreen(win, ins2, 'space')

    distractionTaskInstructions(win)
    

def read_text(path):
    with open(path,  encoding="utf8") as f:
       return f.read()
                

def format_text(text, line_length=38, lines_per_page=9):
    words = text.split()
    pages = []
    current_page = ""
    current_line = ""
    page_count = 0

    for word in words:
        # Check if adding the next word would exceed the line length
        if len(current_line) + len(word) + 1 <= line_length:
            # if current_line:
            current_line += " "
            current_line += word
        else:
            # Add the current line to the current page
            current_line += ' \n '  
            current_page += current_line
            current_line = word
            page_count += 1
            # Check if the current page has enough lines
            if page_count== lines_per_page:
                pages.append(current_page)
                current_page = " "
                page_count = 0

    # Add any remaining text to the current page
    if current_line:
        current_page += current_line
    if current_page:
        pages.append(current_page)

    return pages


def practiceHelperScreens(win, param):
    text =  'The correct answer is {}'.format(param) +  \
   '\n\n\n\n Press "SPACE" to continue'
    instructionScreen(win, text, 'space')

def practiceSliderHelperScreens(win, param):
    text =  'You able to read {} % of the passage'.format(param) +  \
   '\n\n\n\n Press "SPACE" to continue'
    instructionScreen(win, text, 'space')



def block(win, test_type, path, block_type, block_number, paragraph_id):

    global df_data, df_tones, word_count_per_frame 

    text =  read_text(path)

    sentences_for_pages = format_text(text)
 
    number_of_pages = len(sentences_for_pages)

    sound_ind = np.random.binomial(1, 0.2, 1000)
    
    
    count = 0

    continueInnerLoop = True
    continueOuterLoop = True

    current_page_number = 0
    past_page_number = -1
    audio_iter = 0
    sound_playing = False

   
    if(block_type == 'D'):
        distractionTaskInstructions(win)
        if (test_type == 'Test'): 
            sendTcpTag(Stimulations.OVTK_StimulationId_Label_05)
    else:
        if (test_type == 'Test'): 
            sendTcpTag(Stimulations.OVTK_StimulationId_Label_03)

    globalClock.reset()
    kb.clearEvents()
    kb.clock.reset()

    while(continueOuterLoop):

        routineTimer.reset()
        while(continueInnerLoop) :

            if(block_type == 'D'): 
                # Plays the sounds 
                if np.mod(globalClock.getTime(), sound_iti) < sound_duration:
                    if (not sound_playing):
                        sound_type  = sound_ind[audio_iter]
                        auds[sound_type].play()

                        if (sound_type):
                            if (test_type == 'Test'): 
                                sendTcpTag(\
                                Stimulations.OVTK_StimulationId_Label_01)
                            count += 1
                        else:
                            if (test_type == 'Test'): 
                                sendTcpTag(\
                                Stimulations.OVTK_StimulationId_Label_02)    

                        sound_playing = True
                        audio_iter += 1

                
                else:
                    if (sound_playing):
                        auds[sound_ind[audio_iter - 1]].stop()
                        sound_playing = False


            if (past_page_number <  current_page_number):
                opacity_level = 1
                routineTimer.reset()
                thisText = sentences_for_pages[current_page_number]
                passage_stim.setText(thisText)
                passage_stim.setOpacity(opacity_level)

                passage_stim.draw()
                win.flip()

                past_page_number += 1  

            # check and handle keyboard and mouse  
            keys = kb.getKeys(keyList = ['space', 'escape'], clear =True)

            if(keys):
                resp = keys[0].name #take first response

                if resp=='escape':
                    continueInnerLoop = False
                    continueOuterLoop = False
                    save()
                    win.close()
                    sendTcpTag(Stimulations.OVTK_StimulationId_ExperimentStop)
                    core.quit()

                kb.clearEvents()
                kb.clock.reset()

                if resp=='space':
                    routineTimer.addTime(14)
                
            if routineTimer.getTime() > 14:
                current_page_number += 1

            if routineTimer.getTime() > 12:
                opacity_level =  (14 - routineTimer.getTime()) /2
                passage_stim.setOpacity(opacity_level)
                passage_stim.draw()
                win.flip()


            if ((current_page_number == number_of_pages)):
                if(block_type == 'D'):
                    if (test_type == 'Test'):  
                        sendTcpTag(Stimulations.OVTK_StimulationId_Label_06)
                else:
                    if (test_type == 'Test'): 
                        sendTcpTag(Stimulations.OVTK_StimulationId_Label_04)   
                continueInnerLoop = False
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()
                textbox.reset()
                textbox.setText('0')
                
                if (test_type == 'Test'):
                    new_row = {'PID':PID, 'Date':today, \
                               'Timestamp': timeStamp, 'BlockNo':block_number,\
                                  'BlockType': block_type,  \
                                    'Paragraph_id': paragraph_id,\
                                        'Reading_time': globalClock.getTime() }
                    # df_data = pd.concat([df_data, new_row], ignore_index=True)
                    df_data = df_data.append(new_row, ignore_index=True)

                core.wait(0.25) 

            core.wait(0.001) # helps with the keyboard polling issue
        # Text box to enter the "ODD" sound count
        if(block_type == 'D'): 

            question_tone.draw()
            textbox.draw()
            endButton.draw()
            win.flip()

            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        try:
                            iter(endButton)
                            clickableList = endButton
                        except:
                            clickableList = [endButton]
                        for obj in clickableList:
                            if obj.contains(mouse):
                                gotValidClick = True
                                mouse.clicked_name.append(obj.name)
                        x, y = mouse.getPos()
                        mouse.x.append(x)
                        mouse.y.append(y)
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(mouse.mouseClock.getTime())
                        if gotValidClick: 
                            continueOuterLoop = False  # abort routine on response
                            continueInnerLoop = False
                            corr = 0
                            if (textbox.text.isdigit()) and\
                                  (count == int(textbox.text)):
                                corr = 1 

                            logging.info( test_type +' - Actual Count: ' + \
                                          str(count) +', User Count: ' + \
                                            str(textbox.text))  # info, error      

                            if (test_type == 'Test'):
                                new_row = {'PID':PID, 'Date':today, 
                                           'Timestamp': timeStamp,
                                            'BlockNo':block_number, 
                                            'BlockType': block_type,
                                            'Paragraph_id': paragraph_id,  
                                            'KeyPressed':  textbox.text , 
                                            'CorrectAns':  count, 
                                            'Correct': corr }
                                # df_tones = pd.concat([df_tones, new_row],\
                                # ignore_index=True)
                                df_tones = df_tones.append(new_row,\
                                                            ignore_index=True)
                            else:
                                correctAns = count
                                practiceHelperScreens(win, correctAns)    

            core.wait(0.001) # helps with the keyboard polling issue
        else: 
            continueOuterLoop = False  # abort routine on response
            continueInnerLoop = False

    passageCompletetionScreen(win, test_type, block_type, 
                    block_number, paragraph_id)


    return questionsScreen(win, test_type, block_type, 
                    block_number, paragraph_id)

def passageCompletetionScreen(win, test_type, block_type, 
                    block_number, paragraph_id):
        
        global df_comp
        
        mouse.mouseClock.reset()
        prevButtonState = mouse.getPressed()
        continueLoop = True
        slider.reset()
        while (continueLoop):
            slider_text.draw()
            slider.draw()
            endButton.draw()
            win.flip()

            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        try:
                            iter(endButton)
                            clickableList = endButton
                        except:
                            clickableList = [endButton]
                        for obj in clickableList:
                            if obj.contains(mouse):
                                gotValidClick = True
                                mouse.clicked_name.append(obj.name)
                        x, y = mouse.getPos()
                        mouse.x.append(x)
                        mouse.y.append(y)
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(mouse.mouseClock.getTime())
                        if gotValidClick and slider.rating:
                            continueLoop = False 
                            if (test_type == 'Test'):
                                new_row = {'PID':PID, 'Date':today, 
                                            'Timestamp': timeStamp,
                                            'BlockNo':block_number, 
                                            'BlockType': block_type,
                                            'Paragraph_id': paragraph_id,  
                                            'completion':  slider.getRating() }
                                # df_tones = pd.concat([df_tones, new_row],\
                                # ignore_index=True)
                                df_comp = df_comp.append(new_row,\
                                                            ignore_index=True)
                            else:
                                correctAns = slider.getRating()
                                practiceSliderHelperScreens(win, correctAns)        
            core.wait(0.001) # helps with the keyboard polling issue

def questionsScreen(win, test_type, block_type, 
                    block_number, paragraph_id):
    global df_qa
     # reads from a condition file

    path =  os.path.join(current_directory, 'comp_materials','passage_qa',\
                         'questionbank.xlsx')
    all_df_comp_questions = pd.read_excel(path)
    
    df_comp_questions =  all_df_comp_questions.loc[\
        all_df_comp_questions['Paragraph_id'] == paragraph_id]

    continueInnerLoop = True
    number_of_questions = df_comp_questions.shape[0]
    current_question_number = 0
    past_question_number = -1

    kb.clearEvents()
    kb.clock.reset()
    core.wait(1) 
    while(continueInnerLoop):

        if(past_question_number < current_question_number):

            df_item =  df_comp_questions.iloc[current_question_number]
            question_stim.setText(df_item['Question'])
            answers_stim.setText(df_item['Answers'])
            correct_ans = df_item['Correct_answer']

            question_stim.draw()
            answers_stim.draw()
            win.flip()

            past_question_number +=1


        keys = kb.getKeys(keyList = ['a','b','c','d','e', 'escape'], clear =True)

        if(keys):
            resp = keys[0].name #take first response

            if resp=='escape':
                continueInnerLoop = False
                save()
                win.close()
                sendTcpTag(Stimulations.OVTK_StimulationId_ExperimentStop)
                core.quit()


            corr = 0
            if correct_ans == resp:
                corr = 1

            if (test_type == 'Test'):
                new_row = {'PID':PID, 'Date':today, 'Timestamp': timeStamp,\
                            'BlockNo':block_number, 'BlockType': block_type, \
                                  'Paragraph_id': paragraph_id,\
                                      'Question_id': df_item['Question_id'], \
                                        'KeyPressed':  resp ,  'CorrectAns': \
                                              correct_ans, 'Correct': corr }
                # df_data = pd.concat([df_data, new_row], ignore_index=True)
                df_qa = df_qa.append(new_row, ignore_index=True)

            else:
                practiceHelperScreens(win, correct_ans)        


            kb.clearEvents()
            kb.clock.reset()  

            current_question_number += 1

        if (current_question_number == number_of_questions):
              continueInnerLoop = False
        core.wait(0.001) # helps with the keyboard polling issue      

    return True          


def experimentScreen(win):

    
    expIntBlkInterval = 'One minute mandatory break'
    expIntro =' Press a key to begin the block'
    
    
    # reads from a condition file
    path =  os.path.join(current_directory, 'conditions','conditions.xlsx')
    all_df_block_setup = pd.read_excel(path)

    df_block_setup = all_df_block_setup.loc[all_df_block_setup['PID'] == int(PID)]
    df_block_setup = df_block_setup.reset_index(drop = True)

    #todo: do a group by for PID
    number_of_blocks = df_block_setup.shape[0]


    for index, row in df_block_setup.iterrows():

        instructionScreen(win, expIntro , 'anykey')
        path =  os.path.join(current_directory, 'comp_materials','passages',\
                         row["Paragraph_id"] + '.txt')

        block(win, 'Test', path , row["Block_type"], block_number = index, \
               paragraph_id = row["Paragraph_id"])
        
        if not ((index +1) == number_of_blocks): 
            expIntBlkIntervalToText = visual.TextStim(win, \
                                                      text= expIntBlkInterval,\
                                                        pos=[0, 0],\
                                                          wrapWidth=1.6,\
                                                            color='black')  
            expIntBlkIntervalToText.draw()
            win.flip()
            core.wait(60)
         
    return True  


def trainingScreen(win):

    
    expIntBlkInterval = 'One minute mandatory break'
    expIntro =' Press a key to begin the practice block'
    
    
    # reads from a condition file
    path =  os.path.join(current_directory, 'conditions','conditions.xlsx')
    all_df_block_setup = pd.read_excel(path)

    df_block_setup = all_df_block_setup.loc[all_df_block_setup['PID'] == int(-1)]
    df_block_setup = df_block_setup.reset_index(drop = True)

    #todo: do a group by for PID
    number_of_blocks = df_block_setup.shape[0]


    for index, row in df_block_setup.iterrows():

        instructionScreen(win, expIntro , 'anykey')
        path =  os.path.join(current_directory, 'comp_materials','passages',\
                         row["Paragraph_id"] + '.txt')

        block(win, 'Practice', path , row["Block_type"], block_number = index, \
               paragraph_id = row["Paragraph_id"])
             
        if not ((index +1) == number_of_blocks): 
            expIntBlkIntervalToText = visual.TextStim(win, \
                                                      text= expIntBlkInterval,\
                                                        pos=[0, 0],\
                                                          wrapWidth=1.6,\
                                                            color='black')  
            expIntBlkIntervalToText.draw()
            win.flip()
            core.wait(60)
         
    return True  

def save():
    global df_data, df_tones, df_qa, df_comp

    if(df_data.shape[0] > 0):
        df_data.to_csv(data_results_filename + ".csv", encoding='utf-8',\
                        index=False)
    if df_tones.shape[0] > 0:
        df_tones.to_csv(tone_results_file_name + ".csv", encoding='utf-8',\
                         index=False)
    if df_qa.shape[0] > 0 :
        df_qa.to_csv(qa_results_file_name + ".csv", encoding='utf-8', \
                     index=False)
        
    if df_comp.shape[0]> 0 :
        df_comp.to_csv(comp_results_file_name + ".csv", encoding='utf-8', \
                     index=False)

def runExperiment(win):

    prac_test_intro =  'Press "SPACE" to start the practice rounds'

    exp_intro =  'Press "SPACE" to start the experiment'

    end = 'Thank you for participating!\n\nPress "SPACE" to END'

    introScreen(win)
    
    if( practice == 'Y'):
        instructionScreen(win, prac_test_intro, 'space')
        trainingScreen(win)

    instructionScreen(win, exp_intro, 'space')
    sendTcpTag(Stimulations.OVTK_StimulationId_ExperimentStart)
    experimentScreen(win)
    sendTcpTag(Stimulations.OVTK_StimulationId_ExperimentStop)

    instructionScreen(win, end, 'space')

    save()  
    win.close()
    core.quit() 

    
if __name__=="__main__":
    runExperiment(win)
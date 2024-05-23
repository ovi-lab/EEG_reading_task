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
info={'participant':''}
infoDlg=gui.DlgFromDict(dictionary=info, title='Setup',order=['participant'])
if not infoDlg.OK: core.quit()

PID=info['participant']
blocks= ['ND','D']

globalClock = core.Clock()
routineTimer = core.Clock() 

timeStamp=core.getAbsTime()
today=datetime.date.today()

data_results_filename = "data/reading_data/"+ "data_" + PID + "_" + data.getDateStr() 
tone_results_file_name = "data/tones/" + "tones_" + PID + "_" + data.getDateStr() 
qa_results_file_name = "data/qa/" + "tones_" + PID + "_" + data.getDateStr() 

df_data = pd.DataFrame(columns=['PID', 'Date','Timestamp', 'BlockNo', \
                                'BlockType', 'Paragraph_id', 'Reading_time'])

df_tones = pd.DataFrame(columns=['PID', 'Date','Timestamp', 'BlockNo',\
                                  'BlockType', 'KeyPressed', 'CorrectAns', \
                                    'Correct'])

df_qa = pd.DataFrame(columns=['PID', 'Date','Timestamp', 'BlockNo', \
                              'BlockType', 'Paragraph_id', 'Question_id',\
                                  'KeyPressed', 'CorrectAns', 'Correct'])


# monitor setup
win = visual.Window([1920,1080], fullscr=True, units='norm',  screen=1)

# setup keyboard
kb = keyboard.Keyboard(backend='ptb')

# define tones
sound_duration = 0.2
sound_iti = 1.5
sound_playing = False

aud1 = sound.Sound("C", octave=5, sampleRate=44100, secs=sound_duration, \
                   volume= 1)
aud2 = sound.Sound("D", octave=6, sampleRate=44100, secs=sound_duration, \
                   volume= 1)
auds = [aud1, aud2]

# define text stims
word_count_per_frame = 80

passage_stim =  visual.TextBox(win, size = (1.5, 1.5), font_size = 32, \
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


question_stim =  visual.TextStim(win, wrapWidth=1.6, pos= (0.0, 0.25))
                                 

answers_stim =  visual.TextBox(win, size = (1, 1), font_size = 32, 
                                   pos= (0.0, - 0.1), \
                                    grid_vert_justification='center'\
                                   , font_color=[1,1,1])


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
                                  wrapWidth=1.6, color='black')
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


def introScreen(win):

    intro1 = 'Press a key to begin experiment'

    ins1 = 'Here, you are going to do reading comprehension,\n\n' +  \
    'read a passage and then select the correct answers for the questions' + \
        ' \n\nPress "SPACE" for more instructions'
    ins2 =  'In some of the blocks you would here sounds\n\nSpecifically,'+ \
        ' Frequent sounds and ODD sounds\n\n'+\
            'Press "SPACE" for more instructions'
    ins3a = 'The Frequent tone sounds like this'
    ins3b = 'The ODD tone sounds like this'
    ins3c = 'You need to count the number of times you hear the ODD sound'+\
        ' in a block'


    instructionScreen(win, intro1 , 'anykey')
    instructionScreen(win, ins1, 'space')
    instructionScreen(win, ins2, 'space')
    
    
    ins3aToText = visual.TextStim(win, text= ins3a, pos=[0, 0], wrapWidth=1.6,\
                                   color='black')
    ins3aToText.draw()
    win.flip()
    core.wait(1)
    aud1.play()
    core.wait(1)
    aud1.stop()


    ins3bToText = visual.TextStim(win, text= ins3b, pos=[0, 0], wrapWidth=1.6,\
                                   color='black')  
    ins3bToText.draw()
    win.flip()
    core.wait(1)    
    aud2.play()
    core.wait(1)
    aud2.stop()

    ins3cToText = visual.TextStim(win, text= ins3c, pos=[0, 0], wrapWidth=1.6,\
                                   color='black')  
    ins3cToText.draw()
    win.flip()
    core.wait(3)


'''def extract_sentences(path, word_count):
    full_text = []

    with open(path) as f:
        for line in f:
            for word in line.split():
                full_text.append(word)
                
    sentences = []
    start = 0
    end = 0
    num_pages =  int(len(full_text)/ word_count)  \
        if len(full_text) % word_count ==0 \
            else int(len(full_text)/ word_count)  +1
    for i in range(0, num_pages):
        end = end + word_count if end + word_count < len(full_text) else \
            len(full_text)
        text_range = full_text[start: end ]
        sentence = ' '.join(text_range)
        sentences.append(sentence)
        start =  end
    return sentences '''


def read_text(path):
    with open(path) as f:
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
                current_page = " `"

    # Add any remaining text to the current page
    if current_line:
        current_page += current_line
    if current_page:
        pages.append(current_page)

    return pages


def block(win, test_type, path, block_type, block_number, paragraph_id):

    global df_data, df_tones, word_count_per_frame 

    text =  read_text(path)

    sentences_for_pages = format_text(text)

    # sentences =  extract_sentences(path, word_count_per_frame)
    number_of_pages = len(sentences_for_pages)

    np.random.seed(7)
    sound_ind = np.random.binomial(1, 0.2, 10000)
    
    
    count = 0

    continueInnerLoop = True
    continueOuterLoop = True

    current_page_number = 0
    past_page_number = -1
    audio_iter = 0
    sound_playing = False

    globalClock.reset()
    kb.clearEvents()
    kb.clock.reset()

    if(block_type == 'D'): 
        sendTcpTag(Stimulations.OVTK_StimulationId_Label_05)
    else:
         sendTcpTag(Stimulations.OVTK_StimulationId_Label_03)   


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
                            sendTcpTag(\
                                Stimulations.OVTK_StimulationId_Label_01)
                            count += 1
                        else:
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
            keys = kb.getKeys(keyList = ['escape'], clear =True)

            if(keys):
                resp = keys[0].name #take first response

                if resp=='escape':
                    continueInnerLoop = False
                    continueOuterLoop = False
                    win.close()
                    sendTcpTag(Stimulations.OVTK_StimulationId_ExperimentStop)
                    core.quit()

                kb.clearEvents()
                kb.clock.reset()
                
            if routineTimer.getTime() > 14:
                current_page_number += 1

            if routineTimer.getTime() > 12:
                opacity_level =  (14 - routineTimer.getTime()) /2 
                passage_stim.setOpacity(opacity_level)
                passage_stim.draw()
                win.flip()


            if ((current_page_number == number_of_pages)):
                if(block_type == 'D'): 
                    sendTcpTag(Stimulations.OVTK_StimulationId_Label_06)
                else:
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
        else: 
            continueOuterLoop = False  # abort routine on response
            continueInnerLoop = False          
        corr = 0
        if (textbox.text.isdigit()) and (count == int(textbox.text)):
            corr = 1 

        logging.info( test_type +' - Actual Count: ' + str(count) +'\
                      , User Count: ' + str(textbox.text))  # info, error      

        if (test_type == 'Test'):
            new_row = {'PID':PID, 'Date':today, 'Timestamp': timeStamp,\
                        'BlockNo':block_number, 'BlockType': block_type, \
                              'KeyPressed':  textbox.text , \
                                  'CorrectAns':  count, 'Correct': corr }
            # df_tones = pd.concat([df_tones, new_row], ignore_index=True)
            df_tones = df_tones.append(new_row, ignore_index=True)

    return questionsScreen(win, test_type, block_type, 
                    block_number, paragraph_id)


def questionsScreen(win, test_type, block_type, 
                    block_number, paragraph_id):
    global df_qa
     # reads from a condition file

    all_df_comp_questions = pd.read_excel('passage_qa/questionbank.xlsx')
    
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


        keys = kb.getKeys(keyList = ['a','b','c','d', 'escape'], clear =True)

        if(keys):
            resp = keys[0].name #take first response

            if resp=='escape':
                continueInnerLoop = False
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


            kb.clearEvents()
            kb.clock.reset()  

            current_question_number += 1

        if (current_question_number == number_of_questions):
              continueInnerLoop = False

    return True          


def experimentScreen(win):

    
    expIntBlkInterval = 'One minute mandatory break'
    expIntro =' Press a key to begin the block'
    
    
    # reads from a condition file
    all_df_block_setup = pd.read_excel('conditions\conditions.xlsx')

    df_block_setup = all_df_block_setup.loc[all_df_block_setup['PID'] == int(PID)]

    #todo: do a group by for PID
    number_of_blocks = df_block_setup.shape[0]


    for index, row in df_block_setup.iterrows():

        instructionScreen(win, expIntro , 'anykey')
        path = 'passages/' +  row["Paragraph_id"] + '.txt' 

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

def save():
    global df_data, df_tones 
    if (df_data.shape[0] > 0 and df_tones.shape[0] > 0 and df_qa.shape[0] > 0):
        df_data.to_csv(data_results_filename + ".csv", encoding='utf-8',\
                        index=False)
        df_tones.to_csv(tone_results_file_name + ".csv", encoding='utf-8',\
                         index=False)
        df_qa.to_csv(qa_results_file_name + ".csv", encoding='utf-8', \
                     index=False)


def runExperiment(win):

    prac_test_intro =  'Press "SPACE" to start the practice test'

    exp_intro =  'Press "SPACE" to start the experiment'

    end = 'Thank you for participating!\n\nPress "SPACE" to END'

    # introScreen(win)

    # print(trainingTrials)
    
    # if( trainingTrials == '[\'Yes\']'): 
    #     instructionScreen(win, prac_test_intro, 'space')
    #     # trainingScreen(win)

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
from psychopy import visual, event, core, data, gui, sound 
from psychopy.hardware import keyboard
import numpy as np
import pandas as pd
import random
import logging
import datetime
from sklearn.utils import shuffle
import os


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
info={'participant':'','trainingTrials': [['Yes'], ['No']], 'trainingBlockDuration': 30 ,'testBlockDuration': 60 ,'balanceCondition':[['D','ND','D','ND' ],['ND','D', 'ND','D']]}
infoDlg=gui.DlgFromDict(dictionary=info, title='Setup',order=['participant','trainingTrials' ,'trainingBlockDuration', 'testBlockDuration','balanceCondition'])
if not infoDlg.OK: core.quit()

PID=info['participant']
trainingTrials=info['trainingTrials']
blocks= info['balanceCondition']

block_duration_test = int(info['testBlockDuration'])
block_duration_train = int(info['trainingBlockDuration'])

globalClock = core.Clock()
routineTimer = core.Clock() 

timeStamp=core.getAbsTime()
today=datetime.date.today()

data_results_filename = "math_data/"+ "data_" + PID + "_" + data.getDateStr() 
tone_results_file_name = "tones/" + "tones_" + PID + "_" + data.getDateStr() 

df_data = pd.DataFrame(columns=['PID', 'Date','Timestamp', 'BlockNo', 'BlockType', 'TrialNo', 'KeyPressed', 'RT','CorrectAns', 'Correct'])

df_tones = pd.DataFrame(columns=['PID', 'Date','Timestamp', 'BlockNo', 'BlockType', 'KeyPressed', 'CorrectAns', 'Correct'])


# monitor setup
win = visual.Window([1920,1080], fullscr=True, units='norm',  screen=1)

# setup keyboard
kb = keyboard.Keyboard(backend='ptb')

# define tones
sound_duration = 0.3
sound_iti = 0.7
volume = 1
sound_playing = False

aud1 = sound.Sound("C", octave=5, sampleRate=44100, secs=sound_duration)
aud1.setVolume(volume)
aud2 = sound.Sound("D", octave=6, sampleRate=44100, secs=sound_duration)
aud2.setVolume(volume)
auds = [aud1, aud2]


#define stims
question_stim =  visual.TextStim(win, height = 0.2, pos= (0.0, 0.5), color='black')
answer_left_stim =  visual.TextStim(win, height = 0.2, pos= (-0.75, -0.5), color='black')
answer_right_stim =  visual.TextStim(win, height = 0.2,  pos= (0.75, -0.5), color='black')

textToShow = visual.TextStim(win, text= "", pos=[0, 0], wrapWidth=1.6, color='black')

question_text = visual.TextStim(win, text= 'How many times did you hear the "ODD" tone ', pos=[0, 0.3], wrapWidth=1.6, color='black')

textbox = visual.TextBox2(
     win, text='', font='Open Sans',
     pos=(0, 0),     letterHeight=0.2,
     size=(None, None), borderWidth=2.0,
     color='white',
     alignment='center',
    #  borderColor='white',
    #  colorSpace='rgb',
    #  opacity=None,
    #  bold=False, italic=False,
    #  lineSpacing=1.0,
    #  padding=0.0, 
    #  anchor='center',
    #  fillColor=None, borderColor=None,
    #  flipHoriz=False, flipVert=False, languageStyle='LTR',
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


# functions 

def myShuffle(array):
    return shuffle(array)


def instructionScreen(window,instructionText,advance):
    textToShow = visual.TextStim(win=window, text=instructionText, pos=[0, 0], wrapWidth=1.6, color='black')
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
    textToShow = visual.TextStim(win=win, text=message, pos=[0, 0], wrapWidth=1.6, color='black')
    textToShow.draw()
    win.flip()
    #whenever a message screen is displayed the experiment can end early if x is pressed
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

    ins1 = 'Here, you are going to select the correct math answer\n\nPress "SPACE" for more instructions'
    ins2 =  'In some of the blocks you would here sounds\n\nSpecifically, Frequent sounds and ODD sounds\n\nPress "SPACE" for more instructions'
    ins3a = 'The Frequent tone sounds like this'
    ins3b = 'The ODD tone sounds like this'
    ins3c = 'You need to count the number of times you hear the ODD sound in a block'


    instructionScreen(win, intro1 , 'anykey')
    instructionScreen(win, ins1, 'space')
    instructionScreen(win, ins2, 'space')
    
    
    ins3aToText = visual.TextStim(win, text= ins3a, pos=[0, 0], wrapWidth=1.6, color='black')
    ins3aToText.draw()
    win.flip()
    core.wait(1)
    aud1.play()
    core.wait(1)
    aud1.stop()


    ins3bToText = visual.TextStim(win, text= ins3b, pos=[0, 0], wrapWidth=1.6, color='black')  
    ins3bToText.draw()
    win.flip()
    core.wait(1)    
    aud2.play()
    core.wait(1)
    aud2.stop()

    ins3cToText = visual.TextStim(win, text= ins3c, pos=[0, 0], wrapWidth=1.6, color='black')  
    ins3cToText.draw()
    win.flip()
    core.wait(3)



def block(win, test_type, excel_path, block_type, block_number, block_duration):

    global df_data, df_tones 

    np.random.seed(7)
    sound_ind = np.random.binomial(1, 0.2, 10000)

    # file_path = os.path.abspath(os.path.join(current_directory, "/", excel_path))
    # filename = f"{page['title']}.png"
    # img.save(os.path.join(fileDir, filename))

    # df = pd.read_excel('conditions_math_task_v1.xlsx')
    df = pd.read_excel(file_path)

    #shuffle the math questions
    df = myShuffle(df)
    
    count = 0

    continueInnerLoop = True
    continueOuterLoop = True

    # routineForceEnded = False
    number_of_trials = df.shape[0]

    current_trial_number = 0
    past_trial_number = -1
    audio_iter = 0
    sound_playing = False

    globalClock.reset()
    kb.clearEvents()
    kb.clock.reset()

    while(continueOuterLoop):

        while(continueInnerLoop) :

            if(block_type == 'D'): 
                # Plays the sounds 
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


            if (past_trial_number <  current_trial_number):
                thisTrial = df.iloc[current_trial_number]
                question_stim.setText(thisTrial['question'])
                answer_left_stim.setText(thisTrial['left_answer'])
                answer_right_stim.setText(thisTrial['right_answer'])
                correct_side = thisTrial['correct_side']

                question_stim.draw()
                answer_left_stim.draw()
                answer_right_stim.draw()
                win.flip()

                past_trial_number += 1  

            # check and handle keyboard and mouse  
            keys = kb.getKeys(keyList = ['left','right','escape'], clear =True)

            if(keys):
                resp = keys[0].name #take first response
                rt = keys[0].rt

                if resp=='escape':
                    continueInnerLoop = False
                    continueOuterLoop = False
                    save(test_type)
                    win.close()
  
                if correct_side == 'right' and resp=='right':
                    corr = 1
                elif correct_side == 'left' and resp=='left':
                    corr = 1
                else:
                    corr = 0

                if (test_type == 'Test'):
                    new_row = {'PID':PID, 'Date':today, 'Timestamp': timeStamp, 'BlockNo':block_number, 'BlockType': block_type,  'TrialNo': current_trial_number, 'KeyPressed':  resp ,'RT': rt,  'CorrectAns':  correct_side, 'Correct': corr }
                    # df_data = pd.concat([df_data, new_row], ignore_index=True)
                    df_data = df_data.append(new_row, ignore_index=True)


                kb.clearEvents()
                kb.clock.reset()  

                current_trial_number += 1

            
            # When going over the block time limit == last block trial
            # if (current_trial_number < number_of_trials - 1) and (globalClock.getTime() >= block_duration):
            #     current_trial_number = number_of_trials - 1
            
            # if ((current_trial_number == number_of_trials)):
            #     continueInnerLoop = False
            #     mouse.mouseClock.reset()
            #     prevButtonState = mouse.getPressed()
            #     textbox.reset()
            #     textbox.setText('000')
             # ----------------------------------------------------------   
            
            # Going over the block time limit terminates the block
            if ((current_trial_number == number_of_trials) or (globalClock.getTime() >= block_duration)):
                continueInnerLoop = False
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()
                textbox.reset()
                textbox.setText('0')

                core.wait(3) 
  
        # Text box to enter the "ODD" sound count
        if(block_type == 'D'): 

            question_text.draw()
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
        if (textbox.text != '') and (count == int(textbox.text)):
            corr = 1 

        logging.info( test_type +' - Actual Count: ' + str(count) +' , User Count: ' + str(textbox.text))  # info, error      

        if (test_type == 'Test'):
            new_row = {'PID':PID, 'Date':today, 'Timestamp': timeStamp, 'BlockNo':block_number, 'BlockType': block_type,  'KeyPressed':  textbox.text ,'RT': rt,  'CorrectAns':  count, 'Correct': corr }
            # df_tones = pd.concat([df_tones, new_row], ignore_index=True)
            df_tones = df_tones.append(new_row, ignore_index=True)

    return count


def trainingScreen(win):

    prac1a = 'This is a trial block with no auditory sounds'
    prac1b = 'This is a trial block with auditory sounds'
    prac2a = 'Do the Math problems correctly  and fast as you can'
    prac2b = 'Do the Math problems correctly and fast as you can while counting the number of "ODD" sounds you hear'

    prac1aToText = visual.TextStim(win, text= prac1a, pos=[0, 0], wrapWidth=1.6, color='black')  
    prac1aToText.draw()
    win.flip()
    core.wait(5)

    prac1aToText = visual.TextStim(win, text= prac2a, pos=[0, 0], wrapWidth=1.6, color='black')  
    prac1aToText.draw()
    win.flip()
    core.wait(5)

    block(win,  'Practice', 'conditions/b1.xlsx','ND', block_number = None,  block_duration = block_duration_train)

    prac1aToText = visual.TextStim(win, text= prac1b, pos=[0, 0], wrapWidth=1.6, color='black')  
    prac1aToText.draw()
    win.flip()
    core.wait(5)

    prac1aToText = visual.TextStim(win, text= prac2b, pos=[0, 0], wrapWidth=1.6, color='black')  
    prac1aToText.draw()
    win.flip()
    core.wait(5)

    block(win,  'Practice', 'conditions/b1.xlsx','D', block_number = None,  block_duration = block_duration_train)

    return True

def experimentScreen(win):

    expIntBlkInterval = 'One minute mandatory break'
    expIntro =' Press a key to begin the block'
    
    
    # reads from a condition file
    df_block_setup = pd.read_excel('conditions/block_conditions.xlsx')

    # block order 
    if ( block == '[\'ND\',\'D\', \'ND\',\'D\']'):
        first_D_block = df_block_setup.iloc[0]
        df_block_setup.drop([0], inplace = True)
        df_block_setup = df_block_setup.append(first_D_block)

    number_of_blocks = df_block_setup.shape[0]


    for index, row in df_block_setup.iterrows():

        instructionScreen(win, expIntro , 'anykey')

        block(win, 'Test', row["block_condition_file_path"], row["block_type"], block_number = index, block_duration = block_duration_test)
                
        if not ((index +1) == number_of_blocks): 
            expIntBlkIntervalToText = visual.TextStim(win, text= expIntBlkInterval, pos=[0, 0], wrapWidth=1.6, color='black')  
            expIntBlkIntervalToText.draw()
            win.flip()
            core.wait(60)

    return True  



def save(test_type):
    global df_data, df_tones 
    if (test_type == 'Test'):
        df_data.to_csv(data_results_filename + ".csv", encoding='utf-8', index=False)
        df_tones.to_csv(tone_results_file_name + ".csv", encoding='utf-8', index=False)


def runExperiment(win):

    prac_test_intro =  'Press "SPACE" to start the practice test'

    exp_intro =  'Press "SPACE" to start the experiment'

    end = 'Thank you for participating!\n\nPress "SPACE" to END'

    introScreen(win)

    print(trainingTrials)
    
    if( trainingTrials == '[\'Yes\']'): 
        instructionScreen(win, prac_test_intro, 'space')
        trainingScreen(win)

    instructionScreen(win, exp_intro, 'space')
    experimentScreen(win)

    instructionScreen(win, end, 'space')

    save()  
    win.close()
    core.quit() 

    
if __name__=="__main__":
    runExperiment(win)


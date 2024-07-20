@REM @echo off
@REM cmd /k "C:/Users/erang/.conda/envs/psychopy/python.exe c:/Users/erang/Desktop/Reading_task/psychopy_experiment/task.py"


@echo off
@REM cmd /k "cd /d C:\Users\erang\.conda\envs\psychopy\Scripts & activate & cd /d    c:/Users/erang/Desktop/Reading_task/psychopy_experiment & python task.py"


rem Step 1: Activate Miniconda, use the actual path where your Miniconda/Ananconda is installed.
rem call "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)" 

rem Step 2: Activate Conda environment. 
call conda activate psychopy

rem Step 3: Change directory to the desired folder.
cd /d "c:/Users/erang/Desktop/Reading_task/psychopy_experiment"

rem Step 4: Run the Python script.
python task.py

rem Step 5: Keep the command prompt open after execution (optional)
cmd /k
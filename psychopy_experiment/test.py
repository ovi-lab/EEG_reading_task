import subprocess

# val = subprocess.run(["ls", "-l"], shell= True) 

val = subprocess.run(["ov_experiment\\dist\\openvibe-3.5.0-64bit\\openvibe-designer.cmd", 
                      "--play",
                    #   "--no-gui",
                      "ov_experiment\\scenarios\\recorder.xml", 
                      "--define",
                      "Participant",
                    "5"], shell= True) 


# val = subprocess.run(["ov_experiment\\dist\openvibe-3.5.0-64bit\openvibe-designer.cmd", 
#                       "--define myFile ov_experiment\scenarios\recorder.xml",
#                     #   "",
#                     #   "",
#                       "--play",
#                     #   "--no-gui", 
#                       "myFile"], shell= True) 

# 0 for success  ov_experiment\dist\openvibe-3.5.0-64bit\openvibe-designer.cmd
# 2 from cannot access
print(val)
print(str(val.returncode))  

ov_experiment\\dist\\openvibe-3.5.0-64bit\\openvibe-designer.cmd  --define Participant 5 --play ov_experiment\\scenarios\\recorder.xml

ov_experiment\\dist\\openvibe-3.5.0-64bit\\openvibe-designer.cmd -a ov_experiment\\scenarios\\recorder.xml --define Participant 5

openvibe-designer.cmd -a path/to/your/scenario.xml --define MyConfiguration=Value
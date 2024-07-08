"""
Contains utility functions that do not require any other specific dependencies
"""

import os
from datetime import datetime
import sys

def mkdir(folderpath):
    #make the folder if it doesn't exist
    folderpath=ospath(folderpath)
    if not(os.path.isdir(folderpath)):
        os.mkdir(folderpath)

class cprint():
    """
    Class to write 'print' outputs to console and to a given textfile. Use as follows:
    filepath=ospath('testfile.txt')
    with open(filepath,'w') as file:
        t=cprint(file) 
        t.print("Hello World")
    """
    def __init__(self,resultsfile):
        self.resultsfile = resultsfile
    def print(self,*args,**kwargs):    
        temp=sys.stdout 
        print(*args,**kwargs)
        sys.stdout=self.resultsfile #assign console output to a text file
        print(*args,**kwargs)
        sys.stdout=temp #set stdout back to console output

def timer(start_time):
    """
    Simple timer function. Returns the difference (seconds) between the current time and the start time.
    """
    end_time=datetime.now()
    runtime=end_time-start_time
    return runtime.total_seconds()

def now():
    now=datetime.now()
    return now.strftime("%H:%M:%S")

class clock():
    """
    How to use
    c=hcpalign_utils.clock()
    print(c.time())
    """
    def __init__(self):
        self.start_time=datetime.now()       
    def time(self):
        end_time=datetime.now()
        runtime=end_time-self.start_time
        value='{:.1f}s'.format(runtime.total_seconds())
        return value

def getloadavg():
    """
    Prints recent CPU usage
    """
    import psutil
    print([x / psutil.cpu_count() for x in psutil.getloadavg()])

def memused(): 
    """
    Prints RAM memory usage
    """
    import os, psutil
    process = psutil.Process(os.getpid())
    return f'Python mem: {process.memory_info().rss/1e9:.2f} GB, PC mem: {psutil.virtual_memory()[3]/1e9:.2f}/{psutil.virtual_memory()[0]/1e9:.0f} GB'

def sizeof(input_obj):
    """
    Attempts to return the size of a Python variable (e.g. an array) in a human readable format. May not always be correct, particulary for nested arrays/lists or other complex objects
    """

    import sys
    import gc

    def sizeof_fmt(num, suffix='B'):
        #called by sizeof
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)
    def _sizeof(input_obj):
        #called by sizeof
        memory_size = 0
        ids = set()
        objects = [input_obj]
        while objects:
            new = []
            for obj in objects:
                if id(obj) not in ids:
                    ids.add(id(obj))
                    memory_size += sys.getsizeof(obj)
                    new.append(obj)
            objects = gc.get_referents(*new)
        return memory_size


    if type(input_obj)==list:
        return sizeof_fmt(sum([_sizeof(i) for i in input_obj]))
    else:
        return sizeof_fmt(_sizeof(input_obj))



def ospath(x,windows_machine_name='DESKTOP-EGSQF3A'):
    """
    This function specific to Jayson's PC
    If the operating system is Windows, this function converts a path to a Windows path, and vice versa for linux.
    E.g. "D:\\FORSTORAGE\\Data\\HCP_S1200" (windows) to "/mnt/d/FORSTORAGE/Data/HCP_S1200" (linux)
    Parameters:
    -----------
    x: str
        Path to be converted
    windows_machine_name: str
        Name of the Windows machine. Default is 'DESKTOP-EGSQF3A' (for Jayson's PC)
    Returns:
    --------
    x: str
        Converted path
    """

    import socket
    hostname=socket.gethostname()
    if hostname==windows_machine_name:
        if os.name=='nt' and x[0]=='/': #Windows
            return '{}:\\{}'.format(x[5].upper(),x[7:].replace('/','\\'))
        elif os.name=='posix' and x[0]!='/': #WSL2 on Windows PC
            return '/mnt/{}/{}'.format(x[0].lower(),x[3:].replace('\\','/'))
        else:
            return x
    else: #Linux based system
        return x.replace('\\','/')
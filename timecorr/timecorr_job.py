#!/usr/bin/python

# create a bunch of job scripts
from __future__ import absolute_import
from builtins import zip
from builtins import str
from builtins import range
from .config import config
from subprocess import call
import os
import socket
import getpass
import datetime as dt

# ====== MODIFY ONLY THE CODE BETWEEN THESE LINES ======
# each job command should be formatted as a string
job_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'decoding_analysis.py')

job_commands = [x[0]+" "+str(x[1])+" "+str(x[2]) for x in zip([job_script]*600,['../../pieman-paragraph/']*100+['../../pieman-resting/']*100+['../../pieman-intact/']*100+['../../pieman-word/']*100+['../../sherlock/']*100+['../../forrest/']*100,list(range(100))*6)]

#job_commands = [job_script+' ../../forrest/ 100 10']

#job_commands = map(lambda x: x[0]+" "+str(x[1])+" "+x[2]+" "+str(x[3]), zip([job_script]*100,['../../forrest/']*100,['10']*100,range(100)))   

#job_commands = map(lambda x: x[0]+" "+str(x[1])+" "+x[2]+" "+x[3], zip([job_script]*5,['../../pieman-paragraph/','../../pieman-resting/','../../pieman-intact/','../../pieman-word/','../../sherlock/'],['100']*5,['10']*5))

#job_commands = map(lambda x: x[0]+" "+str(x[1])+" "+x[2]+" "+str(x[3]), zip([job_script]*500,['../../pieman-paragraph/']*100+['../../pieman-resting/']*100+['../../pieman-intact/']*100+['../../pieman-word/']*100+['../../sherlock/']*100,['10']*500,range(100)*5))

# job_names should specify the file name of each script (as a list, of the same length as job_commands)
job_names = [str(x)+'_levelup.sh' for x in range(len(job_commands))]
# ====== MODIFY ONLY THE CODE BETWEEN THESE LINES ======

assert(len(job_commands) == len(job_names))


# job_command is referenced in the run_job.sh script
# noinspection PyBroadException,PyUnusedLocal
def create_job(name, job_command):
    # noinspection PyUnusedLocal,PyShadowingNames
    def create_helper(s, job_command):
        x = [i for i, char in enumerate(s) if char == '<']
        y = [i for i, char in enumerate(s) if char == '>']
        if len(x) == 0:
            return s

        q = ''
        index = 0
        for i in range(len(x)):
            q += s[index:x[i]]
            unpacked = eval(s[x[i]+1:y[i]])
            q += str(unpacked)
            index = y[i]+1
        return q

    # create script directory if it doesn't already exist
    try:
        os.stat(config['scriptdir'])
    except:
        os.makedirs(config['scriptdir'])

    template_fd = open(config['template'], 'r')
    job_fname = os.path.join(config['scriptdir'], name)
    new_fd = open(job_fname, 'w+')

    while True:
        next_line = template_fd.readline()
        if len(next_line) == 0:
            break
        new_fd.writelines(create_helper(next_line, job_command))
    template_fd.close()
    new_fd.close()
    return job_fname


# noinspection PyBroadException
def lock(lockfile):
    try:
        os.stat(lockfile)
        return False
    except:
        fd = open(lockfile, 'w')
        fd.writelines('LOCK CREATE TIME: ' + str(dt.datetime.now()) + '\n')
        fd.writelines('HOST: ' + socket.gethostname() + '\n')
        fd.writelines('USER: ' + getpass.getuser() + '\n')
        fd.writelines('\n-----\nCONFIG\n-----\n')
        for k in list(config.keys()):
            fd.writelines(k.upper() + ': ' + str(config[k]) + '\n')
        fd.close()
        return True


# noinspection PyBroadException
def release(lockfile):
    try:
        os.stat(lockfile)
        os.remove(lockfile)
        return True
    except:
        return False


script_dir = config['scriptdir']
lock_dir = config['lockdir']
lock_dir_exists = False
# noinspection PyBroadException
try:
    os.stat(lock_dir)
    lock_dir_exists = True
except:
    os.makedirs(lock_dir)

# noinspection PyBroadException
try:
    os.stat(config['startdir'])
except:
    os.makedirs(config['startdir'])

locks = list()
for n, c in zip(job_names, job_commands):
    # if the submission script crashes before all jobs are submitted, the lockfile system ensures that only
    # not-yet-submitted jobs will be submitted the next time this script runs
    next_lockfile = os.path.join(lock_dir, n+'.LOCK')
    locks.append(next_lockfile)
    if not os.path.isfile(os.path.join(script_dir, n)):
        if lock(next_lockfile):
            next_job = create_job(n, c)

            if ('discovery' in socket.gethostname()) or ('nodoli' in socket.gethostname()):
                submit_command = 'echo "[SUBMITTING JOB: ' + next_job + ']"; qsub'
            else:
                submit_command = 'echo "[RUNNING JOB: ' + next_job + ']"; sh'

            call(submit_command + " " + next_job, shell=True)

# all jobs have been submitted; release all locks
for l in locks:
    release(l)
if not lock_dir_exists:  # remove lock directory if it was created here
    os.rmdir(lock_dir)

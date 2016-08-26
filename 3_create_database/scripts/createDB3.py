# -*- coding: utf-8 -*-

# In this script we want to create a database containing all the features required for analysing the evicted tasks
# who have not been killed/finished/lost. We would like to know :
# Find the time stamp and machine ids that we want their avail capacity
# 1. What was the exact capacity of the machine when a task was assigned to it.
# 2. 

# @author: reyhane_askari
# Universite de Montreal, Jan 2016

from os import chdir, listdir
from pandas import read_csv
from os import path
from random import randint, sample, seed
from collections import OrderedDict
from pandas import DataFrame
import numpy as np 
import matplotlib.pyplot as plt
import csv
import codecs
from sets import Set


chdir('/home/askrey/Dropbox/Project_step_by_step/2_find_violations/csvs')
task_events_csv_colnames = ['time', 'missing', 'job_id', 'task_idx', 'machine_id', 'event_type', 'user', 'sched_cls', 
                            'priority', 'cpu_requested', 'mem_requested', 'disk', 'restriction']         
evictedtasks = OrderedDict([])
violatedtasks = OrderedDict([])
for key, val in csv.reader(open("not_finished_evictedtasks.csv")):
    evictedtasks[key] = val

machines_dictionary = OrderedDict([])

#load machine events table:
chdir('/home/askrey/Final_project') 
reader = csv.reader(codecs.open('part-00000-of-00001.csv','rU','utf-8'))

# key of machines_dictionary is the primary fields of the machine events table (time, machine id) 
# other fields: event type, platform id, CUPs, memory

for row in reader:
  machines_dictionary[(row[0],row[1])] = row[2:]

machines_set = set()

#for fn in sorted(listdir('task_events')):
fp = path.join('task_events', sorted(listdir('task_events'))[0])
task_events_df = read_csv(fp, header = None, index_col = False, names = task_events_csv_colnames, compression = 'gzip')

for index, event in task_events_df.iterrows():
  if str(event['machine_id']) != 'nan':
    machines_set.add(int(event['machine_id']))
    
    
    # if (event['job_id'], event['task_idx']) in violatedtasks:
    #   violatedtasks[event['job_id'],event['task_idx']][0].append(event['time'])
    #   violatedtasks[event['job_id'],event['task_idx']][1].append(event['machine_id'])  
        

    # elif ("("+str(event['job_id'])+ ", "+ str(event['task_idx'])+")") in evictedtasks:
    #   violatedtasks[event['job_id'],event['task_idx']] = [[event['time']],[event['machine_id']]]


# moshkel alan in hast k ye jahaE event ha hanooz machine barashoon assign nashode vase hamin hast k machine id nan hast

# writer = csv.writer(open('/home/askrey/Dropbox/Databases/testDB6.csv', 'wb'))
# for key, value in violatedtasks.items():
#    writer.writerow([key, value])
print sorted(machines_set)

# -*- coding: utf-8 -*-

# In this script we find all the history behind an evicted task that has not been finished/killed/failed/lost.
# We find how many times it has been submitted and what were the events related to this task. 
# The violatedtasks dictionary is the one that is we are looking for. It has only one entry
# for each (jobid, taskindex) and the rest is stored in a multidimensional array.
# This database is a test as it is only using 1/500 of the task_events table. 
# Also the not_finished_evictedtasks is only for the first(1/500) part of the tasks_events table.
# Since the script assumed that the machine state changes in the machine table when a task is added, 
# it is not getting the right results. The problem is that the machines table is only updated when a
# machine is added so it is just the events of the machines, in order to find the available cpu, 
# memory and disk of a specific machine at the time that a task is assigned to a machine, 
# we need to have more complicated calculations.(refer to createDB3) 

# @author: reyhane_askari
# Universite de Montreal, Dec 2015

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

#for fn in sorted(listdir('task_events')):
fp = path.join('task_events',sorted(listdir('task_events'))[0])
task_events_df = read_csv(fp, header = None, index_col = False, names = task_events_csv_colnames, compression = 'gzip')

for index, event in task_events_df.iterrows():
    
    if (event['job_id'], event['task_idx']) in violatedtasks:
      violatedtasks[event['job_id'],event['task_idx']][0].append(event['time'])
      violatedtasks[event['job_id'],event['task_idx']][2].append(event['machine_id'])
      violatedtasks[event['job_id'],event['task_idx']][3].append(event['event_type'])
      violatedtasks[event['job_id'],event['task_idx']][11].append((machines_dictionary[(str(event['time']),str(event['machine_id']))] if (str(event['time']), str(event['machine_id'])) in machines_dictionary else 0))

    elif ("("+str(event['job_id'])+ ", "+ str(event['task_idx'])+")") in evictedtasks:
      violatedtasks[event['job_id'],event['task_idx']] = [[event['time']],event['missing'],[event['machine_id']],
							  [event['event_type']], event['user'], event['sched_cls'], event['priority'], event['cpu_requested'],
							  event['mem_requested'], event['disk'], event['restriction'], 
							  [(machines_dictionary[(str(event['time']),str(event['machine_id']))] if (str(event['time']), str(event['machine_id'])) in machines_dictionary else 0 )]]

# moshkel alan in hast k ye jahaE event ha hanooz machine barashoon assign nashode vase hamin hast k machine id nan hast

writer = csv.writer(open('/home/askrey/Dropbox/Databases/testDB5.csv', 'wb'))
for key, value in violatedtasks.items():
   writer.writerow([key, value])

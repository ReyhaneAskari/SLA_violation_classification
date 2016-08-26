# -*- coding: utf-8 -*-

# In this script we find the avilable resources of the machines that we have already stored in find_machine_resources step. 
# We will find the 

# 9 Feb 2016
# @author: reyhane_askari
# Universite de Montreal, DIRO

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


chdir('/home/askrey/Final_project/')
task_events_csv_colnames = ['time', 'missing', 'job_id', 'task_idx', 'machine_id', 'event_type', 'user', 'sched_cls', 
                            'priority', 'cpu_requested', 'mem_requested', 'disk', 'restriction'] 

machine_is_avail = {}

for fn in sorted(listdir('task_events')):
    
  fp = path.join('task_events', fn)
  task_events_df = read_csv(fp, header=None, index_col=False, compression='gzip', 
                              names=task_events_csv_colnames)
    
  for index, event in task_events_df.iterrows():
      if str(event['machine_id']) == 'NaN':
	machine_is_avail[(event['job_id'],event['task_idx'])] = 1

writer = csv.writer(open('/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs/machine_is_avail.csv', 'wb'))
for key, value in machine_is_avail.items():
    writer.writerow([key, value])

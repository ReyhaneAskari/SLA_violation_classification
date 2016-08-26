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


chdir('/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs')


machine_resources = OrderedDict([])
violatedtasks = OrderedDict([])
sample_moments = set()

for key, val in csv.reader(open("machines_dictionary.csv")):
    machine_resources[key] = val
    sample_moments.add(int(eval(val)[0]))

snapshot_moment = randint(0, 250619902822)

chdir('/home/askrey/Final_project/')
task_events_csv_colnames = ['time', 'missing', 'job_id', 'task_idx', 'machine_id', 'event_type', 'user', 'sched_cls', 
                            'priority', 'cpu_requested', 'mem_requested', 'disk', 'restriction'] 

tasks_dict = {}
samples_dicts = OrderedDict([])
sample_moments_iterator = iter(sorted(sample_moments))
current_sample_moment = next(sample_moments_iterator)


#for fn in sorted(listdir('task_events')):
fp = path.join('task_events',sorted(listdir('task_events'))[0])
    
#fp = path.join('task_events', fn)
task_events_df = read_csv(fp, header=None, index_col=False, compression='gzip', 
                              names=task_events_csv_colnames)
    
for index, event in task_events_df.iterrows():
        
        if current_sample_moment is not None and event['time'] > current_sample_moment:
            tmp_tasks_df = DataFrame(tasks_dict.values())
            samples_dicts[current_sample_moment] = ({'time' : current_sample_moment, 
                                                     'cpu_requested' : np.nansum(tmp_tasks_df['cpu_requested']), 
                                                     'mem_requested' : np.nansum(tmp_tasks_df['mem_requested'])})
            try:
                current_sample_moment = next(sample_moments_iterator)
            except StopIteration:
                current_sample_moment = None
               
            
        if event['event_type'] in [0, 7, 8]:
            tasks_dict[(event['job_id'], event['task_idx'])] = {'task_id' : (event['job_id'], event['task_idx']),
                                                                'machine_id' : event['machine_id'],
                                                                'cpu_requested' : event['cpu_requested'], 
                                                                'mem_requested' : event['mem_requested']}
        elif event['event_type'] in [2, 3, 4, 5, 6]:
            del tasks_dict[(event['job_id'], event['task_idx'])]
            
    
samples_df = DataFrame(samples_dicts.values())
samples_df.to_csv('/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs/samples_df3.csv')






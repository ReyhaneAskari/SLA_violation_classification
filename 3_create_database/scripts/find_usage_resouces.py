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
#key is machine_id and value is all the other stuff in machine events table
    machine_resources[eval(key)] = eval(val)
    sample_moments.add(int(eval(val)[0]))

#print machine_resources[5][0] => prints the time
#snapshot_moment = randint(0, 250619902822)

chdir('/home/askrey/Final_project/')
task_events_csv_colnames = ['time', 'missing', 'job_id', 'task_idx', 'machine_id', 'event_type', 'user', 'sched_cls', 
                            'priority', 'cpu_requested', 'mem_requested', 'disk', 'restriction'] 

tasks_dict = {}
samples_dicts = OrderedDict([])
sample_moments_iterator = iter(sorted(sample_moments))
current_sample_moment = next(sample_moments_iterator)


for fn in sorted(listdir('task_events')):
    
  fp = path.join('task_events', fn)
  task_events_df = read_csv(fp, header=None, index_col=False, compression='gzip', 
                              names=task_events_csv_colnames)
    
  for index, event in task_events_df.iterrows():
     if event['machine_id'] in machine_resources:
	 if event['time'] > machine_resources[event['machine_id']][0]:
	    tmp_tasks_df = DataFrame(tasks_dict(event['machine_id']))
	    print tmp_tasks_df 
	    samples_dicts[event['machine_id']] = ({'time' : current_sample_moment, 
						   'machine_id': event['machine_id'],
                                                   'cpu_requested' : np.nansum(tmp_tasks_df['cpu_requested']), 
                                                   'mem_requested' : np.nansum(tmp_tasks_df['mem_requested'])})
	            
         if event['event_type'] in [0, 7, 8]:
	    temp_events = {}
	    temp_events[(event['job_id'], event['task_idx'])] = {'task_id' : (event['job_id'], event['task_idx']),
								 'machine_id' : event['machine_id'],
								 'cpu_requested' : event['cpu_requested'], 
								 'mem_requested' : event['mem_requested']}

            tasks_dict[event['machine_id']] = temp_events

         elif event['event_type'] in [2, 3, 4, 5, 6]:
	    del tasks_dict['machine_id'][(event['job_id'], event['task_idx'])]
    
samples_df = DataFrame(samples_dicts.values())
samples_df.to_csv('/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs/samples_df4.csv')


task_usage_csv_colnames=['starttime', 'endtime', 'job_id', 'task_idx', 'machine_id', 'cpu_usage', 'mem_usage', 
                         'assigned_mem', 'unmapped_cache_usage', 'page_cache_usage', 'max_mem_usage', 'disk_io_time', 
                         'max_disk_space', 'max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai', 'sampling_rate', 'agg_type']

for moment in samples_dicts:
    samples_dicts[moment].update({'cpu_usage' : 0.0, 'mem_usage' : 0.0})
    

for fn in sorted(listdir('task_usage')):

    fp = path.join('task_usage', fn)
#fp = path.join('task_usage',sorted(listdir('task_usage'))[0])
    task_usage_df = read_csv(fp, header=None, index_col=False, compression='gzip', names=task_usage_csv_colnames)

    laststart = max(task_usage_df['starttime'])
      for machine in machine_resources:
	  moment = machine[task_usage_df['machine_id'][0]
	  task_usage_moment_df = task_usage_df[(task_usage_df['starttime'] <= moment]) & 
					      (moment <= task_usage_df['endtime']) & machine_resources]
	  samples_dicts[moment]['cpu_usage'] += np.nansum(task_usage_moment_df['cpu_usage'])
	  samples_dicts[moment]['mem_usage'] += np.nansum(task_usage_moment_df['mem_usage'])


samples_df = DataFrame(samples_dicts.values())

samples_df.to_csv('/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs/usage_resouces2.csv')
   





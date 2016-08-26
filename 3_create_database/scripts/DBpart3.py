# -*- coding: utf-8 -*-

# This script will read the 

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

def get_evictedevents_times():
  chdir('/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs')
  evictedtimes = set()
  columns = ['key','value']
  reader = read_csv("DB5.csv", header = None, index_col = False, names = columns)

  for index, event in reader.iterrows():
      for i in eval(event.values[1].split("[")[2].split(']')[0]):
	evictedtimes.add(i)

  evictedtimes_sorted = sorted(evictedtimes)
  return evictedtimes_sorted


seed(83)
sample_moments = get_evictedevents_times()
snapshot_moment = randint(0, 250619902822)  
chdir('/home/askrey/Final_project/')
task_events_csv_colnames = ['time', 'missing', 'job_id', 'task_idx', 'machine_id', 'event_type', 'user', 'sched_cls', 
                            'priority', 'cpu_requested', 'mem_requested', 'disk', 'restriction']

tasks_dict = {}
samples_dicts = OrderedDict([])
sample_moments_iterator = iter(sample_moments)
current_sample_moment = next(sample_moments_iterator)
tasks_df = None

task_usage_csv_colnames=['starttime', 'endtime', 'job_id', 'task_idx', 'machine_id', 'cpu_usage', 'mem_usage', 
                         'assigned_mem', 'unmapped_cache_usage', 'page_cache_usage', 'max_mem_usage', 'disk_io_time', 
                         'max_disk_space', 'max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai', 'sampling_rate', 'agg_type']

for moment in samples_dicts:
    samples_dicts[moment].update({'cpu_usage' : 0.0, 'mem_usage' : 0.0, 'assigned_mem' : 0.0})
    
for task in tasks_dict:
    tasks_dict[task].update({'cpu_usage' : 0.0, 'mem_usage' : 0.0, 'assigned_mem' : 0.0,'in_events' : True, 'in_usage' : False})
    
for fn in sorted(listdir('task_usage')):

    fp = path.join('task_usage', fn)
    task_usage_df = read_csv(fp, header = None, index_col = False, compression = 'gzip', names = task_usage_csv_colnames)

    laststart = max(task_usage_df['starttime'])
    if laststart > max(sample_moments) and laststart > snapshot_moment:
        break
        
    for moment in samples_dicts:
        task_usage_moment_df = task_usage_df[(task_usage_df['starttime'] <= moment) & 
                                             (moment <= task_usage_df['endtime'])]
        samples_dicts[moment]['cpu_usage'] += np.nansum(task_usage_moment_df['cpu_usage'])
        samples_dicts[moment]['mem_usage'] += np.nansum(task_usage_moment_df['mem_usage'])
        samples_dicts[moment]['assigned_mem'] += np.nansum(task_usage_moment_df['assigned_mem'])
            
    task_usage_snapshot_df = task_usage_df[(task_usage_df['starttime'] <= snapshot_moment) &
                                           (snapshot_moment <= task_usage_df['endtime'])]
    for index, usage in task_usage_snapshot_df.iterrows():
        task_id = (usage['job_id'], usage['task_idx'])
        if task_id in tasks_dict:
            tasks_dict[task_id].update({'cpu_usage' : usage['cpu_usage'], 'mem_usage' : usage['mem_usage'], 'assigned_mem' : usage['assigned_mem'], 
                                        'in_events': True, 'in_usage' : True})
        else:
            tasks_dict[task_id] = {'cpu_requested' : 0.0, 'mem_requested' : 0.0, 
                                   'cpu_usage' : usage['cpu_usage'], 'mem_usage' : usage['mem_usage'], 'assigned_mem' : usage['assigned_mem'],
                                   'in_events' : False, 'in_usage' : True}

                    
samples_df = DataFrame(samples_dicts.values())
samples_df.to_csv('/home/askrey/Dropbox/Databases/test3.csv')
tasks_df.to_csv('/home/askrey/Dropbox/Databases/test4.csv')

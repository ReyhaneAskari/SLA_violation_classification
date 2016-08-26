# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from os import chdir, listdir
from pandas import read_csv
from os import path
from random import randint, sample, seed
from collections import OrderedDict
from pandas import DataFrame, Series
import numpy as np 
import csv
import codecs

chdir('/home/askrey/Final_project')
fp = path.join('task_events',sorted(listdir('task_events'))[0])

task_events_csv_colnames = ['time', 'missing', 'job_id', 'task_idx', 'machine_id', 'event_type', 'user', 'sched_cls', 
                            'priority', 'cpu_requested', 'mem_requested', 'disk', 'restriction'] 

raw_data = pd.read_csv(fp, header = None, index_col = False, names = task_events_csv_colnames,
                       compression = 'gzip', usecols = [1,2,3,5,7,8,9,10,11])

# events that have a missing field are not valid.
data = raw_data[raw_data.missing.isnull()]

# for now we are omitting time and since each task in in the data servral times
# (with different event types) we are just using the submit(event_type = 0) to avoid 
data = data.loc[data['event_type'] == 0]
data = data.drop(['event_type','missing'], axis=1)


# -------------------------------------------------------------------------------------------
# Now we need to find out which tasks were evicted and not finished (definition of violation)
# not_finished_eveicted_tasks.csv contains all the (job_id,task_idx)s that have been evicted
# at least once but not finished at the end. 

# load violated_tasks
violated_tasks = OrderedDict([])
for key, val in csv.reader(open("/home/askrey/Dropbox/Project_step_by_step/2_find_violations/csvs/gnot_finished_evictedtasks.csv")):
    violated_tasks[eval(key)] = val

violated_series = Series()

for index,event in data.iterrows():
    if (event['job_id'],event['task_idx']) in violated_tasks:
        violated_series.set_value(index, 1)
    else:
        violated_series.set_value(index, 0)

# data = pd.merge(data,violated_series,left_index=True,right_index=True)
violated_series.to_csv('/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs/violated_series.csv')
# data.to_csv('/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs/full_db1_1.csv')
# -*- coding: utf-8 -*-
from os import chdir, listdir
from pandas import read_csv
from os import path
from collections import OrderedDict
import csv

task_events_csv_colnames = [
    'time', 'missing', 'job_id', 'task_idx',
    'machine_id', 'event_type', 'user', 'sched_cls',
    'priority', 'cpu_requested', 'mem_requested',
    'disk', 'restriction']
evictedtasks = OrderedDict([])
chdir('/home/askrey/Final_project')

for fn in sorted(listdir('task_events')):
    fp = path.join('task_events', fn)
    task_events_df = read_csv(
        fp, header=None, index_col=False,
        names=task_events_csv_colnames, compression='gzip')

    for index, event in task_events_df.iterrows():
        if (event['job_id'], event['task_idx']) in evictedtasks:
            evictedtasks[event['job_id'], event['task_idx']].append(event['event_type'])

        if int(event['event_type']) == 2:
            evictedtasks[event['job_id'], event['task_idx']] = [2]

not_finished_evictedtasks = evictedtasks.copy()

for key, value in not_finished_evictedtasks.iteritems():
    if any(x in value for x in [3, 4, 5, 6]):
        del not_finished_evictedtasks[key]


writer = csv.writer(open(
    '/home/askrey/Dropbox/Project_step_by_step/2_find_violations/csvs/violated_all.csv', 'wb'))
for key, value in not_finished_evictedtasks.items():
     writer.writerow([key, value])

# -*- coding: utf-8 -*-
from os import chdir, listdir
from pandas import read_csv
from os import path
from collections import OrderedDict
import csv

chdir('/home/askrey/Final_project')
task_events_csv_colnames = [
    'time', 'missing', 'job_id', 'task_idx',
    'machine_id', 'event_type', 'user', 'sched_cls',
    'priority', 'cpu_requested', 'mem_requested',
    'disk', 'restriction']
evictedtasks = OrderedDict([])

for fn in sorted(listdir('task_events')):
    fp = path.join('task_events', fn)
    task_events_df = read_csv(
        fp, header=None, index_col=False,
        names=task_events_csv_colnames, compression='gzip')

    for index, event in task_events_df.iterrows():
        if (event['job_id'], event['task_idx']) in evictedtasks:
            evictedtasks[event['job_id'],
                event['task_idx']].append(event['event_type'])

        if int(event['event_type']) == 2:
            evictedtasks[event['job_id'], event['task_idx']] = [2]

writer = csv.writer(open('evictedfinalgeneral.csv', 'wb'))
for key, value in evictedtasks.items():
    writer.writerow([key, value])

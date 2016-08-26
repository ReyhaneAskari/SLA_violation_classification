# -*- coding: utf-8 -*-
from os import listdir, chdir
from pandas import read_csv
from os import path
from random import randint, sample, seed
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from pandas import DataFrame


chdir('/home/askrey/Final_project/')
task_events_csv_colnames = [
    'time', 'missing', 'job_id', 'task_idx',
    'machine_id', 'event_type', 'user',
    'sched_cls', 'priority', 'cpu_requested',
    'mem_requested', 'disk', 'restriction']
task_event_df = read_csv(
    path.join('task_events',
              'part-00499-of-00500.csv.gz'),
    header=None, index_col=False,
    compression='gzip', names=task_events_csv_colnames)

seed(83)
sample_moments = sorted(sample(xrange(250619902823), 200))
snapshot_moment = randint(0, 250619902822)
print snapshot_moment

tasks_dict = {}
samples_dicts = OrderedDict([])
sample_moments_iterator = iter(sample_moments)
current_sample_moment = next(sample_moments_iterator)
tasks_df = None


for fn in sorted(listdir('task_events')):
    fp = path.join('task_events', fn)
    task_events_df = read_csv(
        fp, header=None, index_col=False,
        compression='gzip', names=task_events_csv_colnames)

    for index, event in task_events_df.iterrows():
        if (current_sample_moment is not None and
                event['time'] > current_sample_moment):
            tmp_tasks_df = DataFrame(tasks_dict.values())
            samples_dicts[current_sample_moment] = (
                {'time': current_sample_moment,
                 'cpu_requested': np.nansum(tmp_tasks_df['cpu_requested']),
                 'mem_requested': np.nansum(tmp_tasks_df['mem_requested'])})
            try:
                current_sample_moment = next(sample_moments_iterator)
            except StopIteration:
                current_sample_moment = None

        if tasks_df is None and event['time'] > snapshot_moment:
            tasks_df = DataFrame(tasks_dict.values())

        if event['event_type'] in [0, 7, 8]:
            tasks_dict[(event['job_id'], event['task_idx'])] = {
                'task_id': (event['job_id'], event['task_idx']),
                'machine_id': event['machine_id'],
                'cpu_requested': event['cpu_requested'],
                'mem_requested': event['mem_requested']}
        elif event['event_type'] in [2, 3, 4, 5, 6]:
            del tasks_dict[(event['job_id'], event['task_idx'])]

    if tasks_df is not None and current_sample_moment is None:
        break

samples_df = DataFrame(samples_dicts.values())

machines_dict = {}
sample_moments_iterator = iter(sample_moments)
current_sample_moment = next(sample_moments_iterator)
machines_df = None

machine_events_csv_colnames = ['time', 'machine_id', 'event_type',
                               'platform_id', 'cpu', 'mem']

for fn in sorted(listdir('machine_events')):
    fp = path.join('machine_events', fn)
    machine_events_df = read_csv(
        fp, header=None, index_col=False,
        compression='gzip', names=machine_events_csv_colnames)
    for index, event in machine_events_df.iterrows():

        if (current_sample_moment is not None and
                event['time'] > current_sample_moment):
            tmp_machines_df = DataFrame(machines_dict.values())
            samples_dicts[current_sample_moment].update(
                {'cpu_available': sum(tmp_machines_df['cpu']),
                 'mem_available': sum(tmp_machines_df['mem'])})
            try:
                current_sample_moment = next(sample_moments_iterator)
            except StopIteration:
                current_sample_moment = None

        if machines_df is None and event['time'] > snapshot_moment:
            machines_df = DataFrame(machines_dict.values())

        if event['event_type'] in [0, 2]:
            machines_dict[event['machine_id']] = {
                'machine_id': event['machine_id'],
                'cpu': event['cpu'], 'mem': event['mem']}
        elif event['event_type'] in [1]:
            del machines_dict[event['machine_id']]

    if machines_df is not None and current_sample_moment is None:
        break

machines_df = DataFrame(machines_dict.values())
samples_df = DataFrame(samples_dicts.values())


task_usage_csv_colnames = [
    'starttime', 'endtime', 'job_id', 'task_idx', 'machine_id',
    'cpu_usage', 'mem_usage', 'assigned_mem', 'unmapped_cache_usage',
    'page_cache_usage', 'max_mem_usage', 'disk_io_time', 'max_disk_space',
    'max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai', 'sampling_rate', 'agg_type']

for moment in samples_dicts:
    samples_dicts[moment].update({'cpu_usage': 0.0, 'mem_usage': 0.0})

for task in tasks_dict:
    tasks_dict[task].update({'cpu_usage': 0.0, 'mem_usage': 0.0,
                             'in_events': True, 'in_usage': False})

for fn in sorted(listdir('task_usage')):

    fp = path.join('task_usage', fn)
    task_usage_df = read_csv(
        fp, header=None, index_col=False,
        compression='gzip', names=task_usage_csv_colnames)

    laststart = max(task_usage_df['starttime'])
    if laststart > max(sample_moments) and laststart > snapshot_moment:
        break

    for moment in samples_dicts:
        task_usage_moment_df = task_usage_df[(task_usage_df['starttime'] <= moment) &
                                             (moment <= task_usage_df['endtime'])]
        samples_dicts[moment]['cpu_usage'] += sum(task_usage_moment_df['cpu_usage'])
        samples_dicts[moment]['mem_usage'] += sum(task_usage_moment_df['mem_usage'])

    task_usage_snapshot_df = task_usage_df[(task_usage_df['starttime'] <= snapshot_moment) &
                                           (snapshot_moment <= task_usage_df['endtime'])]
    for index, usage in task_usage_snapshot_df.iterrows():
        task_id = (usage['job_id'], usage['task_idx'])
        if task_id in tasks_dict:
            tasks_dict[task_id].update({'cpu_usage': usage['cpu_usage'], 'mem_usage': usage['mem_usage'],
                                        'in_events': True, 'in_usage': True})
        else:
            tasks_dict[task_id] = {'cpu_requested': 0.0, 'mem_requested': 0.0, 
                                   'cpu_usage': usage['cpu_usage'], 'mem_usage': usage['mem_usage'],
                                   'in_events': False, 'in_usage': True}

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(samples_df['time'], samples_df['cpu_requested'], label='cpu requested')
ax.plot(samples_df['time'], samples_df['cpu_available'], label='cpu available')
ax.plot(samples_df['time'], samples_df['cpu_usage'], label='cpu usage')
plt.xlim(min(samples_df['time']), max(samples_df['time']))
plt.legend()
plt.show()

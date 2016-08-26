# -*- coding: utf-8 -*-

#In this script we find the resources of the machines whose id is in the violated list and store all their info in a doctionary(key = machine_id, value = corresponding machine_events row)

# 4 Feb 2016
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


chdir('/home/askrey/Dropbox/Project_step_by_step/2_find_violations/csvs')

task_events_csv_colnames = ['time', 'missing', 'job_id', 'task_idx', 'machine_id', 'event_type', 'user', 'sched_cls', 
                            'priority', 'cpu_requested', 'mem_requested', 'disk', 'restriction']
         
evictedtasks = OrderedDict([])
violatedtasks = OrderedDict([])
  
for key, val in csv.reader(open("not_finished_evictedtasks.csv")):
    evictedtasks[key] = val

#load machine events table:
chdir('/home/askrey/Final_project') 
reader = csv.reader(codecs.open('part-00000-of-00001.csv','rU','utf-8'))

machines_set = set()

fp = path.join('task_events', sorted(listdir('task_events'))[0])
task_events_df = read_csv(fp, header = None, index_col = False, names = task_events_csv_colnames, compression = 'gzip')

for index, event in task_events_df.iterrows():
  if str(event['machine_id']) != 'nan':
    machines_set.add(int(event['machine_id']))

machines_dictionary = OrderedDict([])

# find the info about the machines whose machine id is in the violated list(machines_set).
for row in reader:
  if int(row[1]) in machines_set:
       machines_dictionary[row[1]] = row[0:]

writer = csv.writer(open('/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs/find_machine_resources.csv', 'wb'))
for key, value in machines_dictionary.items():
   writer.writerow([key, value])
   print value[0]








# -*- coding: utf-8 -*-
from os import chdir
from pandas import read_csv
import matplotlib.pyplot as plt

chdir('/home/askrey/Final_project')
samples_df = read_csv('samples_df4.csv', index_col=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(samples_df['time'], samples_df['mem_requested'], label='mem requested')
ax.plot(samples_df['time'], samples_df['mem_available'], label='mem available')
ax.plot(samples_df['time'], samples_df['assigned_mem'], label='mem assigned_mem')
ax.plot(samples_df['time'], samples_df['mem_usage'], label='mem mem_usage')
ax.plot(samples_df['time'], samples_df['mem_usage'], label='mem mem_usage')
plt.xlim(min(samples_df['time']), max(samples_df['time']))
plt.legend()
fig.savefig('assigned_mem.png')
plt.show()

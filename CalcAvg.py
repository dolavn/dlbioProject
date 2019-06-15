import os
from datetime import datetime

PATH = './experiment_3/'


aucs = []
times = []
cpus = []
for file in os.listdir(PATH):
    if file[0] == 'm':
        with open(PATH+file) as curr:
            name = file[file.find('rbp'):file.find('_AUC')]
            aucs.append((name, float(curr.readline())))
    if file.startswith('STDIN.e'):
        with open(PATH+file) as curr:
            line = curr.readlines()[-2:][0]
            time = line[line.find('system')+len('system '):line.find('elapsed')]
            cpu = line[line.find('elapsed ') + len('elapsed '):line.find('CPU')]
            mem = line[line.find('CPU ') + len('CPU '):line.find('k')]
            d = datetime.strptime(time[:-3], '%M:%S')
            times.append(d.minute*60 + d.second)
            cpus.append(cpu)
            print('time:{}'.format(time))
            print('CPU usage:{}'.format(cpu))
            print('mem usage:{}'.format(mem))
for name, result in aucs:
    print('{} - {}'.format(name, result))
print('average - {}'.format(sum([auc[1] for auc in aucs])/len(aucs)))
print('average cpu usage- {}%'.format(sum([int(cpu[:-1]) for cpu in cpus])/len(cpus)))
print('average time- {} seconds ({} minutes)'.format(sum(times)/len(times), (sum(times)/len(times))/60))

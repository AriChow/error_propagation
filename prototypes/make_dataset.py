import os
import sys
import shutil


home = os.path.expanduser('~')
dataset = 'brain1'
data_name = 'brain'
place = 'Documents/research'
data_home = home + '/' + place + '/EP_project/data/'
results_home = home + '/' + place + '/EP_project/results/'
f = open(data_home + 'datasets/' + dataset + '/class-file.txt', 'r')
lines = f.read().split('\n')
images = []
for l in lines:
	l1 = l.split()
	if len(l1) != 3:
		continue
	grade = l1[1][:-1]
	s = l1[2].find('-')
	start = int(l1[2][1:s])
	end = int(l1[2][s+2:])
	images = []
	for j in range(start, end + 1):
		image = 'f' + str(j) + '.jpg'
		if image not in images:
			images.append(image)
			src = data_home + 'datasets/' + dataset + '/' + image
			des = data_home + 'datasets/' + data_name + '/' + grade.lower() + '/' + image
			shutil.copyfile(src, des)


f.close()
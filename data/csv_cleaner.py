'''
This file will take the contents of the csvsetter.ipynb file saved on my home PC
and turn it into a python function that can be used within _One_Step e2e prediction function
'''

def set_csv(
	file_name_in	:	str	=	''
	,file_name_out	:	str	=	''
):
	'''This function takes the contents of csvSetter and makes it usable easily within other python functions'''

	data = open(file_name_in,'r')
	of = open(file_name_out,'w')
	for i in range(6):
		line = data.readline()

	of.write("high,low,close,volume,time\n")

	lines = data.readlines()
	print('header written, lines read..')
	catLine=''
	for i in range(0,len(lines)-7):
		line = lines[i]
		if(line.find('S')==-1):
			catLine = line[line.find('(')+1:line.find(')')]
			catLine = catLine.replace(',','')
			catLine = catLine.replace('|',',')
			of.write(catLine)
			of.write('\n')
			#print(catLine)
			#print(line)
			#print()
	print('file written..')
	of.close()
	data.close()
	print('files closed.')
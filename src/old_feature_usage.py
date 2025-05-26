'''
	Logan Kelsch
	This file is going to be used for the new data that will be collected.

	In V6.0, we will be SCRAPING only 6 features, instead of many.
	These 6 features will be:
	-   High, Low, Open, Close (model training data pre-processing removable)
	-   ToD (minute)
	-   Volume (of minute)
	- JUST ADDED 
	-   DoW

	With these features we will use this python file to create hopefully a few hundred features.
	These features will consist of a large sets of slightly altered dimensions of the provided data.
	
	ENGINEERED FEATURE CATEGORIES:
						note-- price movement is naturally over time, consider terminology
							   with this in mind. (ex: velocity is displacement over time)
						
						notation: 
								(description or term hints)
								[value ranges low to high] %favorable_increments
								{estimated total features}

		!! NOTE NOTE HUGE FINALIZATION NOTE: THERE SHOULD BE NO SUBTRACTION, ONLY DIVISION
		TO AVOID ACTUAL VALUE USAGE, IT SHOULD BE ACTUAL PERCENT USAGE SO ALL FUNCTIONS
		SHOULD LOOK SOMETHING LIKE feature = 100*(val1/val2-1) [this is representation of percent difference] 

		##NOTE TEMPORARY -- never mind
		
	# NOTE indicates complete

	-   Derivative of (looping index only) feature set A set
	-   #Price difference (velocity)             [1 - 60]                {60} 60
	-   #Rate of price difference (acceleration) [1 - 60]                {60} 120
	-   Stochastic K, D, K-D (K is fast)        k=[5-60]%5   d=[k-60]%5 {78}    468
	-   -   D is just moving average, come back to this, K-D can be same function
	-   #-   K
	-   RSI                                     [1 - 60]                {60} 180
	-   #close - Moving average                  [1-60]%5 , [60-240]%20  {68}    536
	-   #Mov-avg-diff                    [5, 10, 15, 30, 60, 120]        {30} 210
	-   #close - lowest low                [5, 15, 30, 60, 120]          {5}
	-   #close - highest high              [5, 15, 30, 60, 120]          {5}  220
	-   #hihi - lolo                      [custom 2 pairs above]         {20} 240 
	-   hilo stoch
	-   #bar height                                                      {1}
	-   #wick height                                                     {1}
	-   #uwick - lwick                                                   {1}     539
	-   high - (close, open, low)(, high holds) [1-5]                   {15} 255
	-   (high, close, open) - low(, low holds)  [1-5]                   {15} 270
	-   #total volume                            [1-60]                  {60} 330
	-   total vol chunk difference              [1-60]%5                {11}    550
	-   #volume - average volume                 [1-60]                  {60} 390
	-   consider h - l 1min vel with moving averages
												550 features predicted

	remaining old data labels and features:
	HL2,H2L,HLdiff12,HLdiff21,vol,vol10,vol15,FT,vol30,vol60,volD10,volD15,volD30,volD60,
	vpm5,vpm10,vpm15,vpm30,vpm60,ToD,DoW,mo,r1,r2,r3,r5,r10,r15,r30,r60

	ENGINEERED TARGET CATEGORIES:

	-   Price Difference (will start here)
		-   price difference                    [1 - 60]                {60}

	NOTE can create these later
	-   Direction Classifications
	-   Volume Classifications
'''

'''
	NOTE ALL DATASETS HANDLED IN THESE FUNCTIONS WILL HANDLED AS PANDAS DATAFRAMES
		 TO ALLOW FOR USAGE OF NAMES OF FEATURES/TARGET COLUMNS
'''

from _Utility import po #percent of, shorthand function
from multiprocessing import Pool
from _Utility import function_executor
import pandas as pd
from typing import Literal
import numpy as np

times = [5,15,30,60,120,240]
idx	  = ['spx','ndx','NULL','NULL','NULL','ndx']

######### '''NOTE NOTE''''''NOTE NOTE''' #########
###* * MOST IMPORTANT FUNCTION IN THIS FILE * *###
######______________________________________######

#this function will take in a dataset and generate 
#all requested features sets as well as target sets
#the output will be a pandas dataframe, fully concatenated
def augmod_dataset(
	data
	,index_names	:	list	=	['spx','ndx']
	,format_mode	:	Literal['live','backtest'] = 'backtest'
	,clip_stochastic:	bool	=	True
	,target_isolate	:	Literal['r','c','a','sc','kr']	=	None
):

	'''NOTE NOTE broke these processes down into a few different areas of multiprocessing based off of
	   NOTE NOTE linear calculation dependancy of various feature categories 
	'''
	'''
	funcs_1 = [fe_ToD, fe_DoW, fe_vel, fe_acc, fe_stoch_k, fe_height_bar, fe_height_wick, fe_diff_hl_wick, \
		  fe_vol_sz_diff, fe_ma_disp]
	args_1  = [(data,), (data,), (data, i[0]), (data, i[0]), (data, i[0]), (data, i[0]), (data, i[0]), (data, i[0]), (data, i[0]), (data, i[0])]

	starmap_inputs = [(funcs_1[i], args_1[i]) for i in range(len(funcs_1))]

	with Pool() as pool:
		print('beginning first round of starmap dataset creation.')
		f_ToD, f_DoW, f_vel, f_acc, f_stchK, f_barH, f_wickH, f_wickD, f_volData, f_maData = pool.starmap(function_executor, starmap_inputs)
		print('ending first round of starmap dataset creation.')

	funcs_2 = [fe_ma_diff, fe_hihi_diff, fe_lolo_diff]
	args_2  = [(f_maData,), (data, i[0]), (data, i[0])]

	starmap_inputs = [(funcs_2[i], args_2[i]) for i in range(len(funcs_2))]

	with Pool() as pool:
		print('beginning second round of starmap dataset creation.')
		f_maDiff, f_hihi, f_lolo = pool.starmap(function_executor, starmap_inputs)
		print('ending second round of starmap dataset creation.')

	funcs_3 = [fe_hilo_diff, fe_hilo_stoch]
	args_3  = [(f_hihi,f_lolo), (data, f_hihi, f_lolo, i[0])]

	starmap_inputs = [(funcs_3[i], args_3[i]) for i in range(len(funcs_3))]

	with Pool() as pool:
		print('beginning third round of starmap dataset creation.')
		f_hilo, f_stochHiLo = pool.starmap(function_executor, starmap_inputs)
		print('ending third round of starmap dataset creation.')

	funcs_4 = [te_vel_reg, te_vel_class, te_area_class]
	args_4  = [(data, i[0]), (data, i[0]), (data, i[0])]

	starmap_inputs = [(funcs_4[i], args_4[i]) for i in range(len(funcs_4))]

	with Pool() as pool:
		print('beginning fourth round of starmap dataset creation.')
		target_r, target_c, target_a = pool.starmap(function_executor, starmap_inputs)
		print('ending fourth round of starmap dataset creation.')
	'''

	i = [0,5]#these are the initial column indices for each given index implemented
	#so far only ES is implemented, with another on its way
	
	
	#FEATURE ENGINEERING		--------------

	#first collect basic data within dataset
	f_ToD = fe_ToD(data)#single
	f_DoW = fe_DoW(data)#single

	#collect spx specific data
	f_vel = fe_vel(data, i[0])#set
	f_acc = fe_acc(data, i[0])#set
	f_stchK = fe_stoch_k(data, i[0], clip_stochastic)#set
	f_barH = fe_height_bar(data, i[0])#single
	f_wickH = fe_height_wick(data, i[0])#single
	f_wickD = fe_diff_hl_wick(data, i[0])#single
	f_volData = fe_vol_sz_diff(data, i[0])#set
	f_maData = fe_ma_disp(data, i[0])#set
	f_maDiff = fe_ma_diff(f_maData, i[0])#set
	f_hihi = fe_hihi_diff(data, i[0])#set
	f_lolo = fe_lolo_diff(data, i[0])#set
	f_hilo = fe_hilo_diff(f_hihi,f_lolo, i[0])#set
	f_stochHiLo = fe_hilo_stoch(data, f_hihi, f_lolo, i[0], clip_stochastic)#set

	if(len(index_names)>1):
		#collect ndx specific data
		f_vel2 = fe_vel(data, i[1])#set
		f_acc2 = fe_acc(data, i[1])#set
		f_stchK2 = fe_stoch_k(data, i[1], clip_stochastic)#set
		f_barH2 = fe_height_bar(data, i[1])#single
		f_wickH2 = fe_height_wick(data, i[1])#single
		f_wickD2 = fe_diff_hl_wick(data, i[1])#single
		f_volData2 = fe_vol_sz_diff(data, i[1])#set
		f_maData2 = fe_ma_disp(data, i[1])#set
		f_maDiff2 = fe_ma_diff(f_maData2, i[1])#set
		f_hihi2 = fe_hihi_diff(data, i[1])#set
		f_lolo2 = fe_lolo_diff(data, i[1])#set
		f_hilo2 = fe_hilo_diff(f_hihi2,f_lolo2, i[1])#set
		f_stochHiLo2 = fe_hilo_stoch(data, f_hihi2, f_lolo2, i[1], clip_stochastic)#set

	#collect index comparison data


	if(format_mode == 'backtest'):
	#TARGET ENGINEERING
		if(target_isolate == None):
			target_r = te_vel_reg(data, i[0])
			target_c = te_vel_class(data, i[0])
			target_a = te_area_class(data, i[0])
			target_sc = te_stoch_class(data, i[0], 2)
		#target_sr = te_stoch_reg(data, i[0])
		else:
			#under the case there is target isolation, that means all other targets will not be calculated other than one.
			match(target_isolate):
				case 'r':
					target_r = te_vel_reg(data, i[0])
				case 'c':
					target_c = te_vel_class(data, i[0])
				case 'a':
					target_a = te_area_class(data, i[0])
				case 'sc':
					target_sc2 = te_stoch_class(data, i[0], 2)
					target_sc3 = te_stoch_class(data, i[0], 3)
					target_sc4 = te_stoch_class(data, i[0], 4)
				case 'kr':
					target_kr = te_kelsch_reg(data, i[0])
				case None:
					target_r = te_vel_reg(data, i[0])
					target_c = te_vel_class(data, i[0])
					target_a = te_area_class(data, i[0])
					target_sc = te_stoch_class(data, i[0], 2)

	#list of dataframes
	if(len(index_names)==1):
		df_list = [data, f_ToD, f_DoW, f_vel, f_acc, \
						f_stchK, f_barH, f_wickH, f_wickD,\
							f_volData, f_maData, f_maDiff, f_hihi, f_lolo,\
								f_hilo, f_stochHiLo]
		if(format_mode == 'backtest'):
			match(target_isolate):
				case 'r':
					df_list.append(target_r)
				case 'c':
					df_list.append(target_c)
				case 'a':
					df_list.append(target_a)
				case 'sc':
					df_list.append(target_sc2)
					df_list.append(target_sc3)
					df_list.append(target_sc4)
				case 'kr':
					df_list.append(target_kr)
				case None:
					df_list.append(target_r)
					df_list.append(target_c)
					df_list.append(target_a)
					df_list.append(target_sc)
					
	if(len(index_names)==2):
		df_list = [data, f_ToD, f_DoW, f_vel, f_acc, \
						f_stchK, f_barH, f_wickH, f_wickD,\
							f_volData, f_maData, f_maDiff, f_hihi, f_lolo,\
								f_hilo, f_stochHiLo, \
						f_vel2, f_acc2, \
							f_stchK2, f_barH2, f_wickH2, f_wickD2,\
								f_volData2, f_maData2, f_maDiff2, f_hihi2, f_lolo2,\
									f_hilo2, f_stochHiLo2]
		if(format_mode == 'backtest'):
			match(target_isolate):
				case 'r':
					df_list.append(target_r)
				case 'c':
					df_list.append(target_c)
				case 'a':
					df_list.append(target_a)
				case 'sc':
					df_list.append(target_sc2)
					df_list.append(target_sc3)
					df_list.append(target_sc4)
				case 'kr':
					df_list.append(target_kr)
				case None:
					df_list.append(target_r)
					df_list.append(target_c)
					df_list.append(target_a)
					df_list.append(target_sc)

	#cut off error head and error tail of dataframes

	#cutting off the tail is only for target inclusion
	if(format_mode == 'backtest'):
		df_trunk_tail = [df.iloc[:-60] for df in df_list]
		#otherwise, we do not need to trim as targets are not included

	#this head trim is necessary, removing oldest samples for smooth+true feature inclusion
	if(format_mode == 'backtest'):
		df_trunk_head = [df.iloc[240:] for df in df_trunk_tail]
	else:
		df_trunk_head = [df.iloc[240:] for df in df_list]

	#concat all dataframes into one parallel set
	full_augmod = pd.concat(df_trunk_head, axis=1)

	return full_augmod

'''-------------------------------------------------------------------------------
	NOTE FEATURE SPECIFIC FUNCTIONS
	NOTE fe_ denotes 'feature engineering'
'''#------------------------------------------------------------------------------

#returns lowest close of different ranges
#and the close difference to each
#this function requires cutting first 240 samples
def fe_lolo_diff(X, index):
	#orig feature #3
	# # # deals with all close of minute values
	close = X.iloc[:, 2+index].values
	new_data = []

	l = len(X)
	for sample in range(l):
		row = []

		#getting lowest lows
		for dist in times:
			if(sample-dist < 0):
				row.append(0)
			else:
				lolo = np.min(close[sample-dist:sample])
				row.append(lolo)

		#getting lolo close displacements
		for i in range(len(times)):
			disp = close[sample] - row[i]
			row.append(disp)

		new_data.append(row)

	cols = [f'lolo{i}_{idx[index]}' for i in times]+\
		[f'disp_lolo{i}_{idx[index]}' for i in times]

	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set

#returns highest close of different ranges
#and the close difference to each
#this function requires cutting first 240 samples
def fe_hihi_diff(X, index):
	#orig feature #3
	# # # deals with all close of minute values
	close = X.iloc[:, 2+index].values
	new_data = []

	l = len(X)
	for sample in range(l):
		row = []

		#getting lowest lows
		for dist in times:
			if(sample-dist < 0):
				row.append(0)
			else:
				hihi = np.max(close[sample-dist:sample])
				row.append(hihi)

		#getting lolo close displacements
		for i in range(len(times)):
			disp = close[sample] - row[i]
			row.append(disp)

		new_data.append(row)

	cols = [f'hihi{i}_{idx[index]}' for i in times]+\
		[f'disp_hihi{i}_{idx[index]}' for i in times]

	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set

#returns vol*time area and avg vol difference
#thsi function requires cutting first 60 samples 
def fe_vol_sz_diff(X, index):
	#orig feature #4
	# # # deals with volume of each minute
	volume = X.iloc[:, 3+index].values
	new_data = []

	l = len(X)
	for sample in range(l):
		row = []
		#creating 59 areas of total volume from 2 -> 60 minutes
		for i in range(1,60):
			t_vol = volume[sample]
			for j in range(1,i+1):
				t_vol+=volume[(sample - j) %l]
			row.append(t_vol)
		
		#creating 59 diffs for vol - avgvol from 2 -> 60 minutes
		for i in range(1,60):
			avg_vol = row[i-1]/(i+1)
			row.append(round(volume[sample] - avg_vol, 2))

		#this is all data for each given sample
		new_data.append(row)

	#custom feature name
	cols = [f'vol_m{i}_{idx[index]}' for i in range(2,61)]+\
		[f'vol_avgDiff{i}_{idx[index]}' for i in range(2,61)]

	#CONTINUE HERE THERE ARE ONLY 59 FEATURES
	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set

#returns moving averages and close-ma difference
#this function requires cutting of first 240 samples
def fe_ma_disp(X, index):
	#orig feature #3
	# # # deals with all close of minute values
	close = X.iloc[:, 2+index].values
	new_data = []

	l = len(X)
	for sample in range(l):
		row = []
		'''first create price MAs'''
		# 2-59 total mins, 58 cases
		for i in range(1,59):
			avg_price = close[sample]
			for j in range(1,i+1):
				avg_price+= close[(sample - j) %l]
			#convert price*time to an average
			row.append(round(avg_price / (i+1),2))
		# 60-240 %20 total mins, 10 cases 
		for i in range(59,240,20):
			avg_price = close[sample]
			for j in range(1,i+1):
				avg_price+= close[(sample - j) %l]
			#convert price*time to an average
			row.append(round(avg_price / (i+1),2))
		'''second create MA-close diffs'''
		for ma in range(68):
			ma_disp = round(close[sample] - row[ma],2)
			row.append(ma_disp)
		
		new_data.append(row)

	cols = [f'ma{i}_{idx[index]}' for i in range(2,60)]+\
		   [f'ma{i}_{idx[index]}' for i in range(60,241,20)]+\
		   [f'disp_ma{i}_{idx[index]}' for i in range(2,60)]+\
		   [f'disp_ma{i}_{idx[index]}' for i in range(60,241,20)]
	
	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set


#return Time of Day in minutes
#this function requires no cutting
def fe_ToD(X):
	#orig feature #5
	# # # deals with time since 1/1/1970@12:00am in seconds
	full_time = X.iloc[:, 4].values
	new_data = []

	l = len(X)
	for sample in range(l):
		'''
			take full time
			minus time zone adjustment
			modulate total seconds around days
			convert into minutes
		'''
		tod = (((full_time[sample] - 18000) % 86400) / 60)
		new_data.append(tod)
	
	feature = pd.DataFrame(new_data, columns=['ToD'])

	return feature

#returns Day of Week (0-7 Sun-Sat, 1-5 Mon-Fri)
#this function requires no cutting
def fe_DoW(X):
	#orig feature #5
	# # # deals with time since 1/1/1970@12:00am in seconds
	full_time = X.iloc[:, 4].values
	new_data = []

	l = len(X)
	for sample in range(l):
		'''
			take full time
			minus time zone and week adjustments
			convert into days
			modulate total days around weeks
			floor division for integer output
		'''
		dow = ((((full_time[sample] - 277200) / 86400) % 7) // 1)
		new_data.append(dow)

	feature = pd.DataFrame(new_data, columns=['DoW'])

	return feature

#difference of upper/lower wick size
#this function requires cutting first 1 sample
def fe_diff_hl_wick(X, index):
	new_data = []
	#get high, low, close values
	low = X.iloc[:, 1+index].values
	high = X.iloc[:, 0+index].values
	close = X.iloc[:, 2+index].values

	l = len(X)
	for sample in range(l):
		#height of upper wick
		u_wick = high[sample] - close[sample]
		#height of lower wick
		l_wick = close[(sample - 1) %l] - low[sample]

		new_data.append(u_wick - l_wick)

	feature = pd.DataFrame(new_data, columns=[f'diff_wick_{idx[index]}'])

	return feature


#high of candle (bar)
#this function requires cutting first 1 sample
def fe_height_bar(X, index):
	#orig feature #3
	# # # deals with all close of minute values
	close = X.iloc[:, 2+index].values
	new_data = []

	l = len(X)
	for sample in range(l):
		#abs difference from close and open
		h = abs(close[sample] - close[(sample - 1) %l])
		new_data.append(h)

	feature = pd.DataFrame(new_data, columns=[f'barH_{idx[index]}'])

	return feature

#height of wicks and bar
#this function requires cutting first 1 sample
def fe_height_wick(X, index):
	#orig feature #1, #2
	# # # deals with open and close
	high = X.iloc[:, 0+index].values
	low = X.iloc[:, 1+index].values
	new_data = []

	l = len(X)
	for sample in range(l):
		#high minus low of each candle/wick
		h = high[sample] - low[sample]
		new_data.append(h)

	feature = pd.DataFrame(new_data, columns=[f'wickH_{idx[index]}'])

	return feature

#velocities
#this function requires cutting first 60 samples (df.iloc[60:])
def fe_vel(X, index):
	#orig feature #3
	# # # deals with all close of minute values
	close = X.iloc[:, 2+index].values
	new_data = []

	l = len(X)
	for sample in range(l):
		row = []
		for displace in range(1,61):
			row.append(close[sample %l] - close[(sample-displace) %l])
		new_data.append(row)
	
	feature_set = pd.DataFrame(new_data, columns=[f'vel{i}_{idx[index]}' for i in range(1,61)])

	return feature_set


#accelerations
#this function requires cutting first 61 samples (df.iloc[61:])
def fe_acc(X, index):
	#orig feature #3
	# # # deals with all close of minute values
	close = X.iloc[:, 2+index].values
	new_data = []

	l = len(X)
	for i in range(l):
		row = []
		for displace in range(1,61):
			# Calculate i + feature_num and handle out-of-bounds by wrapping around using modulo
			j = (i - displace)
			#actual value in csv is 100th of percent move
			vel1 = close[(i-1)  %l] - close[(j-1)   %l]
			vel2 = close[i      %l] - close[j       %l]
			row.append(vel2-vel1)
		new_data.append(row)
	# Convert to a new DataFrame
	feature_set = pd.DataFrame(new_data, columns=[f'acc{i+1}_{idx[index]}' for i in range(60)])

	#print(feature_set)
	return feature_set

#Stochastic K ONLY
#this function requires cutting first 120 samples (df.iloc[120:])
#used zero-out method for Null values instead of looping.
def fe_stoch_k(X, index, clip):

	new_data = []
	#get high, low, close values
	low = X.iloc[:, 1+index].values
	high = X.iloc[:, 0+index].values
	close = X.iloc[:, 2+index].values
	i = 0

	l = len(X)
	for sample in range(l):
		row = []
		for i in range(5,125,5):
			#zero-out to avoid segmentation bound error
			if(sample-i<0):
				row.append(0)
			else:
				lowest_k = np.min(low[sample-i:sample])
				c1 = close[sample] - lowest_k
				c2 = np.max(high[sample-i:sample]) - lowest_k
				k = 0
				if(c2!=0):
					k = c1/c2*100
				if(k>100 and clip):
					k = 100
				row.append(round(k,2))
		new_data.append(row)
	
	features_set = pd.DataFrame(new_data, columns=[f'stchK{i}_{idx[index]}' for i in range(5, 125, 5)])
	
	return features_set
			
#this function is directly interacting with collected stochK data
#this will be a lot of features if working with k values
#up to two hours old, 
#NOTE will come back to this later if wanted
def fe_stoch_d(f_stochK):

	new_data = []

	return new_data

#function return the set of ma differences from a set
#this function requires cutting first 120 samples
def fe_ma_diff(maData, index):

	#all used moving averages
	ma5 = maData.iloc[:, 3].values
	ma15 = maData.iloc[:, 13].values
	ma30 = maData.iloc[:, 28].values
	ma60 = maData.iloc[:, 58].values
	ma120 = maData.iloc[:, 61].values
	ma240 = maData.iloc[:, 67].values

	#array of arrays for ease of access
	mas = [ma5, ma15, ma30, ma60, ma120, ma240]

	new_data = []

	l = len(maData)
	for sample in range(l):
		row = []
		#nested to access two ma values at once for comparison
		for i in range(len(mas)):
			for j in range(i+1,len(mas)):
				ma_1 = mas[i][sample]
				ma_2 = mas[j][sample]
				row.append(round(ma_1 - ma_2, 2))
		new_data.append(row)
	
	cols = []
	lengths = [5,15,30,60,120,240]

	#prepping the feature names according to ma's used
	for i in range(len(lengths)):
		for j in range(i+1,len(lengths)):
			cols.append(f'diff_ma_{lengths[i]}_{lengths[j]}_{idx[index]}')

	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set

#function return the differences between hihi and lolo of time sets
#this function requires cutting first 240 samples
def fe_hilo_diff(hihi_data, lolo_data, index):

	lengths = [5,15,30,60,120,240]
	
	hihi = hihi_data.values
	lolo = lolo_data.values

	new_data = []

	l = len(hihi)
	for sample in range(l):
		row = []
		#nested to access two ma values at once for comparison
		for i in range(len(lengths)):
			for j in range(len(lengths)):
				hi = hihi[sample, i]
				lo = lolo[sample, j]
				row.append(round(hi - lo, 2))
		new_data.append(row)
	
	cols = []

	#prepping the feature names according to ma's used
	for i in range(len(lengths)):
		for j in range(len(lengths)):
			cols.append(f'diff_hilo_{lengths[i]}_{lengths[j]}_{idx[index]}')

	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set

#function returns location percent (like stochastic) between hihi lolo for each
#this function requires cutting first -- samples
def fe_hilo_stoch(X, hihi_data, lolo_data, index, clip):
	#orig feature #3
	# # # deals with all close of minute values
	close = X.iloc[:, 2+index].values

	#   j       -nested in-   i       
	#low ranges -nested in- high ranges
	hihi = hihi_data.values
	lolo = lolo_data.values

	new_data = []

	l = len(X)
	#for each sample
	for sample in range(l):
		row = []
		for i in range(len(times)):
			for j in range(len(times)):
				lowest_k = np.min(lolo[sample, j])
				c1 = close[sample] - lowest_k
				c2 = np.max(hihi[sample, i]) - lowest_k
				k = 0
				if(c2!=0):
					k = c1/c2*100
				if(k>100 and clip):
					k = 100
				row.append(round(k,2))
		new_data.append(row)
	cols = []

	#prepping the feature names according to ma's used
	for i in range(len(times)):
		for j in range(len(times)):
			cols.append(f'hilo_stoch_{times[i]}_{times[j]}_{idx[index]}')

	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set

#this function will be an encapsulation of bollinger bands
def fe_bollinger(
	X,
	index
):
	'''
	TOS code version of this is the following
 	sdev = stdev(close, length)
  	midline = ma(avg type, close, length) length is the same btw
    	bollinger value would be :::
     	val_desired = (close-midline)/sdev
      	#this value would be the standard deviations off of the mean in same time range
 	'''

	lengths = [5, 15, 30, 60, 120, 240]
	
	#deals with all close of minute values
	close = X.iloc[:, 2+index].values

	l = len(X)

	new_data = np.zeros((l, len(lengths)), dtype=np.float32)

	for sample in range(l):
		row = np.zeros(len(lengths), dtype=np.float32)

		for t, time in enumerate(lengths):

			if(sample-time < 0):
				row[t] = 0
			else:
				sdev = np.std(close[sample-time:sample])
				ml = np.mean(close[sample-time:sample])
				row[t] = (close-ml)/sdev

		new_data[sample] = row

	cols = []
	for i in lengths:
		cols.append(f'bbands_{i}_{idx[index]}')

	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set

def fe_true_range(
	X,
	index
):
	'''
		Simple function to collect an array of true range values.
      	the greatest of the following
     	diff between h0 l0
      	diff between h0 c1
	diff between c1 l0
 	'''

	low = X.iloc[:, 1+index].values
	high = X.iloc[:, 0+index].values
	close = X.iloc[:, 2+index].values

	l = len(X)

	new_data = np.zeros((l), dtype=np.float32)

	for i in range(l):

		#value 1 must be trimmed, but should be fully functional as is here
		tr = max(
			abs(high[i]-low[i]),
			abs(high[i]-close[(i-1)%l]),
			abs(low[i]-close[(i-1)%l])
		)

		new_data[i] = tr

	cols = ['true_range']

	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set


def fe_atr(
	X,
	index
):
	true_range = fe_true_range(X,index).values

	#TOS script is the following
	'''
		moving average of average type of the truerange(h,l,c) for given length
  		truerange happens to be
 	'''

	lengths = [5, 15, 30, 60, 120, 240]
	
	l = len(X)

	new_data = np.zeros((l, len(lengths)), dtype=np.float32)

	for sample in range(l):
		row = np.zeros(len(lengths), dtype=np.float32)

		for t, time in enumerate(lengths):

			if(sample-time < 0):
				row[t] = 0
			else:
				atr = np.mean(true_range[sample-time:sample])
				row[t] = atr

		new_data[sample] = row

	cols = []
	for i in lengths:
		cols.append(f'atr_{i}_{idx[index]}')

	feature_set = pd.DataFrame(new_data, columns=cols)

	return feature_set
	
def fe_norm_range(
	X,
 index
):

	'''retutns the same dimensions as ATR generated from below lengths array
		atr is called in here 1 to 1
		'''

	lengths = [5, 15, 30, 60, 120, 240]

	high = X.iloc[:,0+index].values
	low = X.iloc[:,1+index].values
	
	atr = fe_atr(X,index).values
	
	l = len(X)
	r = len(atr[0])
	
	new_data = np.zeros((l, r), dtype=np.float32)
	
	for sample in range(l):
		
		row = np.zeros(r, dtype=np.float32)
		
		for arr in range(r):
			
			#append corresponding atr value from array
			row[arr] = (high[sample]-low[sample])/atr[sample,arr]
			
		new_data[sample] = row
		
	cols = []
	for i in lengths:
		cols.append(f'normr_{i}_{idx[index]}')

	feature_set = pd.DataFrame(new_data, columns=cols)
	
	return feature_set
		
def fe_hawkes_process(
	X,
	index
):
	''' non rolling version, raw values'''
	
	norm_ranges = fe_norm_range(X, index)
	
	l = len(X)
	r = len(norm_ranges[0]) #should be lengths array length
	
	lengths = [5, 15, 30, 60, 120, 240]
	
	raw_kappa = [0.64, 0.16, 0.08, 0.04, 0.01]
	kappa = [np.exp(-k) for k in raw_kappa]
	
	kl = len(kappa)
	
	new_data = np.zeros((l,r*len(kappa)), dtype=np.float32)
	
	for sample in range(l):
		
		row = np.zeros(r*len(kappa), dtype=np.float32)
		
		for nr in range(r):
			for k, kap in enumerate(kappa):
				
				truidx = (nr*kl+k)
			
				if((sample==0) | (sample==240)):
					
					hp = norm_ranges[sample,nr]
					
				else:
					
					hp = new_data[sample-1,truidx] * kap + norm_ranges[sample,nr]
				
				row[truidx] = hp
			
		new_data[sample] = row
	
	cols = []
	
	for l in lengths:
		for k in raw_kappa:
			cols.append(f'hawkes_{l}_{k}_{idx[index]}')

	feature_set = pd.DataFrame(new_data, columns=cols)
	
	return feature_set

def fe_hawkes_stoch(
	X,
	index
):

	'''
	need to check out neurotrader888 hawkes video again to see what data is actually going into 
 	this and then can test on TOS what some reasonable values are to use for cappa parameters.
 	'''

	hawkes = fe_hawkes_process(X, index)

	lengths = [5, 15, 30, 60, 120, 240]
	raw_kappa = [0.64, 0.16, 0.08, 0.04, 0.01]
	
	l = len(X)

	#lengths length and kappa length denotations
	ll = len(lengths)
	kl = len(raw_kappa)

	#row length denotations
	rl = ll*kl


	hawk_low_hold = np.zeros((len(hawkes)[0]), dtype=np.float32)
	hawk_high_hold= np.zeros((len(hawkes)[0]), dtype=np.float32)

	new_data = np.zeros((l, len(lengths)), dtype=np.float32)

	for sample in range(l):
		
		row = np.zeros(rl, dtype=np.float32)

		for h in range(rl):

			#floor division works since kappa is nested loop within lengths, we iterate with lengths
			if(sample<lengths[h//kl]):
				#there is not enough data to collect for this array, will be cut out anyways
				pass#is a zero already
			else:
				#safe sample to work on
				#grab the lowest value of the last lengths[i] hawkes value
				hawk_low_hold = np.min(hawkes[(sample-lengths[h//kl]):hawkes[sample],h])
				hawk_high_hold= np.max(hawkes[(sample-lengths[h//kl]):hawkes[sample],h])

				dsp = hawkes[sample,h] - hawk_low_hold
				rng = hawk_high_hold - hawk_low_hold

				if(rng != 0):
					hs = dsp/rng * 100

				else:
					hs = 0

				row[h] = hs

		new_data[sample] = row

	cols = []
	
	for l in lengths:
		for k in raw_kappa:
			cols.append(f'hawkes_stoch_{l}_{k}_{idx[index]}')

	feature_set = pd.DataFrame(new_data, columns=cols)
	
	return feature_set

'''-------------------------------------------------------------------------------
	NOTE TARGET SPECIFIC FUNCTIONS
	NOTE te_ denotes 'target engineering'
'''#------------------------------------------------------------------------------

#simple price difference for 1-60 minutes 
#this function requires cutting first 60 samples (df.iloc[60:])
def te_vel_reg(X, index):
	#orig feature #3
	# # # deals with all close of minute values
	close = X.iloc[:, 2+index].values
	new_data = []

	l = len(X)
	for i in range(l):
		row = []
		for displace in range(1,61):
			row.append(close[(i + displace) %l] - close[i %l])
		new_data.append(row)
	# Convert to a new DataFrame
	feature_set = pd.DataFrame(new_data, columns=[f'tr_{i+1}' for i in range(60)])

	#print(feature_set)
	return feature_set

def te_kelsch_reg(X, index):
	pass

def te_stoch_class(
	X
	,index
	,num_classes:int=2
):
	assert (num_classes > 1 and num_classes < 5), 'stoch target only supports 2-4 classes'
	
	new_data = []
	#get high, low, close values
	low = X.iloc[:, 1+index].values
	high = X.iloc[:, 0+index].values
	close = X.iloc[:, 2+index].values
	i = 0

	l = len(X)
	for sample in range(l):
		row = []
		for past in range(15,125,15):
			for futr in range(0,65,15):
				#zero-out to avoid segmentation bound error
				if(sample-past<0 or sample+futr>=l):
					row.append(-1)
				else:
					#lowest of past range
					bottom = np.min(low[sample-past:sample])
					#difference between lowest and highest
					diff = np.max(high[sample-past:sample]) - bottom
					#location of future close in regards to bottom of range
					t_val = close[sample+futr] - bottom
					
					k = 0
					if(diff!=0):
						k = t_val/diff

					#define number of classes based on 
					match(num_classes):
						case 2:
							c = 0 if (k<0.5) else 1
						
						case 3:
							c = 0 if (k<0.25) else \
								1 if (k<=0.75) else 2

						case 4:
							c = 0 if (k<=0) else \
								1 if (k<0.5) else \
								2 if (k<1.0) else 3
						
						case _:
							raise ValueError(f'FATAL, num_classes in te_stoch_class was not defined properly, got {num_classes}.')

					row.append(int(c))
		new_data.append(row)
	
	#name is formatted as "target-stochastic-classification", past-range, future-displacement, trading index
	target_set = pd.DataFrame(new_data, columns=[f'tsc_{num_classes}_{i}_{j}_{idx[index]}' for i in range(15, 125, 15) for j in range(0, 65, 15)])
		
	return target_set

def te_stoch_reg():
	return

#simple classification set for 1-60 minutes
#this function requires cutting first 60 samples
def te_vel_class(X, index):
	#orig feature #3
	# # # deals with all close of minute values
	close = X.iloc[:, 2+index].values
	new_data = []

	l = len(X)
	for i in range(l):
		row = []
		#two class data
		for displace in range(5,61,5):
			movement = close[(i + displace) %l] - close[i %l]
			if(movement < 0):
				row.append(0)
			else:
				row.append(1) 
		#three class data
		for displace in range(5,61,5):
			movement = close[(i + displace) %l] - close[i %l]
			c = np.sign(movement) + 1
			mag = 0
			if(abs(movement) >= 5):
				mag=1
			c+=mag
			row.append(c)
		#four class data
		for displace in range(5,61,5):
			movement = close[(i + displace) %l] - close[i %l]
			if(movement < 0):
				s = 0
			else:
				s = 1
			if(abs(movement) >= 5):
				m = 1
			else:
				m = 0
			c = s+1+(m*np.sign(movement))
			row.append(c)

		new_data.append(row)

	cols = [f'tc_2c_{i}m' for i in range(5, 61, 5)]+\
		[f'tc_3c_5p_{i}m' for i in range(5, 61, 5)]+\
		[f'tc_4c_5p_{i}m' for i in range(5, 61, 5)]
	# Convert to a new DataFrame
	feature_set = pd.DataFrame(new_data, columns=cols)

	#print(feature_set)
	return feature_set

#New target section, price area
#price area: 
# integral/area of price displacement of future m minutes
#This function will require cutting out first 60 samples of the dataset
def te_area_class(X, index):
	#original feature #3 is close of candles
	close = X.iloc[:, 2+index].values
	#list to collect all target data across dataset
	new_data = []

	l = len(X)
	#For each sample of the provided dataset
	for i in range(l):
		#This row is the collection of targets for each sample
		row = []
		#binary class data
		#for each target to be added
		for trgt_disp in range(5, 61, 5):
			area = 0

			#add up displacement at each minute. (crnt-time,trgt-time]
			for mins in range(trgt_disp):
				#	  close at each future min,	current close
				area += close[(i+ mins+ 1) %l] - close[i %l]

			#append each target binary classification based on sign of area
			row.append(0 if area < 0 else 1)
		
		#NOTE HERE all targets of the given sample i are in row HERE END#NOTE
		new_data.append(row)

	#target column name information, a for area, 2 for # classes
	cols = [f'tc_2a_{i}m' for i in range(5, 61, 5)]

	#convert all of this to a new DataFrame
	target_set = pd.DataFrame(new_data, columns=cols)

	return target_set

'''-------------------------------------------------------------------------------
	NOTE FEATURE/TARGET NAME FUNCTIONS
	NOTE fn_ denotes 'feature(/set) names' for mass feature dropping
	NOTE tn_ denotes 'target (/set) names' for mass target  dropping
'''#------------------------------------------------------------------------------

def fn_vel(index):
	return [f'vel{i}_{idx[index]}' for i in range(1,61)]
def fn_acc(index):
	return [f'acc{i+1}_{idx[index]}' for i in range(60)]
def fn_stoch_k(index):
	return [f'stchK{i}_{idx[index]}' for i in range(5, 125, 5)]
def fn_vol_m(index):
	'''NOTE subset 1 of fe_vol_sz_diff feature set END NOTE'''
	return [f'vol_m{i}_{idx[index]}' for i in range(2,61)]
def fn_vol_avgDiff(index):
	'''NOTE subset 2 of fe_vol_sz_diff feature set END NOTE'''
	return [f'vol_avgDiff{i}_{idx[index]}' for i in range(2,61)]
def fn_ma_s60(index):
	'''NOTE the sub 60m subset of ma feature set'''
	return [f'ma{i}_{idx[index]}' for i in range(2,60)]
def fn_ma_s240(index):
	'''NOTE the sub 240m subset of ma feature set'''
	return [f'ma{i}_{idx[index]}' for i in range(60,241,20)]
def fn_disp_ma60(index):
	'''NOTE the sub 60m subset of ma feature set'''
	return [f'disp_ma{i}_{idx[index]}' for i in range(2,60)]
def fn_disp_ma240(index):
	'''NOTE the sub 240m subset of ma feature set'''
	return [f'disp_ma{i}_{idx[index]}' for i in range(60,241,20)]
def fn_ma_diff(index):
	cols = []
	lengths = [5,15,30,60,120,240]
	for i in range(len(lengths)):
		for j in range(i+1,len(lengths)):
			cols.append(f'diff_ma_{lengths[i]}_{lengths[j]}_{idx[index]}')
	return cols

'''NOTE TIMES-------------REFERENCE NOTE'''
#NOTE times = [5,15,30,60,120,240]''' NOTE#
'''NOTE TIMES-------------REFERENCE NOTE'''

def fn_hihi(index):
	'''NOTE first subset of fe_hihi_diff END NOTE'''
	return [f'hihi{i}_{idx[index]}' for i in times]
def fn_disp_hihi(index):
	'''NOTE second subset of fe_hihi_diff END NOTE'''
	return [f'disp_hihi{i}_{idx[index]}' for i in times]
def fn_lolo(index):
	'''NOTE first subset of fe_lolo_diff END NOTE'''
	return [f'lolo{i}_{idx[index]}' for i in times]
def fn_disp_lolo(index):
	'''NOTE second subset of fe_lolo_diff END NOTE'''
	return [f'disp_lolo{i}_{idx[index]}' for i in times]
def fn_hilo_stoch(index):
	cols = []
	#prepping the feature names according to ma's used
	for i in range(len(times)):
		for j in range(len(times)):
			cols.append(f'hilo_stoch_{times[i]}_{times[j]}_{idx[index]}')
	return cols
#fe_height_bar
#fe_height_wick
#fe_diff_hl_wick
#fe_vol_sz_diff
#fe_ma_disp
#fe_ma_diff
#fe_lolo_diff
#fe_hihi_diff
#fe_hilo_diff

def fn_all_subsets(real_prices: bool = False, indices:int=-1, keep_time:bool=True):
	'''
	This function creates a list of lists for each subsection of feature types.
	I'm typing this as I am implementing the second index data, this is getting pretty complex.
	Will probably have to look for a more organized method of sorting. Godspeed

	Params:
	- real-prices:
	-	boolean to allow for the original high,low,close,vol,time to be included as features
	- indices:
	-	option to pick how feature subsets are split based off of which index the data is coming from.
	-	 0) first index
	-	 1) second index
	-	-1) all indices as combined set 
	-	-2) all indices as seperate sets
	'''
	#feature name subsets
	fnsub = []
	# will append each individual feature/f_set here
	if(real_prices):
		if(indices == 0):
			if(keep_time):
				fnsub.append(['high','low','close','time','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
			else:
				fnsub.append(['high','low','close','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
		elif(indices == 1):
			if(keep_time):
				fnsub.append(['high.1','low.1','close.1','time','volume.1','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['high.1','low.1','close.1','volume.1','barH_ndx','wickH_ndx','diff_wick_ndx'])
		elif(indices == -1):
			if(keep_time):
				fnsub.append(['high','low','close','time','volume',\
						'high.1','low.1','close.1','volume.1',\
						'ToD','DoW','barH_spx','wickH_spx','diff_wick_spx',\
								'barH_ndx','wickH_ndx','diff_wick_ndx'])#removable real prices
			else:
				fnsub.append(['high','low','close','volume',\
						'high.1','low.1','close.1','volume.1',\
						'ToD','DoW','barH_spx','wickH_spx','diff_wick_spx',\
								'barH_ndx','wickH_ndx','diff_wick_ndx'])#removable real prices
		elif(indices == -2):
			if(keep_time):
				fnsub.append(['high','low','close','time','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
				fnsub.append(['high.1','low.1','close.1','volume.1','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['high','low','close','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
				fnsub.append(['high.1','low.1','close.1','volume.1','barH_ndx','wickH_ndx','diff_wick_ndx'])
	else:
		if(indices == 0):
			if(keep_time):
				fnsub.append(['time','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
			else:
				fnsub.append(['volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
		if(indices == 1):
			if(keep_time):
				fnsub.append(['time','volume.1','ToD','DoW','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['volume.1','ToD','DoW','barH_ndx','wickH_ndx','diff_wick_ndx'])
		if(indices ==-1):
			if(keep_time):
				fnsub.append(['time','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx','barH_ndx','wickH_ndx','diff_wick_ndx'])
		if(indices ==-2):
			if(keep_time):
				fnsub.append(['time','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
				fnsub.append(['volume.1','ToD','DoW','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
				fnsub.append(['volume.1','ToD','DoW','barH_ndx','wickH_ndx','diff_wick_ndx'])
		
	if(indices == 0 or indices ==-2):
		#		NOTE NOTE NOTE HERE IS THE IMPLEMENTATION OF ALL INDEX #1 (SPX) DATA. END NOTE END NOTE END NOTE		#
		fnsub.append(fn_vel(0))
		fnsub.append(fn_acc(0))
		fnsub.append(fn_stoch_k(0))
		fnsub.append(fn_vol_m(0))
		fnsub.append(fn_vol_avgDiff(0))
		if(real_prices):
			fnsub.append(fn_ma_s60(0))#removable
			fnsub.append(fn_ma_s240(0))#removable
		fnsub.append(fn_disp_ma60(0))
		fnsub.append(fn_disp_ma240(0))
		fnsub.append(fn_ma_diff(0))
		if(real_prices):	
			fnsub.append(fn_hihi(0))#removable
		fnsub.append(fn_disp_hihi(0))
		if(real_prices):	
			fnsub.append(fn_lolo(0))#removable
		fnsub.append(fn_disp_lolo(0))
		fnsub.append(fn_hilo_diff(0))
		fnsub.append(fn_hilo_stoch(0))

	if(indices == 1 or indices ==-2):
		#		NOTE NOTE NOTE HERE IS THE IMPLEMENTATION OF ALL INDEX #2 (NDX) DATA. END NOTE END NOTE END NOTE 		#
		fnsub.append(fn_vel(1))
		fnsub.append(fn_acc(1))
		fnsub.append(fn_stoch_k(1))
		fnsub.append(fn_vol_m(1))
		fnsub.append(fn_vol_avgDiff(1))
		if(real_prices):
			fnsub.append(fn_ma_s60(1))#removable
			fnsub.append(fn_ma_s240(1))#removable
		fnsub.append(fn_disp_ma60(1))
		fnsub.append(fn_disp_ma240(1))
		fnsub.append(fn_ma_diff(1))
		if(real_prices):	
			fnsub.append(fn_hihi(1))#removable
		fnsub.append(fn_disp_hihi(1))
		if(real_prices):	
			fnsub.append(fn_lolo(1))#removable
		fnsub.append(fn_disp_lolo(1))
		fnsub.append(fn_hilo_diff(1))
		fnsub.append(fn_hilo_stoch(1))

	if(indices == -1):
		raise NotImplementedError(f"FATAL: inclusive set building for featuresets has not been implemented for general feature-sets. 0,1,-2 (1/26/25) values are working.")
	
	return fnsub

def fnsubset_to_indexdictlist(pddf_features, fnsub):
	'''This function takes: 
	-   a dataframe 
	-   list of lists of feature names
	  and turns it into 
	-   a list of dicts
	  The dicts are each feature name in each subset with its corresponding index in the dataframe.
	  '''
	feature_dicts = [{pddf_features.get_loc(feature): \
					  feature for feature in sublist} \
					  for sublist in fnsub]
	return feature_dicts

#this function returns the set of all names that are being requested.
#these are feature names that will likely be used to drop from the used dataset
def return_name_collection():

	set1 = fn_hilo_prices(0)
	set2 = fn_ma_prices(0)
	set3 = fn_orig_price(0)

	full_set = set1+set2+set3

	return full_set

def fn_hilo_diff(index):
	lengths = [5,15,30,60,120,240]
	cols = []
	#prepping the feature names according to ma's used
	for i in range(len(lengths)):
		for j in range(len(lengths)):
			cols.append(f'diff_hilo_{lengths[i]}_{lengths[j]}_{idx[index]}')
	return cols

def fn_orig_price(index=None):
	if(index == None):
		return ['high','low','close','high.1','low.1','close.1']
	elif(index == 0):
		return ['high','low','close']
	elif(index == 1):
		return ['high.1','low.1','close.1']
	else:
		raise ValueError(f"Fatal: fn_orig_price <-- return_name_collection <-- _Feature_Usage:\nIndex value of {index} not interpretable.")

def fn_orig_vol():
	return ['volume','volume.1']

def fn_orig_time():
	return ['time']

def fn_hilo_prices(index):
	fn_hihi = [f'hihi{i}_{idx[index]}' for i in times]
	fn_lolo = [f'lolo{i}_{idx[index]}' for i in times]
	cols = fn_hihi+fn_lolo
	return cols

def fn_ma_prices(index):
	cols = [f'ma{i}_{idx[index]}' for i in range(2,60)]+\
		   [f'ma{i}_{idx[index]}' for i in range(60,241,20)]
	return cols

def tn_regression():
	cols = [f'tr_{i}' for i in range(1,61)]
	return cols

def tn_regression_excpetion(exc):
	cols = [f'tr_{i}' for i in range(1,exc)]+\
		[f'tr_{i}' for i in range(exc+1,61)]
	return cols

def tn_classification():
	cols = [f'tc_2c_{i}m' for i in range(5, 61, 5)]+\
		[f'tc_3c_5p_{i}m' for i in range(5, 61, 5)]+\
		[f'tc_4c_5p_{i}m' for i in range(5, 61, 5)]
	return cols

def tn_area_classification():
	cols = [f'tc_2a_{i}m' for i in range(5, 61, 5)]
	return cols

def tn_area_classification_exception(exc):
	cols = [f'tc_2a_{i}m' for i in range(5,exc,5)]+\
		[f'tc_2a_{i}m' for i in range(exc+5,61,5)]
	return cols

def tn_classification_exception(num_classes, class_split, minute):
	#value split is currently constant at 5,
	#so class_split is not used
	cols = []
	if(num_classes == 2):
		cols+=\
			[f'tc_2c_{i}m' for i in range(5, minute, 5)]+\
			[f'tc_2c_{i}m' for i in range(minute+5, 61, 5)]+\
			[f'tc_3c_5p_{i}m' for i in range(5, 61, 5)]+\
			[f'tc_4c_5p_{i}m' for i in range(5, 61, 5)]
	if(num_classes == 3):
		cols+=\
			[f'tc_2c_{i}m' for i in range(5, 61, 5)]+\
			[f'tc_3c_5p_{i}m' for i in range(5, minute, 5)]+\
			[f'tc_3c_5p_{i}m' for i in range(minute+5, 61, 5)]+\
			[f'tc_4c_5p_{i}m' for i in range(5, 61, 5)]
	if(num_classes == 4):
		cols+=\
			[f'tc_2c_{i}m' for i in range(5, 61, 5)]+\
			[f'tc_3c_5p_{i}m' for i in range(5, 61, 5)]+\
			[f'tc_4c_5p_{i}m' for i in range(5, minute, 5)]+\
			[f'tc_4c_5p_{i}m' for i in range(minute+5, 61, 5)]
	return cols

def tn_stoch_classification(num_classes, index=0):
	return [f'tsc_{num_classes}_{i}_{j}_{idx[index]}' for i in range(15, 125, 15) for j in range(0, 65, 15)]

def tn_stoch_classification_exception(num_classes, past_future_string, index=0):
	all_tn = tn_stoch_classification(num_classes=num_classes, index=index)
	return [str for str in all_tn if past_future_string not in str]

def tn_stoch_regression():
	return

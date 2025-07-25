'''
This file will contain all evaluation functions - created 3/10/2025
This file is used for reference, it is from my first 
genetic algorithms project and I would like to bring over some ideas,
and to do that I need to really read through the meat of this.
'''



#DEV NOTE ensure all fitfuncs are generated parallel to data to allow for parallel analysis.

import numpy as np
from math import sqrt
from operator import attrgetter
import matplotlib.pyplot as plt
import _00_gene as _0
from typing import Literal
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys

def fitness(
	arr_returns	:	np.ndarray	=	None,
	arr_kratio	:	np.ndarray	=	None,
	arr_holdfor	:	np.ndarray	=	None,
	data		:	np.ndarray	=	None,
	genes		:	list|np.ndarray	=	None,
	#NOTE removing method parameter, as fully inclusive evaluations will be done first.
	#method		:	function	=	None,
	lag_allow	:	int			=	-1,
):
	'''
	### info: ###
	 goes here
	### params:
	returns:
	-
	'''
	#function bare-bones assertions
	#assert data != None, "No data was provided to the fitness function."
	assert genes != None, "No genes were provided to the fitness function."
	#NOTE removed method assertion as im going to do full inclusive runs first END#NOTE!!!
	#assert method != None, "No ground-truth method was provided to the fitness function."
	assert lag_allow != -1, "No Lagallow length was provided to the fitness function."

	#boolean 2d array containing entry/or-not (0|1) for each gene
	#gene_presence = []

	gene_presence_local = []

	length = len(data)

	returns = np.zeros((length, len(genes)), dtype=np.float32)
	kelsch_ratio = np.zeros((length, len(genes)), dtype=np.float32)

	#new variable added to stop gene from entering multiple times 
	#within a single actual trade 
	in_trade_till = np.zeros(len(genes), dtype=int)

	#test all samples in the set, accounting for
	#lag allowance and hold length
	for i in range(length):
		
		if(i < lag_allow | i > length-1):
			#want to avoid usage of these values for safe analysis
			#gene_presence.append([0]*len(genes))
			gene_presence_local = np.zeros(len(genes), dtype=int)

			r = np.multiply(gene_presence_local, arr_returns[i], dtype=np.float32)
			kr= np.multiply(gene_presence_local, arr_kratio[i], dtype=np.float32)

			returns[i] = r
			kelsch_ratio[i] = kr
		else:

			i_presence = np.zeros(len(genes), dtype=int)

			if(i%25000==0):
				sys.stdout.write(f"\r{i}")
				sys.stdout.flush()

			#i_presence = [1 if all(p._op(data[i-p._l1, p._v1], data[i-p._l2, p._v2]) for p in gene._patterns) else 0 for gene in genes]
		
			#check presence of each gene at each sample
			for g, gene in enumerate(genes):

				#always initially set to true
				matches = True

				#added this in trade till if statement this way to enforce that a strategy is only entering one time per single / hold
				#and at the same time it also allows for unneded computation to be skipped out on, since this function is unfortunately
				#already slow that optimization is really helpful.. hopefully. have not run it yet. 

				#check to see if this gene is in a trade already
				#this if is true if it is not in a trade
				if(in_trade_till[g]<i):

					#now that this gene has entered a trade, needs to reset when it is allowed to
					#enter a new trade. we are going to do this by grabbing the hold_for value
					#for the array brought in from _01.collect_parallel_metrics on initial return collection
					in_trade_till[g] = arr_holdfor[i]+i
					#this essentially tells the gene (in a safe and blind manner) that it cannot enter a trade
					#until the time in which it ends up exiting the trade its in.

					#check each pattern within the gene
					for p in gene._patterns:

						#some referrential point printouts if needed
						#print(f"p: {type(p)} at {g}")
						#print(f'v1,v2,l1,l2: {p._v1} {p._v2} {p._l1} {p._l2}')
						
						#if given pattern does not hold true
						if not (p._op(data[i-p._l1, p._v1],data[i-p._l2, p._v2])):
							matches = False
							break

				#this else is reached if the gene is already in the trade
				else:
					matches = False
				#check if matches variable held
				i_presence[g] = 1 if matches else 0
		
			#now have a fully built sample presence
			#gene_presence.append(i_presence)

			gene_presence_local = i_presence

			r = np.multiply(gene_presence_local, arr_returns[i], dtype=np.float32)
			kr= np.multiply(gene_presence_local, arr_kratio[i], dtype=np.float32)

			#since we have moved returns and kelsch ratio to an earlier step, append those values now
			returns[i] = r
			kelsch_ratio[i] = kr
			
	#gene_presence = np.array(gene_presence)
	returns = returns
	kelsch_ratio = kelsch_ratio
	
	#This function will by default return the returns and kelsch_index values for each gene
	#these are iterable, along dim0 (by data sample) are gene column local values
	return returns, kelsch_ratio


def associate(
	genes	:	list,
	returns,
	kelsch_ratio,
	log_normalize,
	with_array	:	bool
):
	#iterate through all genes and associate calculated values and collected arrays
	for gi, gene in enumerate(genes):
		
		#calculate relevant statistics for each gene
		local_profit_factor = profit_factor(returns[:, gi])
		local_avg_return = average_nonzero(returns[:, gi])
		local_avg_kelsch_ratio = average_nonzero(kelsch_ratio[:, gi])
		local_total_return = total_return(returns[:, gi])
		local_frequency = frequency(returns[:, gi])
		local_total_kelsch_ratio = total_return(kelsch_ratio[:, gi])
		local_martin_ratio = martin_ratio(returns[:, gi])
		local_mkr = martin_ratio(kelsch_ratio[:, gi])
		local_r2 = r2(returns[:, gi])
		local_r2_kr = r2(kelsch_ratio[:, gi])

		#if this is reached, this means the returns are coming in as the percent
		#difference for each trade IN LOG SPACE
		#and also that the kelsch ratios are coming in as the percent difference
		#minus standard deviation of drawdowns ALL IN LOG SPACE
		if(log_normalize):
			
			#bring them out of log space
			local_avg_return = (local_avg_return)
			local_avg_kelsch_ratio = (local_avg_kelsch_ratio)
			local_total_return = (local_total_return)
			#these are now exact average % price differences (KR having some complexities)
			

		#update data within the gene for local storage for quick evaluation or recall
		gene.update(
			array_returns		=	returns[:, gi] if with_array else None,
			#array_kelsch_ratio	=	kelsch_ratio[:, gi],
			avg_returns			=	local_avg_return,
			avg_kelsch_ratio	=	local_avg_kelsch_ratio,
			profit_factor		=	local_profit_factor,
			total_return		=	local_total_return,
			frequency			=	local_frequency,
			total_kelsch_ratio	=	local_total_kelsch_ratio,
			martin_ratio		=	local_martin_ratio,
			mkr					=	local_mkr,
			r2					=	local_r2,
			r2_kr				=	local_r2_kr
		)

	#returns updated genes
	return genes

def sort_population(
	population	:	list	=	None,
	criteria	:	Literal['profit_factor','kelsch_ratio','average_return','total_return','consistency',\
							'frequency','total_kelsch_ratio','martin_ratio','mkr','r2','r2_kr']	=	'profit_factor'
):
	'''
	This function sorts a population based on a specific criteria
	'''

	assert (population != None), "Tried to sort a population that came in as None."
	
	#variable used for sorting in sorted function in attribute getter from operator lib
	metric = ""

	#for each type of criteria added
	match(criteria):
		#profit factor
		case 'profit_factor':
			metric = "profit_factor"
		#average return
		case 'average_return':
			metric = "avg_returns"
		#kelsch ratio
		case 'kelsch_ratio':
			metric = "avg_kelsch_ratio"
		case 'total_return':
			metric = "total_return"
		case 'consistency':
			metric = "consistency"
		case 'frequency':
			metric = "frequency"
		case "total_kelsch_ratio":
			metric = "total_kelsch_ratio"
		case "martin_ratio":
			metric = "martin_ratio"
		case "mkr":
			metric = "mkr"
		case "r2":
			metric = "r2"
		case "r2_kr":
			metric = "r2_kr"
		#invalid entry, should be impossible anyways
		case _:
			raise ValueError(f"FATAL: Tried sorting population with invalid criteria ({criteria})")
		
	sorted_pop = sorted(population, key=attrgetter(metric), reverse=True)

	#return population sorted by specified metric within each gene
	return sorted_pop

def show_best_gene_patterns(
	population	:	list	=	None,
	criteria	:	Literal['profit_factor','kelsch_ratio','average_return','total_return','consistency',\
							'frequency','total_kelsch_ratio','martin_ratio','mkr','r2','r2_kr']	=	'profit_factor',
	fss			:	list	=	None,
	exit_cond	:	int		=	-1
):
	'''
	This function shows the basic data of the best gene in a list of genes (population)
	'''

	#variable to collect string
	output=""

	#sort population so we can grab the first guy
	s_p = sort_population(population,criteria)

	#use the pattern class built in function to show all patterns first
	output+=f"{s_p[0].show_patterns(fss)}"

	profit_factor = round(s_p[0]._profit_factor, 5)

	#collect more interpretable data for the return and KRatio
	#this is done by considering their values being percent differences, and
	#multiplying it by a very rough real price guesstimate, also assuming this is on SPY
	avg_return_ticks = round( (s_p[0]._avg_returns)*5000 , 2) #5k is super rough estimate on spy price
	avg_kratio_ticks = round( (s_p[0]._avg_kelsch_ratio)*5000 , 2) #5k is super rough estimate on spy price
	mdn_ret = median_nonzero(s_p[0]._array_returns)
	mdn_return_ticks = (mdn_ret)*5000

	#then show basic metrics of the gene across last test
	output+=f"Profit Factor: {str(profit_factor)}\n"
	output+=str(f"Average Return: {str(round(s_p[0]._avg_returns,5))} (~{round(avg_return_ticks,3)} on /MES == ${round(avg_return_ticks*5, 2)})\n")
	output+=str(f"Median Return: {str(round(mdn_ret,5))} (~{round(mdn_return_ticks,2)} on /MES == ${round(mdn_return_ticks*5, 2)})\n")
	output+=str(f"Average KRatio: {round(s_p[0]._avg_kelsch_ratio, 5)}\n")
	output+=str(f"MKR: {round(s_p[0]._mkr,4)}\n")
	output+=str(f"Frequency: {s_p[0]._frequency}\n")
	output+=str(f"Exit Condition: {exit_cond}\n")
	output+=str(f"r2: {s_p[0]._r2}")

	return output


def profit_factor(
	returns	:	any	=	None
):

	wins, losses = 0, 0
	for i in returns:
		if(i>0):
			wins+=1
		if(i<0):
			losses+=1
	
	if(losses == 0):
		if(wins>50):
			raise ValueError(f'Perfect strategy detected, 0 losses, >50 wins!!!!!! AHAHAHAHA go check your code bro')
		else:
			return 0

	return round((wins/losses), 4)


def total_return(
	returns	:	np.ndarray	=	None
):
	return sum(returns)


def average_nonzero(
	array	:	np.ndarray	=	None	
):
	#will be traditionally used for averaging returns or ratios
	filtered = array[array != 0]

	avg = (np.mean(filtered)) if filtered.size > 0 else 0

	return avg

def median_nonzero(
	array	:	np.ndarray	=	None
):
	filtered = array[array != 0]

	mdn = (np.median(filtered)) if filtered.size > 0 else 0

	return mdn

def total_returns(
	array	:	np.ndarray	=	None
):
	return np.sum(array)


def martin_ratio(
	returns	:	np.ndarray	=	None
):
	#get nonzero returns
	ret_nonzero = returns[returns != 0]

	#collect the average trade return
	if(ret_nonzero.size > 0):
		avg_ret = np.mean(ret_nonzero)
	else:
		avg_ret = 0

	#collect the ulcer index of the strategy
	ulcer_i = ulcer_index(returns)

	#create the martin ratio of the current strategy
	martin_r = (avg_ret / ulcer_i) if ulcer_i > 0 else 0

	#print(f"MR OUT {round(martin_r, 2)}")

	#return said ratio
	return martin_r

def ulcer_index(
	returns	:	np.ndarray	=	None
):
	#array of developing strategy PL
	cum_ret = np.cumsum(returns)

	#array of developing strategy max PL
	cum_max = np.maximum.accumulate(cum_ret)

	#array of developing strategy drawdowns
	ind_ddn = (cum_max - cum_ret)

	#create a standard drawdown variable
	std_ddn = np.sqrt(np.mean(ind_ddn ** 2)) if ind_ddn.size > 0 else 0
	
	#return the standard drawdown
	return std_ddn


def frequency(
	returns	:	np.ndarray	=	None
):
	'''
	This doesnt happen to account for trades that were exactly zero but thats just how it will have to be
	'''
	return (np.sum(returns != 0)/returns.size)

def r2(
	returns	:	np.ndarray	=	None
):
	
	#first collect cumulative value of array
	cum_ret = np.cumsum(returns)

	#collect a time array for linear model building
	time = np.arange(len(cum_ret)).reshape(-1,1)

	#fit a linear model to the dynamics of the return
	model = LinearRegression()
	model.fit(time, cum_ret)
	
	#create a trend line of returns over time
	trend_line = model.predict(time)

	try:
		#using pearsons r2 instead of regular to be less affected by scaling
		#NOTE definitely need to test this out 3/28/25 possibly wrong spot in conf_matx
		pr2 = np.corrcoef(cum_ret, trend_line)[0, 1] ** 2
	except FloatingPointError as e:
		return 0
	except Exception as e:
		return 0

	return pr2


def simple_generational_stat_output(
	population	:	list	=	None,
	metric		:	str		=	None
):
	all_metrics = []

	#for each type of criteria added
	match(metric):
		#profit factor
		case 'profit_factor':
			metric = "profit_factor"
		#average retur
		case 'average_return':
			metric = "avg_returns"
		#kelsch ratio
		case 'kelsch_ratio':
			metric = "avg_kelsch_ratio"
		case 'total_return':
			metric = "total_return"
		case 'consistency':
			metric = "consistency"
		case 'frequency':
			metric = "frequency"
		case "total_kelsch_ratio":
			metric = "total_kelsch_ratio"
		case "martin_ratio":
			metric = "martin_ratio"
		case "mkr":
			metric = "mkr"
		case "r2":
			metric = "r2"
		case "r2_kr":
			metric = "r2_kr"
		#invalid entry, should be impossible anyways
		case _:
			raise ValueError(f"FATAL: Tried sorting population with invalid criteria ({metric})")

	fetch_metric = attrgetter(metric)

	for gene in population:

		all_metrics.append(fetch_metric(gene))

	if(len(all_metrics) > 0):
		avg_metric = np.mean(all_metrics)
		top_metric = max(all_metrics)
	else:
		avg_metric = -1
		top_metric = -1

	return avg_metric, top_metric
	

def show_returns(
	arr_returns	:	np.ndarray	=	None,
	arr_close	:	np.ndarray	=	None,
	gene_kwargs	:	any			=	None
):
	cum_pl = []
	total = 0

	base_pl = []
	base_tot= 0

	print(len(arr_returns))
	print(len(arr_close))

	for i, r in enumerate(arr_returns):
		total+=r
		cum_pl.append(total)
		base_tot=(arr_close[i]/arr_close[0])-1
		base_pl.append(base_tot)

	gene_info = show_best_gene_patterns(**gene_kwargs)

	plt.title(gene_info)

	plt.plot(base_pl, color='black', label='Market Return')
	plt.plot(cum_pl,color='maroon', label='Strategy Return')

	plt.legend()
	plt.show()

	plt.plot(cum_pl,color='maroon')
	plt.show()

	# Plot histogram

	arr_ret_nonzero = arr_returns[arr_returns != 0]

	plt.figure(figsize=(10, 6))
	plt.hist(arr_ret_nonzero, bins='auto', edgecolor='black', alpha=0.75)

	# Formatting
	plt.title('Return Distribution')
	plt.xlabel('Return')
	plt.ylabel('Frequency')
	plt.grid(True)
	plt.tight_layout()

	plt.show()


def filter_population(
	population	:	list	=	[],
	avg_return	:	float	=	-100,
	tot_return	:	float	=	-100,
	profit_factor	:	float=	0,
	kelsch_ratio	:	float=	-100,
	entry_frequency	:	float=	0.00,
	r2			:	float=	0,
	r2_kr		:	float=	0
):
	'''
	This function takes a few different areas and filters the population based on such
	'''

	filtered_population = population

	pop_list = []

	for g, gene in enumerate(population):

		#check first for insufficient avg return
		if(gene._avg_returns < avg_return):
			#print(f"pop avg return {gene._lastavg_returns}")
			pop_list.append(g)
		elif(gene._total_return < tot_return):
			#print(f"pop tot return {gene._last_total_return}")
			pop_list.append(g)
		elif(gene._profit_factor < profit_factor):
			#print(f"pop prof fact {gene._last_profit_factor}")
			pop_list.append(g)
		elif(gene._avg_kelsch_ratio < kelsch_ratio):
			#print(f"pop kratio {gene._lastavg_kelsch_ratio}")
			pop_list.append(g)
		elif(gene._frequency < entry_frequency):
			#print(f"pop frequency {gene._last_frequency}")
			pop_list.append(g)
		elif(gene._r2 < r2):
			pop_list.append(g)
		elif(gene._r2_kr < r2_kr):
			pop_list.append(g)

	pop_list = sorted(pop_list, reverse=True)

	for i in pop_list:
		filtered_population.pop(i)

	return filtered_population

def serial_correlation(
	returns	:	np.ndarray	=	None
):
	'''
	This function will take in a returns array		<br>
	And will return a conf_matx or set of bins of mean next trade return	<br>
	based on number of trades considered (dim1)	and		<br>
	based on avg value of those trades (dim2)
	'''
	return

def single_fitness_association(
	gene	:	any,
	returns	:	np.ndarray,
	kratio	:	np.ndarray,
	data	:	np.ndarray,
	hold_for:	int,
	lag_allow:	int,
	log_norm:	bool
):
	
	r, kr = fitness(
		arr_kratio=kratio,
		arr_returns=returns,
		data=data,
		genes=[gene],
		hold_for=hold_for,
		lag_allow=lag_allow
	)

	gene = associate(
		genes=[gene],
		returns=r,
		kelsch_ratio=kr,
		log_normalize=log_norm
	)

	gene._array_returns = r

	return gene

def show_combined_performance(
	population	:	list,
	arr_close	:	np.ndarray,
	arr_low		:	np.ndarray,
	arr_returns	:	np.ndarray,
	arr_kratio	:	np.ndarray,
	data		:	np.ndarray,
	hold_for	:	int,
	lag_allow	:	int,
	specific_data:	str,#'form_519',
	log_normalize:	bool,
	criteria	:	str,
	fss			:	list
):
	
	returns, kelsch_ratio = fitness(
		arr_close=arr_close,
		arr_low=arr_low,
		arr_returns=arr_returns,
		arr_kratio=arr_kratio,
		data=data,
		genes= population,
		hold_for=hold_for,
		lag_allow=lag_allow,
		specific_data=specific_data,
		log_normalize=log_normalize
	)



	col_ret = returns.mean(axis=1, keepdims=True)
	#col_kr = kelsch_ratio.sum(axis=1, keepdims=True)


	new = _0.Gene(patterns=[])

	unsorted = associate(
		genes=[new],
		returns=col_ret,
		kelsch_ratio=kelsch_ratio,
		log_normalize=log_normalize
	)



	unsorted[0].show_patterns(fss=fss)

	total=0

	base_pl = []
	cum_pl = []


	unsorted[0]._array_returns = col_ret
	

	show_returns(
		unsorted[0]._array_returns,
		arr_close=arr_close,
		gene_kwargs={"population":unsorted,"criteria":criteria,"fss":fss}
	)

	return unsorted[0]
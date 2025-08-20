import transforms as transforms
import population as population
import math
import copy
import statistics
import utility as utility
from typing import Literal
import numpy as np

def reproduce_scarce(
	forest	:	list,
	scores	:	list,
	size	:	int,
	sthresh	:	float,
	dflt_dpth	:	int|tuple	=	None,
	MRC		:	tuple	=	(0.33, 0.33, 0.34),
	metric	:	Literal['adjEV','loss'] = 'loss'
):

	match(metric):
		case 'loss':
			scored_forests = sorted(zip(forest, scores), key=lambda pair: pair[1])
			sorted_forest, sorted_scores = map(list, zip(*scored_forests))
		case 'adjEV':
			scores = list(np.exp(scores))
			scored_forests = sorted(zip(forest, scores), key=lambda pair: pair[1], reverse=True)
			sorted_forest, sorted_scores = map(list, zip(*scored_forests))


	#count how many trees are implied elite from their scores
	i = 0
	while(i<size):
		match(metric):
			case 'loss':
				if(sorted_scores[i]>sthresh):
					break
			case 'adjEV':
				if(sorted_scores[i]<=sthresh):
					break
		i+=1
	
	num_offspring = size-i

	m_size = int(num_offspring*MRC[0])
	c_size = int(num_offspring*MRC[2])
	r_size = size - m_size - c_size

	new_forest = copy.deepcopy(sorted_forest[:i+1])

	#make sure we are inverting scores according to derivative of desirable scores
	match(metric):
		case 'loss':
			invert_weights = True
		case 'adjEV':
			invert_weights = False

	#pick random trees from inverse transform sampling of scores
	m_idx = utility.random_sample_n_weighted(sorted_scores, m_size, inverse=invert_weights)

	for m in m_idx:

		new_tree:transforms.T_node = sorted_forest[m].copy()
		ptr = transforms.get_random_node(new_tree)
		
		#mutate at randomly reached branch in the new tree
		ptr.random()

		new_forest.append(new_tree)

	#second section is we select our C
	#C is crossover (branch swapping)

	for c in range(c_size):

		#need to go get two different trees, one for giving a branch and one for recieving a branch

		#this will have two forest indices, [male, female]
		c_idx = utility.random_sample_n_weighted(sorted_scores, 2, let_dupe=False, inverse=invert_weights)

		#go get detached branch from male tree
		male_ptr = sorted_forest[c_idx[0]]

		new_branch:transforms.T_node = transforms.get_random_node(male_ptr).copy()

		new_tree = sorted_forest[c_idx[1]].copy()

		#go get reattach branch location from female tree
		female_ptr = transforms.get_random_node(new_tree)

		female_ptr.replace(new_branch)

		new_forest.append(new_tree)

	#fourth section is we select our R
	#R is random

	#go find average depth of the current forest
	depths = [transforms.get_tree_depth(tree)[0]+transforms.get_tree_depth(tree)[1] for tree in sorted_forest]
	if(dflt_dpth==None):
		dflt_dpth = int(statistics.mean(depths)-statistics.stdev(depths))
	
	#make new trees and move them over to the new forest
	new_trees = population.generate_random_forest(r_size, dflt_dpth)
	for r in range(r_size):
		new_forest.append(new_trees[r])


	return new_forest


def reproduce(
    forest	:	list,
    scores	:	list,
	dflt_dpth	:	int	=	None,
    MERC	:	tuple	=	(0.05, 0.05, 0.1, 0.8)
):
	'''
	MERC is (mutate, elite, random, crossover)
	'''

    
	size = len(forest)
	m_size = math.floor(size*MERC[0])
	e_size = math.floor(size*MERC[1])
	c_size = math.floor(size*MERC[3])
	r_size = size - m_size - e_size - c_size

	new_forest = []

	#first section is we select our M
	#M is mutation

	#pick random trees from inverse transform sampling of scores
	m_idx = utility.random_sample_n_weighted(scores, m_size)

	for m in m_idx:

		new_tree:transforms.T_node = forest[m].copy()
		ptr = transforms.get_random_node(new_tree)
		
		#mutate at randomly reached branch in the new tree
		ptr.random()

		new_forest.append(new_tree)

	#second section is we select our E
	#E is elitism

	#pick random trees from inverse transform sampling of scores
	e_idx = utility.random_sample_n_weighted(scores, e_size)

	for e in e_idx:
		new_tree:transforms.T_node = forest[e].copy()
		new_forest.append(new_tree)

	#third section is we select our C
	#C is crossover (branch swapping)

	for c in range(c_size):

		#need to go get two different trees, one for giving a branch and one for recieving a branch

		#this will have two forest indices, [male, female]
		c_idx = utility.random_sample_n_weighted(scores, 2, let_dupe=False)

		#go get detached branch from male tree
		male_ptr = forest[c_idx[0]]

		new_branch:transforms.T_node = transforms.get_random_node(male_ptr).copy()

		new_tree = forest[c_idx[1]].copy()

		#go get reattach branch location from female tree
		female_ptr = transforms.get_random_node(new_tree)

		female_ptr.replace(new_branch)

		new_forest.append(new_tree)

	#fourth section is we select our R
	#R is random

	#go find average depth of the current forest
	depths = [transforms.get_tree_depth(tree)[0]+transforms.get_tree_depth(tree)[1] for tree in forest]
	if(dflt_dpth==None):
		dflt_dpth = int(statistics.mean(depths)-statistics.stdev(depths))
	
	#make new trees and move them over to the new forest
	new_trees = population.generate_random_forest(r_size, dflt_dpth)
	for r in range(r_size):
		new_forest.append(new_trees[r])

	return new_forest
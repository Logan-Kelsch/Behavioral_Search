import transforms as transforms
import population as population
import math
import statistics
import utility as utility

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
	m_idx = utility.random_sample_n_inverse_weighted(scores, m_size)

	for m in m_idx:

		new_tree:transforms.T_node = forest[m].copy()
		ptr = transforms.get_random_node(new_tree)
		
		#mutate at randomly reached branch in the new tree
		ptr.random()

		new_forest.append(new_tree)

	#second section is we select our E
	#E is elitism

	#pick random trees from inverse transform sampling of scores
	e_idx = utility.random_sample_n_inverse_weighted(scores, e_size)

	for e in e_idx:
		new_tree:transforms.T_node = forest[e].copy()
		new_forest.append(new_tree)

	#third section is we select our C
	#C is crossover (branch swapping)

	for c in range(c_size):

		#need to go get two different trees, one for giving a branch and one for recieving a branch

		#this will have two forest indices, [male, female]
		c_idx = utility.random_sample_n_inverse_weighted(scores, 2, let_dupe=False)

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
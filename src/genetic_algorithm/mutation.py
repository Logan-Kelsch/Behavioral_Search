'''

'''

import genetic_algorithm.transforms as transforms
import genetic_algorithm.evaluation as evaluation
import genetic_algorithm.visualization as visualization

import random
import copy
import numpy as np
import pandas as pd
from IPython.display import Image, display

def branch_ntimes(
	tree	:	transforms.T_node,
	branches:	int	
)	->	transforms.T_node:
	
	for iter in range(branches):
		tree.mutate_tree()

	return tree


def quarter_forest_evolution(
	init_forest	:	np.ndarray,
	x_raw		:	np.ndarray,
	iterations	:	int	=	5
):
	
	'''
	This function is a beginning stage evolution function, where some
	population of trees (forest) is brought in. for each iteration,
	the forest is evaluated. all dead (<=0 score) trees are removed
	from the forest. for all surviving trees, the worst trees are removed
	down until only 25% of the forest remains, this being the best performing trees.
	'''

	print('entering preloop')
	iter_forest:list = init_forest

	f_size = len(iter_forest)
	q_size = int(np.floor(f_size/4))

	p_gen = transforms.forest2features(iter_forest, x_raw)
	p = pd.DataFrame(p_gen)

	visualization.visualize_all_distributions(x=p)

	p_scores, p_treelist, p_scorelist = evaluation.evaluate_forest(p_gen,x_raw[:,3], n_bins=300,lag_range=(2,4))

	#best_oplist = transforms.get_oplist(iter_forest[p_treelist[0]])
	img = visualization.visualize_tree(iter_forest[p_treelist[0]])
	display(img)

	print('exiting preloop, entering loop')

	for i in range(iterations-1):

		tmp_forest = iter_forest

		#remove all trees until top quarter exist
		while(len(p_treelist) > q_size):
			p_treelist.pop()

		quarter_trees = set(p_treelist)
		crnt_forest_size = len(iter_forest)
		for k in range(crnt_forest_size-1,-1,-1):
			if k not in quarter_trees:
				iter_forest.pop(k)

		print('popped all lists, adding all children')

		#make 3 children for each surviving tree
		for j in range(q_size):

			for three in [0,1,2]:
				iter_forest.append(
					branch_ntimes(
						copy.deepcopy(iter_forest[j]), 
						int(-np.ceil(np.log(random.random())))
					)
			)

		

		#now have a filled forest for iteration
			
		p_gen = transforms.forest2features(iter_forest, x_raw)
		p = pd.DataFrame(p_gen)

		visualization.visualize_all_distributions(x=p)

		p_scores, p_treelist, p_scorelist = evaluation.evaluate_forest(p_gen,x_raw[:,3], n_bins=300,lag_range=(2,4))

		#best_oplist = transforms.get_oplist(iter_forest[p_treelist[0]])
		img = visualization.visualize_tree(iter_forest[p_treelist[0]])
		display(img)
		
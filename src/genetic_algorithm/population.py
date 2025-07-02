from genetic_algorithm.transforms import *
from typing import Literal

def generate_random_forest(
	size	:	int	=	100,
	init_m	:	int =	5
) -> list:
	population = []

	for i in range(size):
		new_tree = T_node(random=True)
		for mutation in range(init_m):
			new_tree.mutate_tree()
		population.append(new_tree)

	return population

def sort_forest(
	population	:	list,
	score_list	:	list,
	order_list	:	list
):
	return [population[i] for i in order_list], [score_list[i] for i in order_list]

def oplist2forests(
	oplists		:	list,
	prll_idx	:	list,
	batch_size	:	int
):
	
	tot_len = len(oplists)
	n_trees = len(oplists)
	n_batches = np.ceil(tot_len/batch_size)
	

	forest_batches	= [[] for _ in range(n_batches)]
	prll_idx_batches= [[] for _ in range(n_batches)]

	for b in range(n_batches):

		for i in range(n_trees):

			#b+1 skips all already evaluated trees
			new_tree = oplist2tree(oplists[int((b)*n_trees)+i])
			forest_batches[b].append(new_tree)

			prll_idx_batches[b].append(prll_idx[int((b)*n_trees)+i])

	return forest_batches, prll_idx_batches
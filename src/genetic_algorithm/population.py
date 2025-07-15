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
	
	n_trees = len(oplists)
	n_batches = int(np.ceil(n_trees/batch_size))


	print(f'n_trees:{n_trees}\nn_batches:{n_batches}')
	

	forest_batches	= [[] for _ in range(n_batches)]
	prll_idx_batches= [[] for _ in range(n_batches)]

	add_n = n_trees

	for b in range(n_batches):

		for i in range(int(min(batch_size,add_n))):

			#print(f'oplist to tree on: {int((b)*batch_size)+i}')

			#b+1 skips all already evaluated trees
			new_tree = oplist2tree(oplists[int((b)*batch_size)+i])
			forest_batches[b].append(new_tree)

			prll_idx_batches[b].append(prll_idx[int((b)*batch_size)+i])

		add_n -= batch_size

	return forest_batches, prll_idx_batches

def extract_n_best_trees(
	forest	:	list,
	scores	:	list,
	n		:	int
):
	wf = forest.copy()
	ws = scores.copy()

	output_forest = []
	output_scores = []

	for it in range(n):
		best_idx = ws.index(min(ws))
		output_forest.append(wf[best_idx])
		output_scores.append(ws[best_idx])
		wf.pop(best_idx)
		ws.pop(best_idx)

	return output_forest, output_scores

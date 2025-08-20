import transforms as transforms
from typing import Literal
import numpy as np

def generate_random_forest(
	size	:	int	=	100,
	init_m	:	int|tuple =	5
) -> list:
	population = []

	for i in range(size):
		if(type(init_m)==tuple):
			muts = np.random.randint(init_m[0], init_m[1]+1)
		else:
			muts = init_m
		new_tree = transforms.T_node(random=True)
		for mutation in range(muts):
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


	#print(f'n_trees:{n_trees}\nn_batches:{n_batches}')
	

	forest_batches	= [[] for _ in range(n_batches)]
	prll_idx_batches= [[] for _ in range(n_batches)]

	add_n = n_trees

	for b in range(n_batches):

		for i in range(int(min(batch_size,add_n))):

			#print(f'oplist to tree on: {int((b)*batch_size)+i}')

			#b+1 skips all already evaluated trees
			new_tree = transforms.oplist2tree(oplists[int((b)*batch_size)+i])
			forest_batches[b].append(new_tree)

			prll_idx_batches[b].append(prll_idx[int((b)*batch_size)+i])

		add_n -= batch_size

	return forest_batches, prll_idx_batches

def remove_duplicates(
	forest	:	list,
	keep_cnt:	int	=	1
):
	
	idx_2del = []
	dupe_cnt = 0
	for t1 in range(len(forest)):
		for t2 in range(t1+1,len(forest)):

			if(forest[t1]==forest[t2]):
				dupe_cnt+=1

			if(dupe_cnt>keep_cnt):
				idx_2del.append(t2)

		dupe_cnt = 0
	
	del_these = sorted((list(set(idx_2del))),reverse=True)
	for didx in del_these:
		forest.pop(didx)

	return forest



import matplotlib.pyplot as plt

def extract_n_best_trees(
	forest	:	list,
	scores	:	list,
	n		:	int,
	run_dir	:	str	=	'',
	vizout	:	bool	=	False
):
	wf = forest.copy()
	ws = scores.copy()

	output_forest = []
	output_scores = []

	break_early = False
	if(n==-1):
		n = len(forest)
		break_early = True

	for it in range(n):
		best_idx = ws.index(min(ws))
		if(break_early):
			if(min(ws)>0.99):
				break
		output_forest.append(wf[best_idx])
		output_scores.append(ws[best_idx])
		wf.pop(best_idx)
		ws.pop(best_idx)

	if(vizout):
		plt.scatter(range(len(output_scores)), output_scores)
		plt.title('Scores of Selected Features for NN')
		plt.savefig(str(run_dir / 'selected_feats.png'))

	plt.close()

	return output_forest, output_scores

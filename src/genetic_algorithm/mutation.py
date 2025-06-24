'''

'''

import genetic_algorithm.transforms as transforms
import random

def mu_inter_feattrans(curr_trans:int):
	match(curr_trans):
		case 0:
			return random.choice([0,1,2,3,4])
		case 1:
			return random.choice([0])
	'''
	THIS STATE BASED GRAPH NEEDS TO CONSIDER HP OF WHAT TRANFORMATION AND MM OF WHAT TRANSFORMATION
	'''

def branch_ntimes(
	tree	:	transforms.T_node,
	branches:	int	
)	->	transforms.T_node:
	
	for iter in range(branches):
		tree.mutate_tree()

	return tree
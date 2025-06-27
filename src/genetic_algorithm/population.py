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

def initialize_interdimensional(
		
):
	return


def initialize_intradimensional(
		
):
	return
from genetic_algorithm.transforms import *
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

	

def initialize_interdimensional(
		
):
	return


def initialize_intradimensional(
		
):
	return
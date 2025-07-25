import dill as pickle

def save_forest(
	forest	:	list,
	name	:	str	=	'forest.4st',
	dirpath	:	str	=	''
):
	with open(str(dirpath / name), 'wb') as f:
		pickle.dump(forest, f, protocol=pickle.HIGHEST_PROTOCOL)
	print('forest saved.')


def load_forest(
	where	:	str	=	None
):
	with open(where, 'rb') as f:
		return pickle.load(f)
	
def save_deeplist(
	deep_list:	list,
	name	:	str	=	'deeplist.pkl',
	dirpath	:	str	=	''
):
	with open(str(dirpath / name), 'wb') as f:
		pickle.dump(deep_list, f, protocol=pickle.HIGHEST_PROTOCOL)
	print('deep list saved.')


def load_deeplist(
	where	:	str	=	None
):
	with open(where, 'rb') as f:
		return pickle.load(f)
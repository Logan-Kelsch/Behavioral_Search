'''
Current transforming function list:
- No transformation				(Identity,		ID, 0)
- Displacement from mean 		(demeaned, 		DM, 1)
- Standard deviation from mean 	(Volatility,	VL, 2)
- Min-Max Range Normalization 	(Stoch, 		MM, 4)
- Hawkes Self-Exciting Process 	(Hawkes,		HP, 3)

A direction graph for how these transformations can be applied on each other:

	_________>	DM ______\
   /			\/	     \/
ID __________>	MM	<____HP
   \			/\		 /\
	_________>	VL ______/
'''

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
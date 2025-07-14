from functools import lru_cache
from typing import List, Any, Tuple
import numpy as np



from typing import List, Any, Tuple

def shortest_common_supersequence(
    seqs: List[List[Any]]
) -> Tuple[List[Any], List[List[int]]]:
    """
    Greedy heuristic for shortest common supersequence:
      - At each step, pick the element that matches the most sequences at their current front.
      - Advance those sequences by one.
      - Continue until all sequences are consumed.
    Returns:
      - scs: the greedy supersequence
      - idx_lists: for each position in scs, the list of sequence indices advanced
    """
    k = len(seqs)
    lengths = [len(s) for s in seqs]
    pos = [0] * k
    scs: List[Any] = []
    idx_lists: List[List[int]] = []

    while not all(pos[i] >= lengths[i] for i in range(k)):
        # Gather candidates and which sequences they match
        candidates = {}
        for i in range(k):
            if pos[i] < lengths[i]:
                e = seqs[i][pos[i]]
                candidates.setdefault(e, []).append(i)

        # Pick the candidate covering the most sequences
        # Tie-break by choosing the candidate with smallest repr
        best_e = max(
            candidates.items(),
            key=lambda item: (len(item[1]), -ord(str(item[0])[0]) if isinstance(item[0], str) else len(item[1]))
        )[0]

        # Advance sequences matching best_e
        advanced = []
        for i in range(k):
            if pos[i] < lengths[i] and seqs[i][pos[i]] == best_e:
                pos[i] += 1
                advanced.append(i)

        scs.append(best_e)
        idx_lists.append(advanced)

    return scs, idx_lists


def quickfix_score_to_loss(
    scores  :   list
)   ->  list:
    
    losses = []

    for score in scores:

        losses.append( np.exp( 1 - np.exp(score) ) )

    return losses

def pinder_resize(
    val     :   float,
    pinder  :   float,
    space   :   tuple
):
    
    size_space = space[1]-space[0]

    if(space[1]<space[0]):
        raise ValueError(f"Defined space for pinder resize is not logical. Got [{space[0]},{space[1]}]")

    p_range = pinder*size_space

    under = val - p_range
    over = val + p_range

    #first check if under needs to be passed to over
    if(under<space[0]):
        over += (space[0]-under)
        under = space[0]

    if(over>space[1]):
        under += (over-space[1])
        over = space[1]

    if((under<space[0]) or (over>space[1])):
        raise(f"pinder resize failed! check code."
              f"val:{val},pinder:{pinder},space:{space}")
    
    return over, under
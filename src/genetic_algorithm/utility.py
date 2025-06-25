from functools import lru_cache
from typing import List, Any, Tuple



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

# Example usage:
# seqs = [[1,2,3], [2,1,3], [1,3,2]]
# scs, idxs = shortest_common_supersequence(seqs)
# print("SCS:", scs)
# print("Indices:", idxs)

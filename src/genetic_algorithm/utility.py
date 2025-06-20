from functools import lru_cache
from typing import List, Any, Tuple

def shortest_common_supersequence(
    seqs: List[List[Any]]
) -> Tuple[List[Any], List[List[int]]]:
    """
    Returns (scs, idx_lists) where
      - scs is a shortest common supersequence of the input `seqs`
      - idx_lists[i] is the list of sequence-indices that contributed
        the element scs[i] (i.e. were advanced) at that position.
    """
    k = len(seqs)
    lengths = tuple(len(s) for s in seqs)

    @lru_cache(None)
    def dp(pos: Tuple[int, ...]) -> Tuple[Tuple[Any, ...], Tuple[Tuple[int, ...], ...]]:
        # If all sequences are fully consumed, no more symbols or indices
        if pos == lengths:
            return (), ()

        # all next‐possible symbols
        candidates = {
            seqs[i][pos[i]]
            for i in range(k)
            if pos[i] < lengths[i]
        }

        best_seq: Tuple[Any, ...] = ()
        best_idxs: Tuple[Tuple[int, ...], ...] = ()
        first = True

        for e in candidates:
            # compute how far we advance in each sequence if we pick e
            new_pos = list(pos)
            advanced = []
            for i in range(k):
                if new_pos[i] < lengths[i] and seqs[i][new_pos[i]] == e:
                    new_pos[i] += 1
                    advanced.append(i)
            new_pos = tuple(new_pos)

            # recurse
            suffix_seq, suffix_idxs = dp(new_pos)

            cand_seq = (e,)+suffix_seq
            cand_idxs = (tuple(advanced),)+suffix_idxs

            if first or len(cand_seq) < len(best_seq):
                best_seq, best_idxs = cand_seq, cand_idxs
                first = False

        return best_seq, best_idxs

    scs_tuple, idxs_tuple = dp(tuple([0]*k))
    # convert to lists
    scs = list(scs_tuple)
    idx_lists = [list(tup) for tup in idxs_tuple]
    return scs, idx_lists
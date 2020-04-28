from functools import reduce
from itertools import product
import operator
from os import path
import pickle
from sympy.combinatorics import Permutation

weingarten_values = {}

# Loads precomputed Weingarten functions from https://motohisafukuda.github.io/RTNI/PYTHON/Weingarten/weingarten.html
def weingarten(perm, d_value):
    """ Compute Weingarten function for a given permutation and dimension.
    
    Parameters
    ----------
    perm: Permutation
        A sympy permutation.
    d_value: int
        The dimension.
    
    Returns
    -------
    float
        Value of the Weingarten function.
    """
    if perm.size > 20:
        raise ValueError("Permutation size should be < 20 to use tabulated Weingarten function values")
    if perm.size not in weingarten_values:
        with open(path.join(path.dirname(__file__), "weingarten", f"functions{perm.size}.pkl"), "rb") as f:
            weingarten_values[perm.size] = {
                tuple(partition): symbolic_value
                for partition, symbolic_value in pickle.load(f)
            }
    cycle_structure = tuple(sorted(reduce(
        operator.add,
        [[cycle_length] * n_cycles for cycle_length, n_cycles in perm.cycle_structure.items()]
    ), reverse=True))
    symbolic = weingarten_values[perm.size][cycle_structure]
    n = list(symbolic.free_symbols)[0]
    return complex(symbolic.subs({ n: d_value })).real

def generate_permuted_lists(l):
    """ Apply all permutations to a list and return the resulting lists (note that some lists
        will be repeated if the initial list has repeated elements).
    
    Parameters
    ----------
    l: list
        The list to be permuted.
    
    Returns
    -------
    list
        A list containing the lists obtained from applying all permutations to 'l'.
    """
    if len(l) == 0:
        return [[]]
    if len(l) == 1:
        return [[l[0]]]
    permuted_lists = []
    for index, elt in enumerate(l):
        for sub_permuted_list in generate_permuted_lists(l[:index] + l[index + 1:]):
            permuted_lists.append([elt] + sub_permuted_list)
    return permuted_lists

def generate_fixing_permutations(l):
    """ Explicitly generate the permutations leaving a list invariant.
    
    Parameters
    ----------
    l: list
        The list mentionned above.
    
    Returns
    -------
    list
        The list of the permutations (in sympy format) leaving the list invariant. 
    """
    distinct_elts = {}
    for index, elt in enumerate(l):
        if elt not in distinct_elts:
            distinct_elts[elt] = []
        distinct_elts[elt].append(index)
    fixing_permutations = []
    for sub_permutations in product(*[
        generate_permuted_lists(distinct_elt_indices)
        for distinct_elt, distinct_elt_indices in distinct_elts.items()
    ]):
        fixing_permutation = [None] * len(l)
        for sub_permutation_index, distinct_elt_indices in enumerate(distinct_elts.values()):
            for distinct_elt_index_enum, distinct_elt_index in enumerate(distinct_elt_indices):
                fixing_permutation[distinct_elt_index] = sub_permutations[sub_permutation_index][distinct_elt_index_enum]
        fixing_permutations.append(Permutation(fixing_permutation))
    return fixing_permutations

def find_mapping_permutation(source_list, target_list):
    """ Determine a permutation p (not unique if source_list has repeated elements) such that
        target_list[p[i]] = source_list[i].
        
    Parameters
    ----------
    source_list: list
        See description of the function.
    target_list: list
        See description of the function.
    
    Returns
    -------
    Permutation
        A sympy permutation describing the permutation p defined in the description of the function.
    """
    source_distinct_elts = {}
    target_distinct_elts = {}
    for source_index, source_value in enumerate(source_list):
        if source_value not in source_distinct_elts:
            source_distinct_elts[source_value] = []
        source_distinct_elts[source_value].append(source_index)
    for target_index, target_value in enumerate(target_list):
        if target_value not in target_distinct_elts:
            target_distinct_elts[target_value] = []
        target_distinct_elts[target_value].append(target_index)
    perm = [None] * len(source_list)
    for source_distinct_elt, source_distinct_elt_indices in source_distinct_elts.items():
        target_distinct_elt_indices = target_distinct_elts[source_distinct_elt]
        for (source_distinct_elt_index, target_distinct_elt_index) in zip(source_distinct_elt_indices, target_distinct_elt_indices):
            perm[source_distinct_elt_index] = target_distinct_elt_index
    return Permutation(perm)

def numerical_haar_average_computational_basis_optimized(
    basis_labels_left,
    basis_labels_mid_left,
    basis_labels_mid_right,
    basis_labels_right,
    d_value
):
    """ Compute the numerical (as opposed to symbolic) average of
        U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'})
        over Haar-distributed U using equation given in 1902.08539, theorem 3.1.
        
    Parameters
    ----------
    basis_labels_left: list
        A list specifying i_1, ..., i_t from the definition above.
    basis_labels_mid_left: list
        A list specifying i_1', ..., i_t' from the definition above.
    basis_labels_mid_right: list
        A list specifying j_1', ..., j_t' from the definition above.
    basis_labels_right: list
        A list specifying j_1, ..., j_t from the definition above.
    d_value: int
        The dimension of the Haar-random unitary.
    
    Returns
    -------
    complex
        The Haar average of U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'})
    """
    mapping_permutations_left_right = [
        fixing_perm * find_mapping_permutation(basis_labels_left, basis_labels_right)
        for fixing_perm in generate_fixing_permutations(basis_labels_left)
    ]
    mapping_permutations_mid_left_mid_right = [
        fixing_perm * find_mapping_permutation(basis_labels_mid_left, basis_labels_mid_right)
        for fixing_perm in generate_fixing_permutations(basis_labels_mid_left)
    ]
    return sum([
        weingarten(~mapping_permutation_left_right * mapping_permutation_mid_left_mid_right, d_value)
        for mapping_permutation_left_right in mapping_permutations_left_right
        for mapping_permutation_mid_left_mid_right in mapping_permutations_mid_left_mid_right
    ])
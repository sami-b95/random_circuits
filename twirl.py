import RTNI
import sympy
import tensornetwork as tn

import converters
import random_unitary


def numerical_twirl(tensors, output_edges, input_edges, d_value, n_tensor_factors, U):
    """
    Compute the numerical twirl U^{\otimes t}X U^{\dagger\otimes t} of some matrix X.
    
    Parameters
    ----------
    tensors: list
        The list of tensor network nodes from which the matrix X is constructed. For instance, this
        allows X to be expressed as a (matrix) product state. It must have shape
        (d_value,) * n_tensor_factors.
    output_edges: list
        The list of output edges, i.e the edges connected to the line indices of the matrix.
    input_edges: list
        The list of input edges, i.e the edges connected to the column indices of the matrix.
    d_value: int
        The dimension of each tensor product factor.
    n_tensor_factors: int
        The number of tensor product factors (that is, t in the general definition above).
    U: Node
        The node representing the unitary matrix U to twirl with.
    
    Returns
    -------
    array
        The twirled matrix.
    """
    # Makes copies of tensors.
    tensors_copy, edges_copy = tn.copy(tensors)
    input_edges_copy = [edges_copy[input_edge] for input_edge in input_edges]
    output_edges_copy = [edges_copy[output_edge] for output_edge in output_edges]
    U_copies = [tn.copy([U])[0][U] for _ in range(n_tensor_factors)]
    U_conj_copies = [tn.copy([U], conjugate=True)[0][U] for _ in range(n_tensor_factors)]
    # Connect the U^dagger to the column (or "input") index of the matrix.
    for (input_edge_index, input_edge) in enumerate(input_edges_copy):
        input_edge ^ U_conj_copies[input_edge_index][1]
    # Connect the line (or "output") index of the matrix to the U.
    for (output_edge_index, output_edge) in enumerate(output_edges_copy):
        U_copies[output_edge_index][1] ^ output_edge
    # Contract the resulting tensor network.
    result = tn.contractors.greedy(
        list(tensors_copy.values()) + U_copies + U_conj_copies,
        output_edge_order=(
            [U_copy[0] for U_copy in U_copies] +
            [U_conj_copy[0] for U_conj_copy in U_conj_copies]
        )                
    )
    return result

def haar_statistical_average_twirl(tensors, output_edges, input_edges, d_value, n_tensor_factors, n_samples):
    """
    Estimate the Haar average of the twirl of some matrix X by a unitary using statistical sampling.
    
    Parameters
    ----------
    tensors: list
        The list of tensor network nodes from which the matrix X is constructed. For instance, this
        allows X to be expressed as a (matrix) product state.
    output_edges: list
        The list of output edges, i.e the edges connected to the line indices of the matrix.
    input_edges: list
        The list of input edges, i.e the edges connected to the column indices of the matrix.
    d_value: int
        The dimension of each tensor product factor.
    n_tensor_factors: int
        The number of tensor product factors (that is, t in the general definition above).
    n_samples: int
        The number of samples on which to average.
    
    Returns
    -------
    array
        The statistical estimate of the Haar average of the twirled matrix.
    """
    return sum([
        numerical_twirl(
            tensors,
            output_edges,
            input_edges,
            d_value,
            n_tensor_factors,
            tn.Node(random_unitary.haar_unitary(d_value))
        ).tensor
        for _ in range(n_samples)
    ]) / n_samples

def numerical_haar_average_twirl(tensor, d_value, n_tensor_factors):
    """
    Compute the numerical Haar average of the twirl of some matrix X by a unitary.
    
    Parameters
    ----------
    tensor: Node
        The tensor network node representing the matrix. It must have shape (d_value,) * n_tensor_factors.
    d_value: int
        The dimension of each tensor product factor.
    n_tensor_factors: int
        The number of tensor product factors (that is, t in the general definition above).
    
    Returns
    -------
    array
        The numerical Haar average
    """
    # Make copies of tensors.
    tensors_copy, edges_copy = tn.copy([tensor])
    tensor_copy = tensors_copy[tensor]
    output_edges_copy = [edges_copy[output_edge] for output_edge in tensor.edges[:int(len(tensor.edges) / 2)]]
    input_edges_copy = [edges_copy[input_edge] for input_edge in tensor.edges[int(len(tensor.edges) / 2):]]
    # Generate RTNI tensor network graph.
    graph = []
    graph.extend([
        [["X", 1, "in", input_edge_index + 1], ["U*", input_edge_index + 1, "out", 1]]
        for input_edge_index in range(len(input_edges_copy))
    ])
    graph.extend([
        [["U", output_edge_index + 1, "in", 1], ["X", 1, "out", output_edge_index + 1]]
        for output_edge_index in range(len(output_edges_copy))
    ])
    # Symbolically Haar-integrate this tensor network.
    d_symbol = sympy.symbols("d")
    symbolic_average = RTNI.integrateHaarUnitary([graph, 1], ["U", [d_symbol], [d_symbol], d_symbol])
    # Convert the symbolic RTNI average to a numerical TensorNetwork one.
    numerical_average = converters.rtni_to_tn(
        symbolic_average,
        {
            "X": tensor
        },
        d_symbol=d_symbol,
        d_value=d_value,
        n_tensor_factors=2
    )
    return numerical_average
#!/lustre/home/ucapsjg/.conda/envs/quantum/bin/python3.8

from argparse import ArgumentParser
from functools import partial
from importlib import reload
import json
import os
import pickle
import sys
import time

sys.path.append("/home/ucapsjg/random_circuits")
import benchmarks
import unitary_designs


parser = ArgumentParser(description="Run job for a given set of parameters in a parameter grid")
parser.add_argument("param_grid_file", type=str, help="The file containing the parameter grid")
parser.add_argument("param_set_id", type=int, help="The identifier of the parameter set to use")
parser.add_argument("--datadir", type=str, help="Data directory")
args = parser.parse_args()

# Load parameter set from parameter grid

with open(args.param_grid_file, "r") as f:
    param_grid = json.load(f)
param_set = param_grid[args.param_set_id]

# Set circuit sampler

if param_set["design"] == "1d_parallel":
    circuit_sampler = partial(
        unitary_designs.pseudorandom_1d_parallel_circuit,
        n_qubits=param_set["n_qubits"],
        length=param_set["length"]
    )
elif param_set["design"] == "2d_parallel":
    circuit_sampler = partial(
        unitary_designs.pseudorandom_parallel_circuit,
        D=2,
        n_qubits_per_dimension=param_set["n_qubits_per_dimension"],
        s=param_set["s"],
        c=param_set["c"]
    )
elif param_set["design"] == "2d_parallel_alternative":
    circuit_sampler = partial(
        unitary_designs.pseudorandom_2d_parallel_circuit,
        n_qubits_rows=param_set["n_qubits_rows"],
        n_qubits_cols=param_set["n_qubits_cols"],
        s=param_set["s"],
        c=param_set["c"],
        two_qubit_gate_choice=param_set.get("two_qubit_gate_choice", "continuous")
    )
elif param_set["design"] == "sycamore_18":
    circuit_sampler = partial(unitary_designs.sycamore_18, n_cycles=param_set["n_cycles"])
else:
    raise ValueError(f"Unrecognized design type {param_set['design']}")

# Run sampling experiment

output_prefix = f"{int(time.time() * 1000000)}_{os.getpid()}"

result = {
    "design": param_set["design"],
    "n_samples": param_set["n_samples"],
    "benchmark": param_set["benchmark"]
}
if param_set["design"] == "1d_parallel":
    result.update({
        "n_qubits": param_set["n_qubits"],
        "length": param_set["length"]
    })
elif param_set["design"] == "2d_parallel":
    result.update({
        "n_qubits_per_dimension": param_set["n_qubits_per_dimension"],
        "s": param_set["s"],
        "c": param_set["c"]
    })
elif param_set["design"] == "2d_parallel_alternative":
    result.update({
        "n_qubits_rows": param_set["n_qubits_rows"],
        "n_qubits_cols": param_set["n_qubits_cols"],
        "s": param_set["s"],
        "c": param_set["c"],
        "two_qubit_gate_choice": param_set.get("two_qubit_gate_choice", "continuous")
    })
elif param_set["design"] == "sycamore_18":
    result.update({
        "n_cycles": param_set["n_cycles"]
    })

if param_set["benchmark"] == "random_coefficient_benchmark":
    deviation, \
    error,  \
    basis_labels_left, \
    basis_labels_mid_left, \
    basis_labels_mid_right, \
    basis_labels_right, \
    samples = benchmarks.twirl_matrix_coefficient_benchmark(
        circuit_sampler=circuit_sampler,
        n_tensor_factors=param_set["n_tensor_factors"],
        n_circuits=param_set["n_samples"]
    )
    result.update({
        "n_tensor_factors": param_set["n_tensor_factors"],
        "deviation": deviation,
        "error": error,
        "basis_labels_left": basis_labels_left,
        "basis_labels_mid_left": basis_labels_mid_left,
        "basis_labels_mid_right": basis_labels_mid_right,
        "basis_labels_right": basis_labels_right
    })
    full_result = dict(result)
    full_result["samples"] = samples
elif param_set["benchmark"] == "xeb_benchmark":
    xeb_mean, xeb_std, samples = benchmarks.xeb_benchmark(circuit_sampler, param_set["n_samples"], moment=param_set.get("moment", 1))
    result.update({
        "moment": param_set.get("moment", 1),
        "xeb_mean": xeb_mean,
        "xeb_std": xeb_std
    })
    full_result = dict(result)
    full_result["samples"] = samples

with open(os.path.join(args.datadir, f"{output_prefix}_info.pickle"), "wb") as f:
    pickle.dump(result, f)
with open(os.path.join(args.datadir, f"{output_prefix}_data.pickle"), "wb") as f:
    pickle.dump(full_result, f)
 

from argparse import ArgumentParser
from functools import partial
from importlib import reload
import json
import os
import pickle
import time

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
    circuit_sampler = partial(unitary_designs.pseudorandom_1d_parallel_circuit, n_qubits=param_set["n_qubits"], length=param_set["length"])
else:
    raise ValueError(f"Unrecognized design type {param_set['design']}")

# Run sampling experiment

output_prefix = f"{int(time.time() * 1000000)}_{os.getpid()}"

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
    with open(os.path.join(args.datadir, f"{output_prefix}_info.pickle"), "wb") as f:
        pickle.dump({
            "deviation": deviation,
            "error": error,
            "basis_labels_left": basis_labels_left,
            "basis_labels_mid_left": basis_labels_mid_left,
            "basis_labels_mid_right": basis_labels_mid_right,
            "basis_labels_right": basis_labels_right
        }, f)
    with open(os.path.join(args.datadir, f"{output_prefix}_data.pickle"), "wb") as f:
        pickle.dump({
            "deviation": deviation,
            "error": error,
            "basis_labels_left": basis_labels_left,
            "basis_labels_mid_left": basis_labels_mid_left,
            "basis_labels_mid_right": basis_labels_mid_right,
            "basis_labels_right": basis_labels_right,
            "samples": samples
        }, f)
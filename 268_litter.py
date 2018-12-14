# model file: ../example-models/bugs_examples/vol1/litter/litter.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'G' in data, 'variable not found in data: key=G'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'r' in data, 'variable not found in data: key=r'
    assert 'n' in data, 'variable not found in data: key=n'
    # initialize data
    G = data["G"]
    N = data["N"]
    r = data["r"]
    n = data["n"]
    check_constraints(G, low=0, dims=[1])
    check_constraints(N, low=0, dims=[1])
    check_constraints(r, low=0, dims=[G, N])
    check_constraints(n, low=0, dims=[G, N])

def init_params(data, params):
    # initialize data
    G = data["G"]
    N = data["N"]
    r = data["r"]
    n = data["n"]
    # assign init values for parameters
    params["p"] = init_matrix("p", low=0, high=1, dims=(G, N)) # matrix
    params["mu"] = init_vector("mu", low=0, high=1, dims=(G)) # vector
    params["a_plus_b"] = init_vector("a_plus_b", low=0.10000000000000001, dims=(G)) # vector

def model(data, params):
    # initialize data
    G = data["G"]
    N = data["N"]
    r = data["r"]
    n = data["n"]
    # INIT parameters
    p = params["p"]
    mu = params["mu"]
    a_plus_b = params["a_plus_b"]
    # initialize transformed parameters
    a = init_vector("a", dims=(G)) # vector
    b = init_vector("b", dims=(G)) # vector
    a = _pyro_assign(a, _call_func("elt_multiply", [mu,a_plus_b]))
    b = _pyro_assign(b, _call_func("elt_multiply", [_call_func("subtract", [1,mu]),a_plus_b]))
    # model block

    a_plus_b =  _pyro_sample(a_plus_b, "a_plus_b", "pareto", [0.10000000000000001, 1.5])
    for g in range(1, to_int(G) + 1):

        for i in range(1, to_int(N) + 1):

            p[g - 1][i - 1] =  _pyro_sample(_index_select(_index_select(p, g - 1) , i - 1) , "p[%d][%d]" % (to_int(g-1),to_int(i-1)), "beta", [_index_select(a, g - 1) , _index_select(b, g - 1) ])
            r[g - 1][i - 1] =  _pyro_sample(_index_select(_index_select(r, g - 1) , i - 1) , "r[%d][%d]" % (to_int(g-1),to_int(i-1)), "binomial", [_index_select(_index_select(n, g - 1) , i - 1) , _index_select(_index_select(p, g - 1) , i - 1) ], obs=_index_select(_index_select(r, g - 1) , i - 1) )

# model file: ../example-models/ARM/Ch.6/wells_probit.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'dist' in data, 'variable not found in data: key=dist'
    assert 'switc' in data, 'variable not found in data: key=switc'
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(dist, dims=[N])
    check_constraints(switc, low=0, high=1, dims=[N])

def transformed_data(data):
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]
    dist100 = init_vector("dist100", dims=(N)) # vector
    dist100 = _pyro_assign(dist100, _call_func("divide", [dist,100.0]))
    data["dist100"] = dist100

def init_params(data, params):
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]
    # initialize transformed data
    dist100 = data["dist100"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector

def model(data, params):
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]
    # initialize transformed data
    dist100 = data["dist100"]
    # INIT parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    for n in range(1, to_int(N) + 1):
        switc[n - 1] =  _pyro_sample(_index_select(switc, n - 1) , "switc[%d]" % (to_int(n-1)), "bernoulli", [_call_func("Phi", [(_index_select(beta, 1 - 1)  + (_index_select(beta, 2 - 1)  * _index_select(dist100, n - 1) ))])], obs=_index_select(switc, n - 1) )

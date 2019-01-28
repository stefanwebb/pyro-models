# model file: ../example-models/ARM/Ch.4/mesquite_va.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'weight' in data, 'variable not found in data: key=weight'
    assert 'diam1' in data, 'variable not found in data: key=diam1'
    assert 'diam2' in data, 'variable not found in data: key=diam2'
    assert 'canopy_height' in data, 'variable not found in data: key=canopy_height'
    assert 'group' in data, 'variable not found in data: key=group'
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    group = data["group"]

def transformed_data(data):
    # initialize data
    log = torch.log
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    group = data["group"]
    log_weight = log(weight)
    log_canopy_volume = log(diam1 * diam2 * canopy_height)
    log_canopy_area   = log(diam1 * diam2)
    data["log_weight"] = log_weight
    data["log_canopy_volume"] = log_canopy_volume
    data["log_canopy_area"] = log_canopy_area

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    group = data["group"]
    # initialize transformed data
    log_weight = data["log_weight"]
    log_canopy_volume = data["log_canopy_volume"]
    log_canopy_area = data["log_canopy_area"]

    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    pyro.sample('log_weight', dist.Normal(beta[0] + beta[1] * log_canopy_volume + beta[2] * log_canopy_area
                                          + beta[3] * group, sigma),
                obs=weight)

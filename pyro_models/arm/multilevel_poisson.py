# model file: example-models/ARM/Ch.17/multilevel_poisson_17.5.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_eth' in data, 'variable not found in data: key=n_eth'
    assert 'n_precint' in data, 'variable not found in data: key=n_precint'
    assert 'eth' in data, 'variable not found in data: key=eth'
    assert 'precint' in data, 'variable not found in data: key=precint'
    assert 'offeset' in data, 'variable not found in data: key=offeset'
    assert 'stops' in data, 'variable not found in data: key=stops'

def init_params(data):
    params = {}
    params["sigma_eth"] = pyro.sample("sigma", dist.HalfCauchy(2.5))
    params["sigma_epsilon"] = pyro.sample("sigma_epsilon", dist.HalfCauchy(2.5))
    params["sigma_precint"] = pyro.sample("sigma_precint", dist.HalfCauchy(2.5))
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_eth = data["n_eth"]
    n_precint = data["n_precint"]
    eth = data["eth"].long() - 1
    precint = data["precint"].long() - 1
    offeset = data["offeset"]
    stops = data["stops"]

    #age = data["age"].long() - 1
    #edu = data["edu"].long() - 1
    
    # init parameters
    sigma_eth = params["sigma_eth"]
    sigma_epsilon = params["sigma_epsilon"]
    sigma_precint = params["sigma_precint"]

    mu = pyro.sample("mu", dist.Normal(0., 100.))

    with pyro.plate("n_eth", n_eth):
        b_eth = pyro.sample("b_eth", dist.Normal(0., sigma_eth))
    #b_eth_adj = b_eth - b_eth.mean()

    with pyro.plate("n_precint", n_precint):
        b_precint = pyro.sample("b_precint", dist.Normal(0., sigma_precint))
    #b_precint_adj = b_precint - b_eth.mean()

    #mu_adj = mu + b_eth.mean() + b_precint.mean()
    with pyro.plate("data", N):
        epsilon = pyro.sample("epsilon", dist.Normal(0., sigma_epsilon))
        lograte = offeset + mu + b_eth[eth] + b_precint[precint] + epsilon        
        pyro.sample('y', dist.Poisson(lograte.exp()), obs=stops)

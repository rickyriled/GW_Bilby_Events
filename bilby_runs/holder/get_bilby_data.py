import bilby
import json
import numpy as np

def get_bilby_data(file_name):
    data=bilby.result.read_in_result(file_name)
    run_dat=data.posterior

    log_ev=data.log_evidence
    
    log_prior=list(run_dat['log_prior'])
    prior=np.exp(log_prior)

    log_likelihood=list(run_dat['log_likelihood'])
    likelihood=np.exp(log_likelihood)

    keys=run_dat.keys()[0:15]
    posterior=[]

    for key in keys:
        posterior.append(list(run_dat[key]))

    result_data=[posterior, prior, likelihood, log_ev, keys]

    return result_data


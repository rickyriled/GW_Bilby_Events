import json
import pickle
import bilby
import numpy as np

def get_results(path):
    values=bilby.result.read_in_result(path)
    return values

def get_ids(vals):
    injected=vals.meta_data['likelihood']['interferometers']['H1']['parameters']
    print("ra: ",injected['ra'],"dec: ",injected['dec']," d: ",injected['luminosity_distance'])
    print("")
    print("")
    print("chirp mass", injected['chirp_mass'])
    print("mass_ratio", injected['mass_ratio'])

    priors=vals.priors
    print("")
    print("")
    print("ra:", priors['ra'])
    print("")
    print("dec:", priors['dec'])
    print("")
    print("d:", priors['luminosity_distance'])
    print("")
    print("chirp mass", priors['chirp_mass'])
    print("mass_ratio", priors['mass_ratio'])

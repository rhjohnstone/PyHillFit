import pymc3 as pm
import doseresponse as dr
import numpy as np

# Uniform Hill prior
loghill_lower = 0.
loghill_upper = 10.

# logHill Logistic prior
loghill_mu = 0.
loghill_s = 0.1

# Shifted exponential pIC50 (EC50?) prior
pic50_lower = -3
pic50_rate = 0.2

# Shifted Gamma obseravation noise s.d. prior
sigma_lower = 1e-3
sigma_shape = 5.
sigma_mode = 6.
sigma_rate = (sigma_shape-1) / sigma_mode

# Uniform saturation prior
sat_lower = 0.
sat_upper = 2.

n_models = 3


def f_pic50(p):
    return p + pic50_lower

def f_sigma(s):
    return s + sigma_lower

def f_hill(loghill):
    return np.exp(loghill)

def f_sat(sat):
    return sat


def expt_model(model_number, concs, responses):

    with pm.Model() as model:
    
        if model_number == 1:
            hill = 1
            sat = 0
            #keep = ["pIC50", "$\sigma$"]
            fs = {"pIC50": f_pic50, "$\sigma$": f_sigma}
        elif model_number == 2:
            #hill = pm.Uniform('Hill', lower=hill_lower, upper=hill_upper)
            loghill = pm.Logistic("Hill", mu=loghill_mu, s=loghill_s)
            #hill = pm.Deterministic("Hill", pm.math.exp(loghill))
            hill = pm.math.exp(loghill)
            sat = 0
            #keep = ["pIC50", "$\sigma$", "Hill"]
            fs = {"pIC50": f_pic50, "$\sigma$": f_sigma, "Hill": f_hill}
        elif model_number == 3:
            #hill = pm.Uniform('Hill', lower=hill_lower, upper=hill_upper)
            loghill = pm.Logistic("Hill", mu=loghill_mu, s=loghill_s)
            #hill = pm.Deterministic("Hill", pm.math.exp(loghill))
            hill = pm.math.exp(loghill)
            sat = pm.Uniform("Saturation", lower=sat_lower, upper=sat_upper)
            #keep = ["pIC50", "$\sigma$", "Hill", "Saturation"]
            fs = {"pIC50": f_pic50, "$\sigma$": f_sigma, "Hill": f_hill,
                  "Saturation": f_sat}
        
        # pIC50 value
        p = pm.Exponential("pIC50", lam=pic50_rate)
        #p_shift = pm.Deterministic("pIC50", p + pic50_lower)
        p_shift = f_pic50(p)
        
        # Noise standard deviation sigma
        s = pm.Gamma("$\sigma$", alpha=sigma_shape, beta=sigma_rate)
        #s_shift = pm.Deterministic("$\sigma$", s + sigma_lower)
        s_shift = f_sigma(s)
        
        # Actual data model
        pred = dr.per_cent_block(concs, hill, p_shift, sat)
        obs = pm.Normal("y", mu=pred, sigma=s_shift, observed=responses)
        
    return model, fs

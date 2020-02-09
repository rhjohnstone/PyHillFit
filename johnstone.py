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
            fs = {"pIC50": f_pic50, "$\sigma$": f_sigma}
        elif model_number == 2:
            loghill = pm.Logistic("Hill", mu=loghill_mu, s=loghill_s)
            hill = pm.math.exp(loghill)
            sat = 0
            fs = {"pIC50": f_pic50, "$\sigma$": f_sigma, "Hill": f_hill}
        elif model_number == 3:
            loghill = pm.Logistic("Hill", mu=loghill_mu, s=loghill_s)
            hill = pm.math.exp(loghill)
            sat = pm.Uniform("Saturation", lower=sat_lower, upper=sat_upper)
            fs = {"pIC50": f_pic50, "$\sigma$": f_sigma, "Hill": f_hill,
                  "Saturation": f_sat}
        
        # pIC50 value
        p = pm.Exponential("pIC50", lam=pic50_rate)
        p_shift = f_pic50(p)
        
        # Noise standard deviation sigma
        s = pm.Gamma("$\sigma$", alpha=sigma_shape, beta=sigma_rate)
        s_shift = f_sigma(s)
        
        # Actual data model
        pred = dr.per_cent_block(concs, hill, p_shift, sat)
        obs = pm.Normal("y", mu=pred, sigma=s_shift, observed=responses)
        
    
    def dr_model(x, trace, sample, model_number):
        """For plotting sample dose-response curves after the inference."""
        pic50 = trace["pIC50"][sample]
        if model_number == 1:
            hill = 1
            saturation = 0
        elif model_number == 2:
            hill = trace["Hill"][sample]
            saturation = 0
        elif model_number == 3:
            hill = trace["Hill"][sample]
            saturation = trace["Saturation"][sample]
        return dr.per_cent_block(x, hill, pic50, saturation)
        
    return model, fs, dr_model

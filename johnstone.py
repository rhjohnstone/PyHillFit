import pymc3 as pm
import doseresponse as dr

# Uniform Hill prior
hill_lower = 0.
hill_upper = 10.

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

def model(model_number, concs, responses):

    with pm.Model() as model:
    
        # Hill coefficient
        if model_number == 1:
            hill = 1
            sat = 0
        elif model_number == 2:
            hill = pm.Uniform('Hill', lower=hill_lower, upper=hill_upper)
            sat = 0
        elif model_number == 3:
            hill = pm.Uniform('Hill', lower=hill_lower, upper=hill_upper)
            sat = pm.Uniform('saturation', lower=sat_lower, upper=sat_upper)
        
        # pIC50 value
        p = pm.Exponential('p', lam=pic50_rate)
        p_shift = pm.Deterministic("pIC50", p + pic50_lower)
        
        # Noise standard deviation sigma
        s = pm.Gamma('s', alpha=sigma_shape, beta=sigma_rate)
        s_shift = pm.Deterministic("$\sigma$", s + sigma_lower)
        
        # Actual data model
        pred = dr.per_cent_block(concs, hill, p_shift, sat)
        obs = pm.Normal('y', mu=pred, sigma=s_shift, observed=responses)
        
    remove = ["p", "s"]
        
    return model, remove

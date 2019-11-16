import pymc3 as pm
import doseresponse as dr

hill_lower = 0.
hill_upper = 10.
pic50_lower = -3
pic50_rate = 0.2
sigma_lower = 1e-3
sigma_shape = 5.
sigma_mode = 6.
sigma_rate = (sigma_shape-1) / sigma_mode

n_models = 2

def model(model_number, iterations, tune, concs, data):

    with pm.Model():
    
        # Hill coefficient
        if model_number == 1:
            h = 1
        elif model_number == 2:
            h = pm.Uniform('Hill', lower=hill_lower, upper=hill_upper)
        
        # pIC50 value
        p = pm.Exponential('p', lam=pic50_rate)
        p_shift = pm.Deterministic("pIC50", p + pic50_lower)
        
        # Noise standard deviation sigma
        s = pm.Gamma('s', alpha=sigma_shape, beta=sigma_rate)
        s_shift = pm.Deterministic("$\sigma$", s + sigma_lower)
        
        # Actual data model
        pred = dr.hill_curve(concs, h, p_shift)
        obs = pm.Normal('y', mu=pred, sigma=s_shift, observed=data)
        
        # Do the inference!
        trace = pm.sample(iterations, tune=tune)
    trace.remove_values("p")
    trace.remove_values("s")
    return trace

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

n_models = 3

def model(model_number, iterations, tune, concs, responses):

    with pm.Model():
    
        # Hill coefficient
        if model_number == 1:
            h = 1
            sat = 100
        elif model_number == 2:
            h = pm.Uniform('Hill', lower=hill_lower, upper=hill_upper)
            sat = 100
        elif model_number == 3:
            h = pm.Uniform('Hill', lower=hill_lower, upper=hill_upper)
            sat = pm.Uniform('saturation', lower=0., upper=100.)
        
        # pIC50 value
        p = pm.Exponential('p', lam=pic50_rate)
        p_shift = pm.Deterministic("pIC50", p + pic50_lower)
        
        # Noise standard deviation sigma
        s = pm.Gamma('s', alpha=sigma_shape, beta=sigma_rate)
        s_shift = pm.Deterministic("$\sigma$", s + sigma_lower)
        
        # Actual data model
        pred = dr.saturated_hil_curve(concs, h, p_shift, sat)
        obs = pm.Normal('y', mu=pred, sigma=s_shift, observed=responses)
        
        # Do the inference!
        trace = pm.sample(iterations, tune=tune)
    trace.remove_values("p")
    trace.remove_values("s")
    return trace

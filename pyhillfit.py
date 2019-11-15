import pymc3 as pm
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import doseresponse as dr

f = "data/crumb.csv"
data = dr.Data(f)
channels_drugs = data.select_channel_drug()
for channel, drug in channels_drugs:
    concs, data = data.load_data(channel, drug)

    model = 2
    hill_lower = 0.
    hill_upper = 10.
    pic50_lower = -3
    pic50_rate = 0.2
    sigma_lower = 1e-3
    sigma_shape = 5.
    sigma_mode = 6.
    sigma_rate = (sigma_shape-1) / sigma_mode

    iterations = 10000
    tune = iterations
    with pm.Model():
        if model == 1:
            h = 1
        elif model == 2:
            h = pm.Uniform('h', lower=hill_lower, upper=hill_upper)
        
        p = pm.Exponential('p', lam=pic50_rate)
        p_shift = pm.Deterministic("p_shift", p + pic50_lower)
        s = pm.Gamma('s', alpha=sigma_shape, beta=sigma_rate)
        s_shift = pm.Deterministic("s_shift", s + sigma_lower)
        pred = dr.hill_curve(concs, h, p_shift)
        obs = pm.Normal('y', mu=pred, sigma=s_shift, observed=data)
        trace = pm.sample(iterations, tune=tune)
    trace.remove_values("p")
    trace.remove_values("s")

    tp = pm.traceplot(trace)
    fig = plt.gcf() # to get the current figure...
    fig.savefig(f"{channel_{drug}.png") # and save it directly
    

import pymc3 as pm
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def hill_curve(x, hill, pic50):
    ic50 = 10**(6-pic50)
    return 100 * ( 1 - 1 / (1 + (x/ic50)**hill) )


fix_hill = False

concs = np.array([0.0008, 0.0008, 0.0008, 0.08, 0.08, 0.08,
                  0.8, 0.8, 0.8, 8, 8, 8])
data = np.array([0, 0, 0, 12, 17.8, 14.8, 
                 37.5, 62.7, 60.2, 66.5, 80.2, 73.3])

hill_lower = 0.
hill_upper = 10.
pic50_lower = -3
sigma_lower = 1e-3
sigma_shape = 5.
sigma_mode = 6.
sigma_rate = (sigma_shape-1) / sigma_mode

iterations = 2000
tune = iterations//5
with pm.Model():
    if fix_hill:
        h = 1
    else:
        h = pm.Uniform('h', lower=hill_lower, upper=hill_upper)
    p = pm.Exponential('p', lam=0.2)
    p_shift = pm.Deterministic("p_shift", p + pic50_lower)
    s = pm.Gamma('s', alpha=sigma_shape, beta=sigma_rate)
    s_shift = pm.Deterministic("s_shift", s + sigma_lower)
    pred = hill_curve(concs, h, p_shift)
    obs = pm.Normal('y', mu=pred, sigma=s_shift, observed=data)
    trace = pm.sample(iterations, tune=tune)
for param in ["p", "s"]:
    trace.remove_values(param)

tp = pm.traceplot(trace)
fig = plt.gcf() # to get the current figure...
fig.savefig("hill_exp_gamma.png") # and save it directly
    

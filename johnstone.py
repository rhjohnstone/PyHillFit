import pymc3 as pm
import doseresponse as dr
import numpy as np
from functools import partial


def gamma_distn_rate(shape, mode, lower):
    return (shape - 1) / (mode - lower)


n_models = 5

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

# Hyperparameters for hierarchical model
# We use experimental results from https://doi.org/10.1016/j.vascn.2013.04.007,
# along with some empirically-chosen values, to inform our prior distributions
elkins_pic50_mus = [5.235, 5.765, 6.060, 5.315, 5.571, 7.378, 7.248, 5.249, 
                    6.408, 5.625, 7.321, 6.852, 6.169, 6.217, 5.927, 7.414,
                    4.860]
mu_mode = np.mean(elkins_pic50_mus)
mu_lower = -4
mu_shape = 7.5
mu_rate = gamma_distn_rate(mu_shape, mu_mode, mu_lower)

elkins_pic50_ss = [0.0760, 0.1388, 0.1459, 0.2044, 0.1597, 0.2216, 0.1856,
                   0.1560, 0.1034, 0.1033, 0.1914, 0.1498, 0.1464, 0.1053,
                   0.1342,0.1808,0.0860]
s_mode = np.mean(elkins_pic50_ss)
s_lower = 0.01
s_shape = 2.5
s_rate = gamma_distn_rate(s_shape, s_mode, s_lower)

elkins_hill_alphas = [1.188, 1.744, 1.530, 0.930, 0.605, 1.325, 1.179, 0.979,
                      1.790, 1.708, 1.586, 1.469, 1.429, 1.127, 1.011, 1.318,
                      1.063]
alpha_mode = np.mean(elkins_hill_alphas)
alpha_lower = 0
alpha_shape = 5
alpha_rate = gamma_distn_rate(alpha_shape, alpha_mode, alpha_lower)

elkins_hill_betas_inv = [0.0835, 0.1983, 0.2089, 0.1529, 0.1206, 0.2386,
                         0.2213, 0.2263, 0.1784, 0.1544, 0.2486, 0.2031,
                         0.2025, 0.1510, 0.1837, 0.1677, 0.0862]
elkins_hill_betas = 1 / np.array(elkins_hill_betas_inv)
beta_mode = np.mean(elkins_hill_betas)
beta_lower = 2
beta_shape = 2.5
beta_rate = gamma_distn_rate(beta_shape, beta_mode, beta_lower)


def expt_model(model_number, concs, responses, expt_labels):
    """
    Define all statistical models in here. Take care of what you want to be
    consistent across all models (e.g. an observation noise model, data
    likelihood model, etc.).
    
    You can use numpy.arrays of data, but not numpy operations when
    transforming variables, etc., because PyMC3 uses Theano tensors.
    
    Because of how PyMC3 deals with transformation of variables, I chose to use
    pm.Deterministic, which stores both the original and transformed variable
    samples. This is slightly annoying, as it doubles the memory used. For a
    large number of parameters or sampling iterations, this could be a problem.
    Given the relatively small number of parameters in the models we are
    considering here, I don't think it is a problem. I keep a list of the
    original parameters I don't need later, and remove them after sampling is
    finished. There are some solutions that don't require storing both sets of
    parameters, but they are more difficult to deal with after the sampling is
    finished. If we start experimenting with models with many more parameters,
    we may have to revisit these alternatives. But I believe the current
    solution has the best balance for now.
    """
    n_expts = expt_labels.max() + 1
    with pm.Model() as model:
        
        # Noise standard deviation sigma, shared by all models.
        sigma = pm.Gamma("xsigma", alpha=sigma_shape, beta=sigma_rate)
        sigma_shift = pm.Deterministic(r"$\sigma$", sigma + sigma_lower)
    
        if model_number == 1:  # Single-level
            pic50 = pm.Exponential("xpIC50", lam=pic50_rate)
            pic50_shift = pm.Deterministic("pIC50", pic50 + pic50_lower)
            hill = 1
            sat = 0
            remove = ["xsigma", "xpIC50"]
        elif model_number == 2:  # Single-level
            pic50 = pm.Exponential("xpIC50", lam=pic50_rate)
            pic50_shift = pm.Deterministic("pIC50", pic50 + pic50_lower)
            loghill = pm.Logistic("xlogHill", mu=loghill_mu, s=loghill_s)
            hill = pm.Deterministic("Hill", pm.math.exp(loghill))
            sat = 0
            remove = ["xsigma", "xpIC50", "xlogHill"]
        elif model_number == 3:  # Single-level
            pic50 = pm.Exponential("xpIC50", lam=pic50_rate)
            pic50_shift = pm.Deterministic("pIC50", pic50 + pic50_lower)
            loghill = pm.Logistic("xlogHill", mu=loghill_mu, s=loghill_s)
            hill = pm.Deterministic("Hill", pm.math.exp(loghill))
            sat = pm.Uniform("Saturation", lower=sat_lower, upper=sat_upper)
            remove = ["xsigma", "xpIC50", "xlogHill"]
        elif model_number == 4:  # Hierarchical
            hill = 1
            sat = 0
            mu = pm.Gamma("xmu", alpha=mu_shape, beta=mu_rate)
            mu_shift = pm.Deterministic(r"$\mu$", mu + mu_lower)
            s = pm.Gamma("xs", alpha=s_shape, beta=s_rate)
            s_shift = pm.Deterministic("s", s + s_lower)
            pic50 = pm.Logistic("pIC50", mu=mu_shift, s=s_shift, shape=n_expts)
            pic50_shift = pic50[expt_labels]
            remove = ["xsigma", "xmu", "xs"]
        elif model_number == 5:  # Hierarchical
            sat = 0
            mu = pm.Gamma("xmu", alpha=mu_shape, beta=mu_rate)
            mu_shift = pm.Deterministic(r"$\mu$", mu + mu_lower)
            s = pm.Gamma("xs", alpha=s_shape, beta=s_rate)
            s_shift = pm.Deterministic("s", s + s_lower)
            pic50 = pm.Logistic("pIC50", mu=mu_shift, s=s_shift, shape=n_expts)
            pic50_shift = pic50[expt_labels]
            
            alpha = pm.Gamma(r"$\alpha$", alpha=alpha_shape, beta=alpha_rate)
            beta = pm.Gamma("xbeta", alpha=beta_shape, beta=beta_rate)
            beta_shift = pm.Deterministic(r"$\beta$", beta + beta_lower)
            loghill = pm.Logistic("xlogHill", mu=alpha, s=beta_shift,
                                  shape=n_expts)
            xhill = pm.Deterministic("Hill", pm.math.exp(loghill))
            hill = xhill[expt_labels]
            remove = ["xsigma", "xmu", "xs", "xbeta", "xlogHill"]
        elif model_number == 6:  # Hierarchical
            sat = pm.Uniform("Saturation", lower=sat_lower, upper=sat_upper)
            mu = pm.Gamma("xmu", alpha=mu_shape, beta=mu_rate)
            mu_shift = pm.Deterministic(r"$\mu$", mu + mu_lower)
            s = pm.Gamma("xs", alpha=s_shape, beta=s_rate)
            s_shift = pm.Deterministic("s", s + s_lower)
            pic50 = pm.Logistic("pIC50", mu=mu_shift, s=s_shift, shape=n_expts)
            pic50_shift = pic50[expt_labels]
            
            alpha = pm.Gamma(r"$\alpha$", alpha=alpha_shape, beta=alpha_rate)
            beta = pm.Gamma("xbeta", alpha=beta_shape, beta=beta_rate)
            beta_shift = pm.Deterministic(r"$\beta$", beta + beta_lower)
            loghill = pm.Logistic("xlogHill", mu=alpha, s=beta_shift,
                                  shape=n_expts)
            xhill = pm.Deterministic("Hill", pm.math.exp(loghill))
            hill = xhill[expt_labels]
            remove = ["xsigma", "xmu", "xs", "xbeta", "xlogHill"]
        
        # Data likelihood, shared by all models.
        pred = dr.per_cent_block(concs, hill, pic50_shift, sat)
        obs = pm.Normal("y", mu=pred, sigma=sigma_shift, observed=responses)
        
    def get_sample(model_number, trace, sample):
        """
        When we want to plot sample dose-response curves, or make predictions
        later, we need to subsample our chains. But the existence of certain
        parameter chains depends on our model definition, so we need to tell it
        when to look for a parameter or when to give it a default value. We
        could use a try/except block, but we know there will be a lot of fails,
        so we'll just check which model it is.
        """
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
        return pic50, hill, saturation
        
    return model, remove, get_sample

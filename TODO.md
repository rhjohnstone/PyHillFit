# To-do

* Posterior Predictive, possibly sampling along with posterior: https://stats.stackexchange.com/questions/169223/how-to-generate-the-posterior-predictive-distribution-for-hierarchal-model-in-py
  * Actually, might be better to use pymc3.sampling.sample_posterior_predictive from https://docs.pymc.io/api/inference.html
  * Also, check https://docs.pymc.io/notebooks/posterior_predictive.html
* Plotting sample dose-response curves, from both Post. and Post. Pred.?
* Any other MCMC analysis figures we want to include
* Summary statistics / compression? Possibly impractical to save loads of samples from each chain, etc.
* Add option to model as "channel block" or "conductance scale", and choose the appropriate model/function
* Possibly move some stuff into separate files, e.g. plotting
* pm.model_to_graphviz(model)


# PyHillFit

Bayesian inference on dose-response (concentration-effect) data.

 - Quantify and propagate uncertainty in dose-response data for use in cardiac action potential modelling.
 - Compute Bayes Factor to compare statistical models.
 
 (This is intended to replace our [earlier version](https://github.com/mirams/pyhillfit).)

## Quickstart

First install dependencies:

`python -m pip install --user -r requirements.txt`

Run default experiment (`johnstone`) on default data (`crumb`):

`python pyhillfit.py`

and select a channel and drug from the menu when prompted. I like hERG (7) and Amiodarone (1).

Output figures will be generated in an `output` folder organised as:

output > data name > channel > drug > experiment name.

This was chosen to make it easier to compare two different statistical models of the same data, but I am open to suggestions on how to improve this.

## New experiments

By "experiment", we mean "collection of statistical models". Users can define their own models (and hence experiments) in the same format as in `johnstone.py`, and running

`python pyhillfit.py --experiment $new_experiment.py`

Due to how PyMC3 saves parameters for single-level and hierarchical models, there is currently no general method for plotting sample dose-response curves after performing the inference, so that also has to be defined in the experiment file.

Examples of new experiments include changing hyperparameter values in `johnstone.py` (we recommend, rather, making a copy and changing those values!), or choosing completely different prior distributions over different parameters.

Models can be added or removed, but be sure to change `n_models` to reflect these changes (currently line 104 in `johnstone.py`).

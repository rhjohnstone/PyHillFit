import doseresponse as dr
import argparse
import os
import sys
import itertools as it
from time import time

# TODO - basic optimization
#      - add documentation
#      - double-check requirements.txt

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="dose-response data file",
                    default=os.path.join("data", "crumb.csv"))
parser.add_argument("--model", type=str, help="probabilistic model definition",
                    default="johnstone")
parser.add_argument("--all", action="store_true",
                    help="run all channel/drug combinations")
parser.add_argument("--bf", action="store_true",
                    help="compute Bayes Factors through SMC")
parser.add_argument("--iterations", type=int, default=10000,
                    help="number of MCMC samples (plus the same number "
                         "again for tuning")
args = parser.parse_args()

data = dr.Data(args.input)
channels_drugs = data.select_channel_drug(args.all)


# importing here so we don't have to wait before choosing channel/drug
import pymc3 as pm
import arviz as az
#az.style.use("arviz-darkgrid")
import matplotlib.pyplot as plt
#plt.style.use("ggplot")
from importlib import import_module
import numpy as np
import numpy.random as npr
import plots

module = import_module(args.model)

data_name = os.path.basename(args.input).split(".")[0]
all_output_dir = os.path.join("output", data_name)

for xchannel, xdrug in channels_drugs:
    expt_labels, concs, responses = data.load_data(xchannel, xdrug)
    
    channel = xchannel.replace("/", "_").replace("\\", "_")
    drug = xdrug.replace("/", "_").replace("\\", "_")
    output_dir = os.path.join(all_output_dir, channel, drug, args.model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig_prefix_0 = f"{data_name}_{channel}_{drug}"
    plots.plot_data(output_dir, fig_prefix_0, channel, drug, expt_labels, concs,
                    responses)
    
    if args.bf:
        marginal_lls = []
    for model_number in range(1, module.n_models+1):
        print("Model", model_number)
        fig_prefix = f"{fig_prefix_0}_model_{model_number}"
        model, remove, get_sample = module.expt_model(model_number, concs,
                                                  responses, expt_labels)
        try:
            # Draw the plate diagram of the statistical model.
            # This is good to check that the model looks how it should.
            # However, because of transformations of some variables, it is
            # rather cluttered, so I recommend drawing your own cleaner version
            # if you plan to publish it somewhere.
            graph = pm.model_to_graphviz(model)
            graph.render(os.path.join(output_dir,
                                      f"model_{model_number}_graph"))
        except:
            print("Can't render graph: graphviz (Python and/or system) is not"
                  + " installed.")
        with model:
            if args.bf:
                trace = pm.sample_smc(args.iterations, n_steps=50)
                marginal_lls.append(model.marginal_likelihood)
                n_iterations = args.iterations
            else:
                n_iterations = 4*args.iterations
                trace = pm.sample(args.iterations, tune=args.iterations)
            for name in remove:
                trace.remove_values(name)
        
        if 1 <= model_number <= 3:
            samples = npr.randint(n_iterations, size=600)
            plots.plot_sample_curves_pooled(output_dir, fig_prefix, channel, drug,
                                            expt_labels, concs, responses,
                                            model_number, samples, get_sample,
                                            trace)
        
        # Change data type to avoid pesky warning when plotting the rest
        trace = az.from_pymc3(trace, log_likelihood=False)
        
        # Save pair plots without displaying
        plots.plot_pairs(output_dir, fig_prefix, trace)
        
        # Save posterior KDE plots without displaying
        plots.plot_kdes(output_dir, fig_prefix, trace)
        
    if args.bf:
        bf_file = f"{data_name}_{channel}_{drug}_{args.model}_BFs.txt"
        output_file = os.path.join(output_dir, bf_file)
        with open(output_file, "w") as outf:
            outf.write("Bayes Factors\n")
            for i, j in it.combinations(range(module.n_models), r=2):
                line = f"B{j+1}{i+1} = {marginal_lls[j] / marginal_lls[i]}"
                print(line)
                outf.write(f"{line}\n")

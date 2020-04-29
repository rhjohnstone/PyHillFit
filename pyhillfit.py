import doseresponse as dr
import argparse
import os
import sys
import itertools as it
from time import time

# TODO - basic optimization
#      - more plots/analysis
#      - plot dose-response curves from samples
#      - BF computation
#      - double-check requirements.txt
#      - for now, revert back to saving all parameters then getting rid of
#        the unshifted ones. later, maybe, if I want to use my own plotting
#        tools, can save only unshifted parameters, then shift before analysis

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

module = import_module(args.model)

data_name = os.path.basename(args.input).split(".")[0]
output_dir = os.path.join("output", data_name)

for xchannel, xdrug in channels_drugs:
    concs, responses = data.load_data(xchannel, xdrug)
    channel = xchannel.replace("/", "_").replace("\\", "_")
    drug = xdrug.replace("/", "_").replace("\\", "_")
    current_output_dir = os.path.join(output_dir, channel, drug, args.model)
    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)
    data_plot_f = os.path.join(current_output_dir, f"{data_name}_{channel}_{drug}_data.png")
    fig = dr.plot_data(channel, drug, concs, responses)
    fig.savefig(data_plot_f)
    plt.close()
    if args.bf:
        marginal_lls = []
    for model_number in range(1, module.n_models+1):
        model, fs, dr_model = module.expt_model(model_number, concs, responses)
        with model:
            if args.bf:
                trace = pm.sample_smc(args.iterations, n_steps=50)
                marginal_lls.append(model.marginal_likelihood)
                n_iterations = args.iterations
            else:
                trace = pm.sample(args.iterations, tune=args.iterations)
                n_iterations = 4*args.iterations
        trace = {varname: f(trace[varname]) for varname, f in fs.items()}
        pp = pm.pairplot(trace, plot_kwargs={"alpha":0.01})
        fig = plt.gcf()
        fig_file = f"{data_name}_{channel}_{drug}_{args.model}_model_{model_number}_pair.png"
        output_fig = os.path.join(current_output_dir, fig_file)
        fig.savefig(output_fig)
        
        tp = pm.plot_posterior(trace, credible_interval=0.95)
        fig = plt.gcf()
        fig_file = f"{data_name}_{channel}_{drug}_{args.model}_model_{model_number}_trace.png"
        output_fig = os.path.join(current_output_dir, fig_file)
        fig.savefig(output_fig)
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(5,4))
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        samples = npr.randint(n_iterations, size=500)
        x = np.logspace(-4, 4, 101)
        fig_file = f"{data_name}_{channel}_{drug}_{args.model}_model_{model_number}_sample_curves.png"
        output_fig = os.path.join(current_output_dir, fig_file)
        for sample in samples:
            
            ax.plot(x, dr_model(x, trace, sample, model_number),
                    color="k", alpha=0.01)
        fig.savefig(output_fig)
        plt.close()
        
    if args.bf:
        bf_file = f"{data_name}_{channel}_{drug}_{args.model}_BFs.txt"
        output_file = os.path.join(current_output_dir, bf_file)
        with open(output_file, "w") as outf:
            outf.write("Bayes Factors\n")
            for i, j in it.combinations(range(module.n_models), r=2):
                line = f"B{j+1}{i+1} = {marginal_lls[j] / marginal_lls[i]}"
                print(line)
                outf.write(f"{line}\n")

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

#T0 = time()

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
    #continue
    if args.bf:
        marginal_lls = []
    for model_number in [2,3]:#range(1, module.n_models+1):
    
        model, fs = module.expt_model(model_number, concs, responses)
        #t0 = time()
        with model:
            if args.bf:
                trace = pm.sample_smc(args.iterations, n_steps=50)
                marginal_lls.append(model.marginal_likelihood)
                n_iterations = args.iterations
            else:
                trace = pm.sample(args.iterations, tune=args.iterations)
                n_iterations = 4*args.iterations
        #print("TIME TAKEN:", time() - t0, "s")
        trace = {varname: f(trace[varname]) for varname, f in fs.items()}
        pp = az.plot_pair(trace, plot_kwargs={"alpha":0.01})
        fig = plt.gcf()
        fig_file = f"{data_name}_{channel}_{drug}_{args.model}_model_{model_number}_pair.png"
        output_fig = os.path.join(current_output_dir, fig_file)
        fig.savefig(output_fig)
        tp = az.plot_trace(trace)
        fig = plt.gcf()
        fig_file = f"{data_name}_{channel}_{drug}_{args.model}_model_{model_number}_trace.png"
        output_fig = os.path.join(current_output_dir, fig_file)
        fig.savefig(output_fig)
        plt.close()
        
        if model_number == 2:
            fig, ax = plt.subplots(1, 1, figsize=(5,4))
            ax.set_xscale("log")
            ax.set_ylim(0, 100)
            samples = npr.randint(n_iterations, size=500)
            x = np.logspace(-4, 4, 101)
            for sample in samples:
                ax.plot(x, dr.per_cent_block(x, trace["Hill"][sample],
                                             trace["pIC50"][sample], 0),
                        color="k", alpha=0.01)
            fig.savefig("samples2.png")
            plt.close()
        elif model_number == 3:
            fig, ax = plt.subplots(1, 1, figsize=(5,4))
            ax.set_xscale("log")
            ax.set_ylim(0, 100)
            samples = npr.randint(n_iterations, size=500)
            x = np.logspace(-4, 4, 101)
            for sample in samples:
                ax.plot(x, dr.per_cent_block(x, trace["Hill"][sample],
                                             trace["pIC50"][sample],
                                             trace["Saturation"][sample]),
                        color="k", alpha=0.01)
            fig.savefig("samples3.png")
            plt.close()
        
        
    #print(time() - T0)
            
    if args.bf:
        for i, j in it.combinations(range(module.n_models), r=2):
            print(f"B{j+1}{i+1} = {marginal_lls[j] / marginal_lls[i]}")
    

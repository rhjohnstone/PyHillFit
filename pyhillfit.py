import doseresponse as dr
import argparse
import os

# TODO - basic optimization
#      - more plots/analysis
#      - plot dose-response curves from samples
#      - BF computation
#      - double-check requirements.txt

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="dose-response data file",
                    default=os.path.join("data", "crumb.csv"))
parser.add_argument("--model", type=str, help="probabilistic model definition",
                    default="johnstone")
parser.add_argument("--all", action="store_true",
                    help="run all channel/drug combinations")
parser.add_argument("--iterations", type=int, default=10000,
                    help="number of MCMC samples (plus the same number "
                         "again for tuning")
args = parser.parse_args()

data = dr.Data(args.input)
channels_drugs = data.select_channel_drug(args.all)


# importing here so we don't have to wait to choose channel/drug
import pymc3 as pm
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from importlib import import_module

module = import_module(args.model)

data_name = os.path.basename(args.input).split(".")[0]
output_dir = os.path.join("output", data_name)

for channel, drug in channels_drugs:
    concs, responses = data.load_data(channel, drug)
    current_output_dir = os.path.join(output_dir, channel, drug, args.model)
    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)

    for model_number in range(1, module.n_models+1):
    
        model, remove = module.model(model_number, concs, responses)
        with model:
            trace = pm.sample(args.iterations, tune=args.iterations)
        for p in remove:
            trace.remove_values(p)

        tp = pm.traceplot(trace)
        fig = plt.gcf() # to get the current figure...
        fig_file = f"{data_name}_{channel}_{drug}_{args.model}_model_{model_number}.png"
        output_fig = os.path.join(current_output_dir, fig_file)
        fig.savefig(output_fig) # and save it directly
        

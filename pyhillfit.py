import pymc3 as pm
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import doseresponse as dr
import argparse
import os
from importlib import import_module

# TODO - data and model as command line arguments
#      - basic optimization
#      - BF computation
#      - programmatically import model
#      - save output to model-specific folder

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="dose-response data file",
                    default=os.path.join("data", "crumb.csv"))
parser.add_argument("--model", type=str, help="probabilistic model definition",
                    default="johnstone")
parser.add_argument("--all", action="store_true",
                    help="run all channel/drug combinations")
args = parser.parse_args()

model = import_module(args.model)

output_dir = os.path.join("output", args.model)

data = dr.Data(args.input)
channels_drugs = data.select_channel_drug(args.all)

iterations = tune = 1000
for channel, drug in channels_drugs:
    concs, data = data.load_data(channel, drug)
    current_output_dir = os.path.join(output_dir, channel, drug)
    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)

    for model_number in range(1, model.n_models+1):
    
        trace = model.model(model_number, iterations, tune, concs, data)

        tp = pm.traceplot(trace)
        fig = plt.gcf() # to get the current figure...
        output_fig = os.path.join(current_output_dir, f"{channel}_{drug}_model_{model_number}.png")
        fig.savefig(output_fig) # and save it directly
        

import pymc3 as pm
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import doseresponse as dr
from johnstone import johnstone

f = "data/crumb.csv"
data = dr.Data(f)
channels_drugs = data.select_channel_drug()

model = 1
iterations = tune = 2000
for channel, drug in channels_drugs:
    concs, data = data.load_data(channel, drug)

    trace = johnstone(model, iterations, tune, concs, data)

    tp = pm.traceplot(trace)
    fig = plt.gcf() # to get the current figure...
    fig.savefig(f"{channel}_{drug}_model_{model}.png") # and save it directly
    

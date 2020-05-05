import pandas as pd
import itertools as it
import numpy as np


def encode_labels(input):
    """
    Reduce experiment numbers/labels to smallest unique integers.
    We do this so we can correctly index random variables in hierarchical
    models.
    """
    current = 0
    labels = dict()
    output = []
    for x in input:
        label = labels.get(x, current)
        if label == current:
            labels[x] = current
            current += 1
        output.append(label)
    return np.array(output)
    
    
def conductance_scale(x, hill, pic50, saturation):
    """
    Saturated Hill curve model of maximal conductance scale.
    'concudctance scale' = 1 - 'conductance block'.
    
    Must use Theano-frienly operations here for PyMC3 to work.
    """
    ic50 = 10**(6-pic50)
    return saturation + (1-saturation) / (1 + (x/ic50)**hill)


def per_cent_block(x, hill, pic50, saturation):
    """
    If we know the data is in the form of blocks, and that it is an inhibitor,
    not an activator.
    """
    return 100 * (1 - conductance_scale(x, hill, pic50, saturation))
    

# Data file must be in CSV format with a header line.
# The columns must be in the same format as col_names, but the names themselves
# can be different.
col_names = ["drug", "channel", "expt", "conc", "block"]


class Data:
    """
    Ion channel screening data handler.
    Doesn't really need to be a class, but it just lets us reuse the same input
    file.
    
    The CSV file is loaded initially just to scan for channel and drug names
    and allow us to choose which combination to run (or to run all).
    
    The same CSV file is reloaded every time we run a new channel/drug
    combination, and the relevant channel/drug entries are kept with everything
    else forgotten.
    This is definitely not optimal for small enough datasets, like the included
    Crumb dataset, but if we had a massive dataset then it might cause problems
    if we kept everything in memory, along with the MCMC samples.
    Even though it's not optimal, reloading the CSV file is a small task
    compared to the actual MCMC sampling, so I don't really think it matters.
    """
    def __init__(self, f):
        self.f = f

    def select_channel_drug(self, run_all=False):
        """
        Print list of channels and drugs from the data file, for the user to
        choose one of each.
        
        Alternatively, if run_all=True, select all channel/drug combinations.
        """
        data = pd.read_csv(self.f, header=0, names=col_names)
        channels = sorted(data["channel"].unique())
        drugs = sorted(data["drug"].unique())
        if run_all:
            return it.product(channels, drugs)
        else:
            labels = ["Channels", "Drugs"]
            options = [channels, drugs]
            channel_drug = []
            for label, option in zip(labels, options):
                print(f"\n{label}:")
                for i, opt in enumerate(option, 1):
                    print(f"{i:>2}. {opt}")
                choice_idx = int(input("Choose number: ")) - 1
                channel_drug.append(option[choice_idx])
            # Return as list so we can use the same iterating code as when we
            # do run_all.
            return [tuple(channel_drug)]

    def load_data(self, channel, drug):
        """
        Load the relevant channel/drug entries.
        We also keep the experiment numbers/labels for indexing in hierarchical
        models.
        """
        data = pd.read_csv(self.f, header=0, names=col_names)
        channel_rows = (data["channel"] == channel)
        drug_rows = (data["drug"] == drug)
        data = data[channel_rows & drug_rows].drop(columns=["channel", "drug"])
        expts, concs, responses = data.values.T
        # Reduce labels to smallest unique integers for correct indexing
        expt_labels = encode_labels(expts)
        return expt_labels, concs, responses




#import numpy as np
import pandas as pd
import itertools as it

#np.seterr("raise")
def hill_curve(x, hill, pic50):
    ic50 = 10**(6-pic50)
    return 100 * ( 1 - 1 / (1 + (x/ic50)**hill) )

def saturated_hil_curve(x, hill, pic50, saturation):
    ic50 = 10**(6-pic50)
    return saturation - saturation / (1 + (x/ic50)**hill)

# Data file must be in CSV format with a header line.
# The columns must be in the same format as col_names, but the names themselves
# can be different.
col_names = ["drug", "channel", "expt", "conc", "block"]


class Data:
    def __init__(self, f):
        self.f = f

    def select_channel_drug(self, run_all=False):
        """Print list of channels and drugs from the data file, for the user to
        choose one of each.
        
        Alternatively, if run_all=True, select all channel/drug combinations.
        """
        data = pd.read_csv(self.f, header=0, names=col_names)
        channels = sorted(data.channel.unique().tolist())
        drugs = sorted(data.drug.unique().tolist())
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
            return [tuple(channel_drug)]

    def load_data(self, channel, drug):
        data = pd.read_csv(self.f, header=0, names=col_names)
        channel_rows = (data.channel == channel)
        drug_rows = (data.drug == drug)
        data = data[channel_rows & drug_rows]
        return data.conc.values, data.block.values


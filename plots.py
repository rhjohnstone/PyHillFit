import matplotlib.pyplot as plt
import os
import numpy as np
from math import floor, ceil
import doseresponse as dr
import arviz as az


def plot_data(output_dir, fig_prefix, channel, drug, expt_labels, concs,
              responses):
    """
    Simple scatter plot of ion channel screening data.
    
    Different experiments (repeats) are plotted in different colours just so we
    can check for inter-experiment patterns, e.g. one particular experiment
    might consistently have higher measurements than other experiments.
    """
    fig_file = f"{fig_prefix}_data.png"
    f_out = os.path.join(output_dir, fig_file)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    xmin = floor(np.log10(np.min(concs)))
    xmax = ceil(np.log10(np.max(concs))) + 1
    ax.grid()
    ax.set_xscale("log")
    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(0, 100)
    for i in sorted(set(expt_labels)):
        which = np.where(expt_labels == i)[0]
        ax.scatter(concs[which], responses[which], s=100, edgecolor="k",
                   clip_on=False, zorder=10, label=f"Expt {i}")
    ax.set_xlabel(f"{drug} concentration ($\mu$M)")
    ax.set_ylabel(f"{channel} block (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f_out)
    plt.close()


def plot_pairs(output_dir, fig_prefix, trace):
    """Marginal 2-d scatter plots of all parameter pairs."""
    pp = az.plot_pair(trace, plot_kwargs={"alpha":0.01})
    fig = plt.gcf()
    fig_file = f"{fig_prefix}_pairs.png"
    output_fig = os.path.join(output_dir, fig_file)
    fig.savefig(output_fig)
    plt.close()


def plot_kdes(output_dir, fig_prefix, trace):
    """KDE plots of marginal posterior distributions."""
    tp = az.plot_posterior(trace, credible_interval=0.95)
    fig = plt.gcf()
    fig_file = f"{fig_prefix}_kdes.png"
    output_fig = os.path.join(output_dir, fig_file)
    fig.savefig(output_fig)
    plt.close()


def plot_sample_curves_pooled(output_dir, fig_prefix, channel, drug,
                              expt_labels, concs, responses, model_number,
                              samples, get_sample, trace):
    """
    Draw samples from posterior distribution (from the chain, really) and
    plot dose-response curves, constructing a probability distribution of the
    'true' dose-response behaviour.
    
    Need to pass the function get_sample as an argument so it knows how to
    sample parameters that aren't there. This was done to try and keep the
    model definition script as separate as possible from the rest.
    """
    fig_file = f"{fig_prefix}_sample_curves.png"
    f_out = os.path.join(output_dir, fig_file)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    xmin = floor(np.log10(np.min(concs)))
    xmax = ceil(np.log10(np.max(concs))) + 2
    x = np.logspace(xmin, xmax, 101)
    ax.grid()
    ax.set_xscale("log")
    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(0, 100)
    for sample in samples:
        pic50, hill, saturation = get_sample(model_number, trace, sample)
        pred = dr.per_cent_block(x, hill, pic50, saturation)
        ax.plot(x, pred, color="k", alpha=0.01)
    for i in sorted(set(expt_labels)):
        which = np.where(expt_labels == i)[0]
        ax.scatter(concs[which], responses[which], s=100, clip_on=False,
                   zorder=10)
    ax.set_xlabel(f"{drug} concentration ($\mu$M)")
    ax.set_ylabel(f"{channel} block (%)")
    fig.tight_layout()
    fig.savefig(f_out)
    plt.close()

import os
import json
import numpy as np
from matplotlib import pyplot as plt


def plot_metric(loss, fig_path, exp_num, name='Loss'):
    fig, ax = plt.subplots()
    ax.plot(loss, 'r')
    # ax.legend(loc='upper right')
    ax.set_xlabel('Iterations')
    ax.set_ylabel(name)
    plt.savefig(fig_path + "_{}_{}.png".format(name, exp_num))


def plot_vae_loss(loss, fig_path, exp_num, name='Loss'):
    fig, ax = plt.subplots()
    ax.plot(loss, 'r')
    # ax.legend(loc='upper right')
    ax.set_xlabel('Iterations')
    ax.set_ylabel(name)
    plt.savefig(fig_path + "_VAE_{}.png".format(exp_num))


def plot_vae_alternating_loss(sup_loss, unsup_loss, fig_path, exp_num):
    fig, ax = plt.subplots()
    ax.semilogy(sup_loss, 'b', label='Supervised Loss')
    ax.semilogy(unsup_loss, 'r', label='Unsupervised Loss')
    ax.legend(loc='upper right')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    plt.savefig(fig_path + "_VAE_{}.png".format(exp_num))
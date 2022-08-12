from matplotlib import pyplot as plt


def plot_metric(metric, fig_path, y_label='Loss', x_label='Iterations'):
    fig, ax = plt.subplots()
    ax.plot(metric, 'r')
    
    ax.grid(True)
    # ax.legend(loc='upper right')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(fig_path)
    plt.close()


def plot_metrics(metrics, labels, fig_path, y_label='Loss', x_label='Iterations'):
    fig, ax = plt.subplots()
    for metric, label in zip(metrics, labels):
        ax.semilogy(metric, label=label)
    
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(fig_path)
    plt.close()

    
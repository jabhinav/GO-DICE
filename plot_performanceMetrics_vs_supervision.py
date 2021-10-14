import os
import json
import numpy as np
import matplotlib.pyplot as plt

root_dir = "ArchivesForGraph"
num_data_dir = os.listdir(root_dir)
num_data_dir = [dir for dir in num_data_dir if not dir.startswith(".")]

# Gather Data
supervision = [0, 10, 25, 50, 100]
num_traj_in_data = [30, 100]
num_data_perf = {
    '30': {
        'zero_one_loss' : [], # [[mean, var]]
        'hamming' : [],
        'kl_policy' : [],
        'weighted_zero_one_loss' : [],
        'weighted_kl_policy' : []
    },
    '100': {
        'zero_one_loss' : [],
        'hamming' : [],
        'kl_policy' : [],
        'weighted_zero_one_loss' : [],
        'weighted_kl_policy' : []
    }
}
for percent in num_traj_in_data:
    data_dir_path = os.path.join(root_dir, "{}datas".format(percent))
    json_files = ["averages_{}percent.json".format(num) for num in supervision]
    for _file in json_files:
        json_path = os.path.join(data_dir_path, _file)
        with open(json_path, 'r') as f:
            _dict = json.load(f)

        num_data_perf[str(percent)]['zero_one_loss'].append(_dict["munkrees metrics"]["0-1 loss"])
        num_data_perf[str(percent)]['hamming'].append(_dict["munkrees metrics"]["hamming"])
        num_data_perf[str(percent)]['kl_policy'].append(_dict["munkrees metrics"]["kl_policy"])
        num_data_perf[str(percent)]['weighted_zero_one_loss'].append(_dict["munkrees metrics"]["weighted_0-1 loss"])
        num_data_perf[str(percent)]['weighted_kl_policy'].append(_dict["munkrees metrics"]["weighted_kl_policy"])

# print(num_data_perf)

# Plot the data
for fig_type in num_data_perf["30"].keys():
    fig, ax = plt.subplots()
    for percent in num_traj_in_data:
        ax.errorbar(x = np.array(supervision),
                    y = np.array([_data[0] for _data in num_data_perf[str(percent)][fig_type]]),
                    yerr=np.array([_data[1] for _data in num_data_perf[str(percent)][fig_type]]),
                    label=str(percent)+" traj", marker='o')
    ax.grid(zorder=0)
    ax.legend(loc='upper right', numpoints=1)
    ax.set_xlabel("Percantage Supervision")
    ax.set_ylabel(fig_type)
    plt.savefig(fig_type + ".png")
    plt.clf()

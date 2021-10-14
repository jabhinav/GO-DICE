import os
import numpy as np
import pickle as pkl

# For python3-python2
path = "../training_data/TeamBoxWorld"

files = os.listdir(path)
print (files)
for f in files:
    if not f.startswith(".") and not f.endswith(".json"):
        input_path = os.path.join(path, f)
        output_path = os.path.join(path, f.split(".")[0]+".pkl")
        data = np.load(input_path, allow_pickle=True)
        with open(output_path, 'wb') as o:
            pkl.dump(data, o, protocol=2)

import numpy as np
import pandas as pd
import os

tune = []
for i in os.listdir("/mnt/data6/DeepPY/ai_lung_damage/data/tune/"):
    if i.endswith("_img.nrrd.npy"):
        id = i.split("_")[0]
        print(id)
        tune.append(id)
np.save("/mnt/data6/DeepPY/ai_lung_damage/tune_pids.npy", tune)
        
print(tune)

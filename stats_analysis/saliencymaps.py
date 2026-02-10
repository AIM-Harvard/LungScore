from LungScore.run import lunghealth_load, lunghealth_predict
import torch
import numpy as np
import SimpleITK as sitk

device = torch.device('cuda')

#preprocessed_lung = np.load('/mnt/data6/DeepPY/ai_lung_damage/data/test/100214_T0_img.nrrd.npy')

preprocessed_lung = np.load('/mnt/data6/DeepPY/ai_lung_damage/data/FHS_SLCREMOVED28/1-2354.npy')


preprocessed_lung = torch.tensor(preprocessed_lung, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
print(preprocessed_lung.shape)


model = lunghealth_load()
ai_lunghealth_score = lunghealth_predict(model, preprocessed_lung)

# Backprop from the output w.r.t. input
ai_lunghealth_score.backward()  # computes gradient w.r.t input

# Extract gradients
saliency = preprocessed_lung.grad.abs().detach().cpu().numpy()[0, 0]  # shape [D, H, W]

saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

print(saliency.shape)
print(saliency.max())


np.save("/mnt/data6/DeepPY/saliencymaps/saliencymapFHSCT2_verylowlunghealth_1-2354.npy", saliency)       
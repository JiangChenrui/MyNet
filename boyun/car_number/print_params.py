import torch
import models.dfanet as dfanet
import numpy as np
import sys

# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
np.set_printoptions(threshold=sys.maxsize)

model_path = "model_best_8_13_checkpoint.pth.tar"
model = dfanet.dfanet()
model.load_state_dict(torch.load(model_path)['state_dict'])
params_txt = 'params.txt'
pf = open(params_txt, 'w')
for name, param in model.named_parameters():
    pf.write(name)
    pf.write('\n')
    pf.write('\n' + name + '_weight:\n\n')
    weight = param.detach().numpy()
    weight.shape = (-1, 1)
    for w in weight:
        pf.write('%ff,' % w)
    pf.write('\n\n')
pf.close()

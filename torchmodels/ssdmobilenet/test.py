# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import torch
import numpy as np



from ssd_mobilenet_v1 import create_mobilenetv1_ssd


model = create_mobilenetv1_ssd(10)
model.eval()


image = torch.rand(1,3,300,300)


with torch.no_grad():
    boxes, labels, scores = model.predict(image)







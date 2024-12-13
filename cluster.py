from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from init_parameter import *
from dataloader import *

#from ..builder import HEADS
#from .cls_head import ClsHead
import torch.distributed as dist
import ast
import math
import time
import pandas as pd
import random
import numpy
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
# from config import args
import numpy as np
# import resnet

from sklearn.cluster import KMeans
import numpy as np
from dataloader import *
from init_parameter import *
from cluster2 import GBNR

class gbcluster(nn.Module):

    def __init__(self,args,data):
        super(gbcluster, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, args, features, labels,select):
        if select == False:
            a_purity = args.purity_train
        else:
            a_purity = args.purity_get_ball
        original_target=labels
        index1=torch.arange(len(labels)).to('cuda:0')

        label_features = torch.cat((labels.reshape(-1, 1), features), dim=1)

        out = torch.cat((index1.reshape(-1, 1), label_features ), dim=1)
        out = torch.cat((original_target.reshape(-1, 1), out), dim=1)
        pur_tensor = torch.Tensor([[a_purity]] * out.size(0))
        out = torch.cat((pur_tensor.to(self.device), out), dim=1)
        self.center, self.labels, self.radius = GBNR.apply(args,out.to(self.device),select)

        gb_centroids =self.center
        gb_radii=self.radius
        gb_labels=self.labels

        return  gb_centroids, gb_radii,gb_labels
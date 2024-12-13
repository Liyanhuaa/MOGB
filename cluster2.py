import csv
#from config import opt
# from pandocfilters import Math
from scipy import stats
import torch
import numpy as np
# from config import args

import cluster3 as new_GBNR



def calculate_distances(center, p):
    return ((center - p) ** 2).sum(axis=0) ** 0.5


class GBNR(torch.autograd.Function):
    @staticmethod
    def forward(self,args, input_,select):

        self.batch_size = input_.size(0)
        input_main = input_[:, 3:]  # noise_label+64 [bs,65]
        self.input = input_[:, 4:]
        self.res = input_[:, 1:2]
        self.index = input_[:, 2:3]
        pur = input_[:, 0].cpu().numpy().tolist()[0]

        self.flag = 0

        numbers,  result, center, radius = new_GBNR.main(args,input_main,select)


        labels=[]
        centroids=[]
        for gb in center:
            label=  gb[0]
            centroid=gb[1:]
            labels.append(label)
            centroids.append(centroid)
        self.labels  = labels
        self.centroids=  centroids


        return  self.centroids, self.labels, radius,

    @staticmethod
    def backward(self, output_grad, input, index, id, _):

        result = np.zeros([self.batch_size, 154], dtype='float64')

        for i in range(output_grad.size(0)):

            for a in self.balls[i]:
                input_np = np.array(self.input)
                a_np = np.array(a[1:])

                if input_np.shape[1:] == a_np.shape:
                    mask = (input_np == a_np).all(axis=1)
                    if mask.any():
                        result[mask, 4:] = output_grad[i, :].cpu().numpy()

        return torch.Tensor(result)



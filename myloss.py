from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from cluster import gbcluster
from dataloader import*
from sklearn.metrics import pairwise_distances_argmin_min
from init_parameter import *
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)

    b = b.unsqueeze(0).expand(n, m, -1)
    logits = ((a - b) ** 2).sum(dim=2)
    return logits


class clusterLoss(nn.Module):

    def __init__(self, args,data):

        super(clusterLoss, self).__init__()
        self.num_labels = data.num_labels
        self.feat_dim = args.feat_dim


        self.gbcluster=gbcluster(args,data)
        self.gb_centroids = None
        self.gb_radii = None
        self.gb_labels = None


    def forward(self, args, features, labels, select=True):
        gb_centroids, gb_radii,  gb_labels = self.gbcluster.forward(args, features, labels, select)

        self.gb_centroids = torch.tensor(gb_centroids).float().to(features.device)
        self.gb_radii = torch.tensor(gb_radii).float().to(features.device)

        self.gb_labels = torch.tensor(gb_labels).long().to(features.device)


        sorted_indices = torch.argsort(self.gb_labels)
        self.gb_centroids = self.gb_centroids[sorted_indices]
        self.gb_radii = self.gb_radii[sorted_indices]

        self.gb_labels = self.gb_labels[sorted_indices]




        loss = self.compute_classification_loss(features, labels, self.gb_centroids, self.gb_labels)

        return self.gb_centroids, self.gb_radii, self.gb_labels, loss

    def compute_classification_loss(self, features, labels, centroids, centroid_labels):
        if features.size(0) == 0:
            return torch.tensor(0.0).to(features.device)


        logits = torch.cdist(features, centroids, p=2)
        distances = torch.full((features.shape[0], self.num_labels), float('inf')).to(logits.device)


        for label in range(self.num_labels):
            class_mask = (centroid_labels == label)
            if class_mask.any():
                class_distances = logits[:, class_mask]
                distances[:, label] = class_distances.min(dim=1)[0]


        distances = F.normalize(distances, p=1, dim=1)

        probabilities = F.softmax(-distances, dim=1)

        true_probabilities = probabilities[torch.arange(probabilities.size(0)), labels]

        loss = -torch.log(true_probabilities).mean()
        return loss

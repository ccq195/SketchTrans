import torch
from torch import nn
import numpy as np

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

def hard_example_mining2(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an



def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

class PWeightedRegularizedTriplet(object):

    def __init__(self, gamma=0.0,beta=1.0):
        self.ranking_loss = nn.SoftMarginLoss(reduction='none')
        self.gamma = gamma
        self.beta = beta

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos#48*48
        dist_an = dist_mat * is_neg#48*48


        weights_ap = softmax_weights(dist_ap, is_pos)#48*48
        weights_an = softmax_weights(-dist_an, is_neg)#48*48

        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)#48
        # closest_negative = torch.sum(dist_an * weights_an, dim=1)#48

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)#48

        D = dist_an - furthest_positive

        loss = self.ranking_loss(dist_an - furthest_positive, y)
        loss_tr = ((loss*is_neg).sum())/len(is_neg[is_neg==True])

        # loss_reg = (dist_ap[is_pos==True].mean())

        return loss_tr, furthest_positive, dist_an

class WeightedRegularizedTriplet(object):

    def __init__(self):
        self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss, furthest_positive, closest_negative

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx

class TDRLoss(nn.Module):
    """Tri-directional ranking loss.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TDRLoss, self).__init__()
        self.margin = margin
        # self.ranking_loss = nn.MarginRankingLoss(reduction='none', margin=margin)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, m_targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.shape[0] // 3
        input1 = inputs[:2*n][m_targets==0]
        input2 = inputs[2*n:]
        input3 = inputs[:2 * n][m_targets == 1]

        dist1 = pdist_torch(input1, input2)
        dist2 = pdist_torch(input2, input3)
        dist3 = pdist_torch(input1, input3)

        # compute mask
        targets = targets[2*n:]
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # P: 1 2 N: 3
        dist_ap1, dist_an1 = [], []
        for i in range(n):
            dist_ap1.append(dist1[i][mask[i]].max().unsqueeze(0))
            dist_an1.append(dist3[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an1)
        loss1 = self.ranking_loss(dist_an1, dist_ap1, y)
        weights1 = loss1.data#.exp()
        # weights1 = loss1.data.pow(2)

        # P: 2 3 N: 1
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist2[i][mask[i]].max().unsqueeze(0))
            dist_an2.append(dist1[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)

        # Compute ranking hinge loss
        loss2 = self.ranking_loss(dist_an2, dist_ap2, y)
        weights2 = loss2.data#.exp()
        # weights2 = loss2.data.pow(2)


        # P: 3 1 N: 2
        dist_ap3, dist_an3 = [], []
        for i in range(n):
            dist_ap3.append(dist3[i][mask[i]].max().unsqueeze(0))
            dist_an3.append(dist2[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap3 = torch.cat(dist_ap3)
        dist_an3 = torch.cat(dist_an3)

        # Compute ranking hinge loss
        loss3 = self.ranking_loss(dist_an3, dist_ap3, y)
        weights3 = loss3.data#.exp()
        # weights3 = loss3.data.pow(2)

        # compute accuracy
        correct1 = torch.ge(dist_an1, dist_ap1).sum().item()
        correct2 = torch.ge(dist_an2, dist_ap2).sum().item()
        correct3 = torch.ge(dist_an3, dist_ap3).sum().item()

        loss_reg = dist_ap1.mean() + dist_ap2.mean() + dist_ap3.mean()

        return (loss1+loss2+loss3)/3.0, loss_reg, correct1 + correct2 + correct3

class WTDRLoss(nn.Module):
    """Tri-directional ranking loss.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3,gamma=1.0):
        super(WTDRLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(reduction='none', margin=margin)
        self.gamma = gamma

    def forward(self, inputs, targets,m_targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        eps = 0.000000001
        n = inputs.shape[0] // 3
        input1 = inputs[:2 * n][m_targets == 0]
        input2 = inputs[2 * n:]
        input3 = inputs[:2 * n][m_targets == 1]

        dist1 = pdist_torch(input1, input2)
        dist2 = pdist_torch(input2, input3)
        dist3 = pdist_torch(input1, input3)

        # compute mask
        targets = targets[2 * n:]
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # dist1 = pdist_torch(input1, input2)
        # dist2 = pdist_torch(input2, input3)
        # dist3 = pdist_torch(input1, input3)
        #
        # # compute mask
        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # P: 1 2 N: 3
        dist_ap1, dist_an1 = [], []
        for i in range(n):
            dist_ap1.append(dist1[i][mask[i]].max().unsqueeze(0))
            dist_an1.append(dist3[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an1)
        loss1 = self.ranking_loss(dist_an1, dist_ap1, y)
        # weights1 = loss1.data.exp()
        # weights1 = loss1.data.pow(2)
        weights1 = loss1.data
        weights1 = weights1 / (weights1.max() + eps)

        # P: 2 3 N: 1
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist2[i][mask[i]].max().unsqueeze(0))
            dist_an2.append(dist1[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap2= torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)

        # Compute ranking hinge loss
        loss2 = self.ranking_loss(dist_an2, dist_ap2, y)
        # weights2 = loss2.data.exp()
        # weights2 = loss2.data.pow(2)
        weights2 = loss2.data
        weights2 = weights2/(weights2.max()+eps)


        # P: 3 1 N: 2
        dist_ap3, dist_an3 = [], []
        for i in range(n):
            dist_ap3.append(dist3[i][mask[i]].max().unsqueeze(0))
            dist_an3.append(dist2[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap3 = torch.cat(dist_ap3)
        dist_an3 = torch.cat(dist_an3)

        # Compute ranking hinge loss
        loss3 = self.ranking_loss(dist_an3, dist_ap3, y)
        # weights3 = loss3.data.exp()
        # weights3 = loss3.data.pow(2)
        weights3 = loss3.data
        weights3 = weights3/(weights3.max()+eps)

        # compute accuracy
        correct1 = torch.ge(dist_an1, dist_ap1).sum().item()
        correct2 = torch.ge(dist_an2, dist_ap2).sum().item()
        correct3 = torch.ge(dist_an3, dist_ap3).sum().item()

        H = (3*weights1*weights2*weights3)/((weights1*weights3)+(weights2*weights3)+(weights1*weights2)+eps)
        wloss1 = (1-(weights1+eps)*H)** self.gamma * loss1
        wloss2 = (1-(weights2+eps)*H)** self.gamma * loss2
        wloss3 = (1-(weights3+eps)*H)** self.gamma * loss3

        # weighted aggregation loss
        # weights_sum = torch.cat((weights1, weights2, weights3),0)
        # wloss1 = torch.mul(weights1.div_(weights_sum.sum()), loss1).sum()
        # wloss2 = torch.mul(weights2.div_(weights_sum.sum()), loss2).sum()
        # wloss3 = torch.mul(weights3.div_(weights_sum.sum()), loss3).sum()

        return (wloss1.mean() + wloss2.mean() + wloss3.mean())/3.0, correct1 + correct2+correct3
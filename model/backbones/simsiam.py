import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1)#.mean()
    else:
        raise Exception


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=768): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam_LOSS(nn.Module):
    def __init__(self):
        super().__init__()

        self.predictor = prediction_MLP()
    
    def forward(self, z1, z2):
        p1, p2 = self.predictor(z1), self.predictor(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return L.mean() 


def global_euclidean_dist(x, y):

    """
    :param x: (pytorch Variable) global features [M, d]
    :param y: (pytorch Variable) global features [N, d]
    :return: pytorch Variable euclidean distance matrix [M, N]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist_mat = xx + yy
    dist_mat.addmm_(1, -2, x.float(), y.float().t())
    dist_mat = dist_mat.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist_mat

class Mutual_Loss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, z1, z2):
        
        # global_dist_mat1 = global_euclidean_dist(z1, z1)
        # global_dist_mat2 = global_euclidean_dist(z2, z2)
        z1 = F.normalize(z1, dim=1) # l2-normalize 
        z2 = F.normalize(z2, dim=1) # l2-normalize 
       

        global_dist_mat1 = z1.mm(z1.t())
        global_dist_mat2 = z2.mm(z2.t())
        loss_mutual1 = ((global_dist_mat1-global_dist_mat2)**2).mean(0).mean(0)

        return loss_mutual1


# if __name__ == "__main__":
#     model = SimSiam()
#     x1 = torch.randn((2, 3, 224, 224))
#     x2 = torch.randn_like(x1)

#     model.forward(x1, x2).backward()
#     print("forward backwork check")

#     z1 = torch.randn((200, 2560))
#     z2 = torch.randn_like(z1)
#     import time
#     tic = time.time()
#     print(D(z1, z2, version='original'))
#     toc = time.time()
#     print(toc - tic)
#     tic = time.time()
#     print(D(z1, z2, version='simplified'))
#     toc = time.time()
#     print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469













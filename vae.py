# from mnist import MNIST
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torchvision.utils import save_image

# from torchvision.datasets import MNIST

# mndata = MNIST('/home/ccq/Desktop/ML_learning/pytorch_example-master/data/')
# images, labels = mndata.load_training()
#
# img_transform = transforms.Compose([
#                 transforms.ToTensor()
#                 ])
#
# num_epochs = 20
# batch_size = 128
# learning_rate = 1e-3
# IMG_SIZE = 784
# IMG_WIDTH = 28
# IMG_HEIGHT = 28
#
# dataset = torch.tensor(images)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# def to_img(x):
#     x = x.clamp(0, 1)
#     x = x.view(x.size(0), 1, 28, 28)
#     return x

def calc_mean_std(f,eps=1e-5):
	b,n,c = f.size()
	f = f.transpose(1,2)
	fvar = f.var(dim=-1)+eps
	fstd = fvar.sqrt().view(b,c,1)
	fmean = f.mean(dim=-1).view(b,c,1)
	return ((f-fmean.expand(f.size())) / fstd.expand(f.size())).transpose(1,2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlpadin(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc21 = nn.Linear(hidden_features, out_features)
        self.fc22 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, xc,s1):
        xc = self.fc1(xc)
        xc = self.act(xc)
        xc = self.drop(xc)

        b = self.fc21(xc)
        g = self.fc22(xc)

        return calc_mean_std(s1)*(1+g)+b

class VAE(nn.Module):
	"""
	"""
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		"""
		TODO: doconvolution
		"""
		super(VAE, self,).__init__()

		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()

		self.fc21 = nn.Linear(hidden_features, out_features)
		self.fc22 = nn.Linear(hidden_features, out_features)

		self.fc3 = nn.Linear(out_features*2, out_features)
		self.drop = nn.Dropout(drop)

	def reparametrize(self, mu, std):
		"""
		why we need reparameterize:
			- we want to learn p(z|x) distribution of latent variable given dataset
			- we want p(z|x) constraint on unit guassian
			- but we have fixed input (train-set), if we want model to be randomness, 
			- so we incoporate some noise ~ N(0, 1), and redirect this data to decoder layer
		"""
		eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps).cuda()
		return eps.mul(std).add_(mu)

	def encoder(self, x):
		h1 = self.drop(self.act(self.fc1(x)))
		mu, std = self.drop(self.fc21(h1)), self.drop(self.fc22(h1))
		return mu, std

	def decoder(self, x,att):

		x = torch.cat([x, att], dim=-1)

		h2 = self.drop(self.act((self.fc3(x))))
		return h2
		
	def forward(self, x, att):
		mu, var = self.encoder(x)
		z = self.reparametrize(mu, var)
		return self.decoder(z,att), mu, var

# ref: http://kvfrans.com/variational-autoencoders-explained/
# 	1) encoder loss = mean square error from original image and decoder image
# 	2) decoder loss = KL divergence 
encoder_loss = nn.MSELoss(size_average=True)

def loss_function(output, x, mu, var):
		"""
		"""
		mse = encoder_loss(output, x)
		#	0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) 
		latent_loss = mu.pow(2).add_(var.pow(2)).mul(-1.).add_(torch.log(var.pow(2))).add_(1).mul_(0.5)
		KLD = torch.sum(latent_loss)
		return mse - 0.00000001*KLD

# way to construct DNN network
# 	1) topology 
# 	2) loss function
#		3) optimizer
# 	4) forward
# 	5) free zero grad of variable
# 	6) backward

# model = VAE()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
# for epoch in range(num_epochs):
# 	train_loss = 0
# 	for batch_idx, data in enumerate(dataloader):
# 		img = data.view(data.size(0), -1)
# 		img = Variable(img.float())
# 		# free zero grad
# 		optimizer.zero_grad()
# 		output, mu, var = model(img)
# 		# backward
# 		loss = loss_function(output, img, mu, var)
# 		loss.backward()
# 		train_loss += loss.item()
# 		optimizer.step()
# 		if batch_idx % 100 == 0:
# 			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch,
#                 batch_idx * len(img),
#                 len(dataloader.dataset), 100. * batch_idx / len(dataloader),
#                 loss.item() / len(img)))
# 	print('====> Epoch: {} Average loss: {:.4f}'.format(
#       	epoch, train_loss / len(dataloader.dataset)))
# 	if epoch % 10 == 0:
# 		save = to_img(img.cpu().data)
# 		save_image(save, './var_encoder_img/image_{}.png'.format(epoch))
#
# torch.save(model.state_dict(), './vae.pth')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.linalg as L

def perspective_project(points, matrix):
	p = points.float()
	if p.dim() == 2: p = p.unsqueeze(0)

	n,s,d = p.shape
	if d == 2:
		p = torch.cat((p, torch.ones(n,s,1,device=points.device)), dim=2)
	elif not d == 3:
		raise ValueError(
			"Expected last dimension of points to be of size 2 or 3. "
			"Got {}".format(points.shape[-1]))

	p = p @ matrix.transpose(-2,-1)
	p = p[:,:,:2]/p[:,:,2:]
	return p.view_as(points)

def linspace_from_neg_one(num_steps):
	r = torch.linspace(-1, 1, num_steps)
	r = r * (num_steps - 1) / num_steps
	return r

def perspective_grid_generator(theta, size):
	n,c,h,w = size
	grid = torch.zeros(n,h,w,3, device=theta.device)
	grid.select(-1, 0).copy_(linspace_from_neg_one(w))
	grid.select(-1, 1).copy_(linspace_from_neg_one(h).unsqueeze(-1))
	grid.select(-1, 2).fill_(1)
	grid = grid.view(n,h*w,3) @ theta.transpose(1,2)
	grid = grid.view(n,h,w,3)
	grid = grid[:,:,:,:2] / grid[:,:,:,2:3]
	return grid

'''
# Autograd should take care of this
def perspective_grid_generator_backward(grad_grid, size):
	n,c,h,w = size
	assert grad_grid.shape == (n,h,w,2)
	grid = torch.zeros(n,h,w,3, device=grad_grid.device)
	grid.select(-1, 0).copy_(linspace_from_neg_one(w, device))
	grid.select(-1, 1).copy_(linspace_from_neg_one(h, device))
	grid.select(-1, 2).fill_(1)
	grad_theta = grid.view(n,h*w,3).transpose(1,2).mm(grad_grid.view(n,h*w, 2))
	return grad_theta.transpose(1,2)

class PerspectiveGridGeneratorFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, theta, size):
		n,c,h,w = size
		grid = perspective_grid_generator(theta, size)
		ctx.save_for_backward(n,c,h,w)
		return grid

	@staticmethod
	def backward(ctx, grad_output):
		n,c,h,w = ctx.saved_variables
		theta_grad = perspective_grid_generator_backward(grad_output, (n,c,h,w))
		return theta_grad, None
'''

def perspective_grid(theta, size):
	if not theta.is_floating_point():
		raise ValueError("Expected theta to have floating point type, but got {}".format(theta.dtype))
	if len(size) == 4:
		if theta.dim() != 3 or theta.shape[-2] != 3 or theta.shape[-1] != 3:
			raise ValueError(
				"Expected a batch of 2D projection matrices of shape 3x3 "
				"for size {}. Got {}".format(size, theta.shape))
		h,w = size[-2:]
	else:
		raise NotImplementedError("perspective grid only supports 4D sizes. Got size {}".format(size))

	if min(size) <= 0:
		raise ValueError("Expected non-zero, positive output size. Got {}".format(size))

	#return PerspectiveGridGeneratorFunction.apply(theta, size)
	return perspective_grid_generator(theta, size)


class Localizer(nn.Module):
	def __init__(self,in_channels, in_size, kernel_size=5,  
			out_channels_incr=6, scaling=2,
			matrix_size = (2,3)):

		super(Localizer, self).__init__()
	
		self.body = nn.Sequential()
		c = in_channels
		n = min(in_size)
		m = max(in_size)
		while n >= scaling+kernel_size-1:
			self.body.append(
				nn.Sequential(
					nn.Conv2d(c, c + out_channels_incr, kernel_size),
					nn.MaxPool2d(scaling),
					nn.LeakyReLU()))
			n = n - (kernel_size - 1)
			m = m - (kernel_size - 1)
			n //= scaling
			m //= scaling
			c += out_channels_incr
			
		self.matrix_size = matrix_size

		area = matrix_size[0] * matrix_size[1]
		fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(c * n * m, 4 * area),
			nn.LeakyReLU(),
			nn.Linear(4 * area, area))
		
		fc[-1].weight.data.zero_()
		fc[-1].bias.data.copy_(2 * torch.eye(*matrix_size, dtype=torch.float32).view(-1))
		self.body.append(fc)

	def forward(self, x): 
		y = self.body(x).view(-1,*self.matrix_size)
		return y

class SpatialTransformer(nn.Module):
	def __init__(self, in_channels, in_size, out_size=None, localizer={}):
		super(SpatialTransformer, self).__init__()
		self.out_size = out_size
		self.localizer = Localizer(in_channels, in_size, **localizer)
	
	def forward(self, x):
		assert x.dim() == 4
		theta = self.localizer(x)
		n,c = x.shape[:2]
		h,w = self.out_size if self.out_size is not None else x.shape[2:]
		grid = F.affine_grid(theta, (n,c,h,w), align_corners=False)
		y = F.grid_sample(x, grid, align_corners=False)
		return y

class SpatialProjector(nn.Module):
	def __init__(self, in_channels, in_size, out_size=None, localizer={}):
		super(SpatialProjector, self).__init__()
		self.out_size = out_size
		self.localizer = Localizer(in_channels, in_size, matrix_size=(3,3), **localizer)
	
	def forward(self, x, include_theta = False):
		assert x.dim() == 4
		theta = self.localizer(x)
		n,c = x.shape[:2]
		h,w = self.out_size if self.out_size is not None else x.shape[2:]
		grid = perspective_grid(theta, (n,c,h,w))
		y = F.grid_sample(x, grid, align_corners=False)
		if include_theta:
			return y, theta
		else:
			return y

	@staticmethod
	def get_matrix(src_points, dst_points):
		s0,s1,s2,s3 = src_points.float()
		d0,d1,d2,d3 = dst_points.float()
		x0,y0 = s0
		x1,y1 = s1
		x2,y2 = s2
		x3,y3 = s3
		u0,v0 = d0
		u1,v1 = d1
		u2,v2 = d2
		u3,v3 = d3

		A = torch.tensor([
			[x0, y0, 1,  0,  0, 0, -x0*u0, -y0*u0],
			[x1, y1, 1,  0,  0, 0, -x1*u1, -y1*u1],
			[x2, y2, 1,  0,  0, 0, -x2*u2, -y2*u2],
			[x3, y3, 1,  0,  0, 0, -x3*u3, -y3*u3],
			[ 0,  0, 0, x0, y0, 1, -x0*v0, -y0*v0],
			[ 0,  0, 0, x1, y1, 1, -x1*v1, -y1*v1],
			[ 0,  0, 0, x2, y2, 1, -x2*v2, -y2*v2],
			[ 0,  0, 0, x3, y3, 1, -x3*v3, -y3*v3]]).cpu()
		B = torch.tensor([u0,u1,u2,u3,v0,v1,v2,v3]).cpu()
		x = L.solve(A,B)

		return torch.tensor([
			[x[0], x[1], x[2]],
			[x[3], x[4], x[5]],
			[x[6], x[7],   1]]).to(src_points.device)



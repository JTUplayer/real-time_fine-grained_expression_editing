import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from torchvision.utils import save_image,make_grid
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from criteria.lpips.lpips import LPIPS
from models.stylefeat import StyleFeat
from training.ranger import Ranger
from models.D_AU import Discriminator_AU
from models.stylegan2.model import Discriminator
import numpy as np
from torchvision import transforms
import random
import math
from Facecrop.FaceBoxes import FaceBoxes
from torch import autograd
class Coach:
	def __init__(self, opts):
		self.opts = opts
		#self.opts.learning_rate=0.002
		self.global_step = 0
		os.environ['CUDA_VISIBLE_DEVICES'] = '1'
		#self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		#self.opts.device = self.device

		if self.opts.use_wandb:
			from utils.wandb_utils import WBLogger
			self.wb_logger = WBLogger(self.opts)

		# Initialize network
		self.net = StyleFeat(self.opts)
		# Estimate latent_avg via dense sampling if latent_avg is not available
		# if self.net.latent_avg is None:
		# 	self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()
		if self.opts.is_train:
			self.D = Discriminator_AU()
			self.D.load_state_dict(torch.load("models/D-AU3.pth"))
			self.D = nn.DataParallel(self.D).cuda()
			self.face_boxes = FaceBoxes(cuda=True)
			#self.D=self.D.cuda()
			if self.opts.train_d:
				#self.Discriminator = Discriminator(size=256).cuda()
				self.Discriminator=Discriminator(size=256).cuda()
			#Initialize loss
			if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
				raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')
			self.mse_loss = nn.MSELoss().cuda().eval()
			if self.opts.lpips_lambda > 0:
				#self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
				self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
			if self.opts.id_lambda > 0:
				self.id_loss = id_loss.IDLoss().cuda().eval()
			if self.opts.w_norm_lambda > 0:
				self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
			if self.opts.moco_lambda > 0:
				self.moco_loss = moco_loss.MocoLoss().cuda().eval()

			# Initialize optimizer
			self.configure_optimizers()

			# Initialize dataset
			self.train_dataset,self.train_real_dataset,self.test_dataset= self.configure_datasets()
			if self.train_dataset is not None:
				self.train_dataloader = DataLoader(self.train_dataset,
											   batch_size=self.opts.batch_size,
											   shuffle=True,
											   num_workers=int(self.opts.workers),
											   drop_last=True)
			if self.train_real_dataset is not None:
				self.train_real_dataloader = DataLoader(self.train_real_dataset,
												   batch_size=self.opts.batch_size,
												   shuffle=True,
												   num_workers=int(self.opts.workers),
												   drop_last=True)
			if self.test_dataset is not None:
				self.test_dataloader = DataLoader(self.test_dataset,
												  batch_size=4,
												  shuffle=False,
												  num_workers=int(self.opts.test_workers),
												  drop_last=True)
		else:
			self.generate_img_from_au_direction()
		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps
		self.w_loss = lambda x, y: -torch.mean(x) if y else torch.mean(x)
		self.resize128=transforms.Resize((128,128))
		self.resize256=transforms.Resize((256,256))

	@torch.no_grad()
	def generate_img_from_au_direction(self):
		AU_direction = []
		for i in range(17):
			AU_direction.append(torch.from_numpy(
				np.load(
					'checkpoint/AU-%d.npy' % i)).cuda().unsqueeze(
				0))
		z = torch.randn([4, 512]).cuda().unsqueeze(1)
		print(z.shape)
		w=self.net.decoder.style(z)
		latent=w.repeat(1,18,1)
		print(latent.shape)
		select_au = [i for i in range(17)]
		print(latent.shape)
		for i in range(len(latent)):
			for k in range(len(select_au)):
				au = select_au[k]
				save_img = []
				for j in range(0, 4):
					x = latent[i] + AU_direction[au] * j * 5/4
					y_hat = self.net.face_pool(self.net.decoder([x], input_is_latent=True,
																  randomize_noise=True, return_latents=False)[0])
					save_img.append(y_hat)
				save_img = torch.cat(save_img, dim=0)
				save_img = (save_img + 1) / 2
				grid = make_grid(save_img, nrow=4, padding=0, normalize=False)
				save_image(grid, 'images/%d-%d.png' % (i,k))

	def train(self):
		self.net.train()
		self.optimizer.zero_grad()
		self.global_step=0
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				x, tar_au, y = batch
				x = x * 2 - 1
				y = y * 2 - 1
				x, tar_au = x.cuda().float(), tar_au.cuda().float()
				y_hat, latent = self.net.forward(x, tar_au, return_latents=True, resize=True)
				if self.opts.train_d:
					d_regularize = batch_idx % self.opts.d_reg_every == 0
					fake_pred = self.Discriminator(y_hat.detach())
					real_pred = self.Discriminator(x.detach())
					d_loss = self.d_logistic_loss(real_pred, fake_pred)
					d_loss.backward()
					if self.global_step % self.opts.board_interval == 0:
						print('real_loss:', float(real_pred.mean()))
						print("fake_loss:", float(fake_pred.mean()))
						print("total_loss:", float(d_loss))
					self.optimizer_D.step()
					self.optimizer_D.zero_grad()
					if d_regularize:
						x.requires_grad = True
						real_pred = self.Discriminator(x)
						r1_loss = self.d_r1_loss(real_pred, x)
						self.optimizer_D.zero_grad()
						(self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]).backward()
						self.optimizer_D.step()
				loss, loss_dict, id_logs = self.calc_loss(x, y_hat, tar_au, need_au=True)
				if loss is None:
					y_hat = None
					continue
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				if self.global_step % (self.opts.image_interval) == 0 or (
						self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')
				if self.global_step % 5000 == 0 and self.global_step != 0:
					torch.save(self.net.encoder.state_dict(), "checkpoint/E/E-%d.pth" % self.global_step)
					if self.opts.train_d:
						torch.save(self.Discriminator.state_dict(), "checkpoint/E/D-%d.pth" % self.global_step)
				self.global_step += 1

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
				if self.opts.use_wandb:
					self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		g_reg_ratio = self.opts.g_reg_every / (self.opts.g_reg_every + 1)
		d_reg_ratio = self.opts.d_reg_every / (self.opts.d_reg_every + 1)
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			self.optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate * g_reg_ratio,
			betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
		else:
			self.optimizer = Ranger(params, lr=self.opts.learning_rate)
		if self.opts.train_d:
			self.optimizer_D = torch.optim.Adam(self.Discriminator.parameters(), lr=self.opts.learning_rate * d_reg_ratio,
			betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))
			self.optimizer_D.zero_grad()

	def configure_datasets(self):
		class Train_Data(Dataset):
			def __init__(self, img_path, au_path):
				import numpy as np
				self.imgs = (torch.from_numpy(np.load(img_path).transpose((0, 3, 1, 2))))
				self.aus = torch.from_numpy(np.load(au_path))

			def __getitem__(self, idx):
				# return self.w[idx], self.imgs[idx],self.au[idx]
				tar_idx = random.randint(0, self.aus.size(0) - 1)
				return self.imgs[idx] / 255.0, self.aus[tar_idx], self.imgs[tar_idx] / 255.0

			def __len__(self):
				return self.aus.size(0)

		class Test_Data(Dataset):
			def __init__(self, img_path, au_path):
				self.imgs = (torch.from_numpy(np.load(img_path).transpose((0, 3, 1, 2))))
				self.aus = torch.from_numpy(np.load(au_path))
				self.l = self.aus.size(0)

			def __getitem__(self, idx):
				# return self.w[idx], self.imgs[idx],self.au[idx]
				idx = idx % self.l
				tar_idx = random.randint(0, self.l - 1)
				src_img = (self.imgs[idx] - 127.5) / 127.5
				# src_au = self.aus[idx]
				tar_img = (self.imgs[tar_idx] - 127.5) / 127.5
				tar_au = self.aus[tar_idx]
				return src_img, tar_au, tar_img

			def __len__(self):
				return self.l * 10

		train_dataset = Train_Data('data/img-rafd-train.npy', 'data/au-rafd-train.npy')
		test_dataset = Test_Data('data/img-rafd-test.npy', 'data/au-rafd-test.npy')
		return train_dataset, None, test_dataset
	def crop(self,y_hat):
		try:
			y_crop = None
			resize = transforms.Resize([128, 128])
			for img in y_hat:
				det = self.face_boxes(img[[2, 1, 0]])
				det = list(map(int, det))
				img_crop = img[:, det[1]:det[3], det[0]:det[2]]
				img_crop = resize(img_crop)
				# save_image(img_crop.cpu().detach().float(), "results/%d.png" % random.randint(0,100000), nrow=1, normalize=True)
				img_crop = img_crop.unsqueeze(0)
				if y_crop == None:
					y_crop = img_crop
				else:
					y_crop = torch.cat((y_crop, img_crop), dim=0)
			return y_crop
		except:
			return None
	def calc_loss(self, y, y_hat, tar_au,need_au):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		#loss-real
		y_hat_crop = self.crop(y_hat)
		y_crop = self.crop(y)
		if y_hat_crop is None or y_crop is None:
			loss = 0
			return None, None, None
		if self.opts.train_d:
			fake_pred = self.Discriminator(y_hat)
			loss_real = self.g_nonsaturating_loss(fake_pred)
			loss_dict['loss_real'] = float(loss_real)
			loss += 0.05*loss_real
		if self.opts.id_lambda > 0:
			loss_id,id_logs = self.id_loss(y_hat_crop,y_crop,cropped=True)
			loss_dict['loss_id'] = float(loss_id)
			loss += loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			y_hat_mean=y_hat_crop.mean(dim=(2,3))
			y_mean=y_crop.mean(dim=(2,3))
			loss_l2 = F.mse_loss(y_hat_mean, y_mean)
			loss_dict['loss_l2_2'] = float(loss_l2)
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop
		if self.opts.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.opts.moco_lambda
		if need_au:
			new_au=self.D(y_hat_crop)
			loss_au=F.mse_loss(tar_au, new_au)
			loss_dict["loss_au"]=float(loss_au)
			loss+=0.15*loss_au
		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def gradient_penalty(self, y, x):
		"""Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
		weight = torch.ones(y.size()).cuda()
		dydx = torch.autograd.grad(outputs=y,
								   inputs=x,
								   grad_outputs=weight,
								   retain_graph=True,
								   create_graph=True,
								   only_inputs=True)[0]

		dydx = dydx.view(dydx.size(0), -1)
		dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
		return torch.mean((dydx_l2norm - 1) ** 2)
	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
		if self.opts.use_wandb:
			self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], self.opts),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
		else:
			path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict

	# @contextlib.contextmanager
	# def no_weight_gradients(self):
	# 	global weight_gradients_disabled
	# 	old = weight_gradients_disabled
	# 	weight_gradients_disabled = True
	# 	yield
	# 	weight_gradients_disabled = old
	def g_path_regularize(self,fake_img, latents, mean_path_length, decay=0.01):
		noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
		grad, = autograd.grad(
			outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
		)
		path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

		path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

		path_penalty = (path_lengths - path_mean).pow(2).mean()

		return path_penalty, path_mean.detach(), path_lengths
	def g_nonsaturating_loss(self,fake_pred):
		loss = F.softplus(-fake_pred).mean()
		return loss
	def d_logistic_loss(self,real_pred, fake_pred):
		real_loss = F.softplus(-real_pred)
		fake_loss = F.softplus(fake_pred)
		return real_loss.mean() + fake_loss.mean()

	def d_r1_loss(self,real_pred, real_img):
		grad_real, = autograd.grad(
				outputs=real_pred.sum(), inputs=real_img, create_graph=True)
		grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

		return grad_penalty



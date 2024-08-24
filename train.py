import argparse
import os
from math import log10
import time
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import Dataset
from matplotlib import pyplot as plt
import torch
import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=78, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
batch_size = 8

train_set = TrainDatasetFromFolder('data/VOC2012/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
val_set = ValDatasetFromFolder('data/VOC2012/val', upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

num_train_batches = len(train_loader)
num_val_batches = len(val_loader)

netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(netG.parameters(), lr = 0.0004)
optimizerD = optim.Adam(netD.parameters(), lr = 0.0004)

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
end_epoch_times = []
train_epoch_times = []
standard_epoch_times =[]

for epoch in range(1, NUM_EPOCHS + 1):
    
    train_bar = tqdm(train_loader)
    start_time = time.time()
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    d_loss_value = []
    g_loss_value = []
    d_score_value = []
    g_score_value = []
    
    netG.train()
    netD.train()
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)

        netD.zero_grad()
        
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss_real = -torch.log(real_out)
        d_loss_fake = -torch.log(1 - fake_out)
        d_loss = d_loss_real + d_loss_fake

        
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()

        g_loss = generator_criterion(fake_out, fake_img, real_img)
        running_results['g_loss'] += g_loss.item() * batch_size
        d_loss = 1 - real_out + fake_out
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))
        
        d_loss_value.append(running_results['d_loss'] / running_results['batch_sizes'])
        g_loss_value.append(running_results['g_loss'] / running_results['batch_sizes'])
        d_score_value.append(running_results['d_score'] / running_results['batch_sizes'])
        g_score_value.append(running_results['g_score'] / running_results['batch_sizes'])

    train_time = time.time()
    train_epoch_duration = train_time - start_time
    train_epoch_times.append({'epoch': epoch, 'duration': train_epoch_duration})
    print(f'Epoch {epoch} training time is {train_epoch_duration:.2f} seconds')
    df = pd.DataFrame(train_epoch_times)
    df.to_csv('times/train_time.csv', index=False)
                 
    netG.eval()
    
    epochs = range(1, len(d_loss_value) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, d_loss_value, 'bo-', label='d_loss_value')
    plt.title('d_loss_value')
    plt.xlabel('batches')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('1.png')
    plt.show()

    epochs = range(1, len(g_loss_value) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, g_loss_value, 'bo-', label='g_loss_value')
    plt.title('g_loss_value')
    plt.xlabel('batches')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('2.png')
    plt.show()

    epochs = range(1, len(d_score_value) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, d_score_value, 'bo-', label='d_score_value')
    plt.title('d_score_value')
    plt.xlabel('batches')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.savefig('3.png')
    plt.show()

    epochs = range(1, len(g_score_value) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, g_score_value, 'bo-', label='g_score_value')
    plt.title('g_score_value')
    plt.xlabel('batches')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.savefig('4.png')
    plt.show()
    
    out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    for val_lr, val_hr_restore, val_hr in val_bar:
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size
        lr = Variable(val_lr, volatile=True)
        hr = Variable(val_hr, volatile=True)
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = netG(lr)

        batch_mse = ((sr - hr) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).item()
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        val_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))

        val_images.extend(
            [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
             display_transform()(sr.data.cpu().squeeze(0))])
    
    standard_caculation_time = time.time()
    standard_epoch_duration = standard_caculation_time - train_time
    standard_epoch_times.append({'epoch': epoch, 'duration': standard_epoch_duration})
    print(f'Epoch {epoch} standard caculation time is {standard_epoch_duration:.2f} seconds')
    df = pd.DataFrame(standard_epoch_times)
    df.to_csv('times/standard_time.csv', index=False) 
               
    if epoch % 5 == 0 and epoch != 0:
        val_images = torch.stack(val_images[:450])
        val_images = torch.chunk(val_images, val_images.size(0) // 15)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1
        for image in val_save_bar:
            try:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            except:
                pass
            index += 1    
    

    # save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    if epoch % 10 == 0 and epoch != 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    end_epoch_times.append({'epoch': epoch, 'duration': epoch_duration})
    print(f'Epoch {epoch} completed in {epoch_duration:.2f} seconds')
    df = pd.DataFrame(end_epoch_times)
    df.to_csv('times/end_time.csv', index=False)
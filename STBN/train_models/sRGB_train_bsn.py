import random
import sys

import numpy as np

sys.path.append('..')
import argparse
from datasets import SrgbTrainDataset, SrgbValDataset
import os
import time
import torch
from torch.utils.data import DataLoader
from utils.base_functions import resume_training, save_model
from utils.fastdvdnet_utils import fastdvdnet_batch_psnr, normalize_augment
from utils.io import log
from utils.post_processing import post_process, post_process_batch
from utils.loss_funtions import loss_function, loss_function_batch


torch.backends.cudnn.benchmark = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# setup_seed(1)

def main(**args):
    dataset_train = SrgbTrainDataset(seq_dir=args['trainset_dir'],
                                     train_length=args['train_length'],
                                     patch_size=args['patch_size'],
                                     patches_per_epoch=args['patches_per_epoch'],
                                     image_postfix=None,
                                     pin_memory=True)
    loader_train = DataLoader(dataset=dataset_train, batch_size=args['batch_size'], num_workers=4, shuffle=True, drop_last=True)
    dataset_val = SrgbValDataset(valsetdir=args['valset_dir'])
    loader_val = DataLoader(dataset=dataset_val, batch_size=1)

    model_module = __import__(f"models.{args['model']}", fromlist=[args['model']])
    model_class = getattr(model_module, args['model'])
    
    model = model_class(img_channels=3, num_resblocks=args['num_resblocks'])
    model = torch.nn.DataParallel(model).cuda()
    
    optimizer = torch.optim.Adam(model.module.trainable_parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args['milestones'], gamma=args['gamma'])
    start_epoch = resume_training(args, model, optimizer, scheduler)
    # for param_group in optimizer.param_groups:
    #     print(f"Learning rate: {param_group['lr']:5f}")
    for epoch in range(start_epoch, args['epochs']):
        start_time = time.time()
        # training
        model.train()
        for i, data in enumerate(loader_train):
            # start_time = time.time()

            seq = data['data'].cuda()
            seq = normalize_augment(seq, aug=False)

            stdn = torch.empty((seq.shape[0], 1, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1]) if len(
                args['noise_ival']) == 2 else torch.FloatTensor(args['noise_ival']).cuda()
            noise_level_map = stdn.expand_as(seq)

            noise = torch.normal(mean=torch.zeros_like(seq), std=noise_level_map).cuda()
            seqn = seq + noise
            # N2N
            seqdn, sigma = model(seqn, noise_level_map)
            sigma = stdn.squeeze() if sigma == None else sigma
            loss = loss_function_batch(seqdn, seqn, mode="loglike", sigma=sigma)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            optimizer.zero_grad()
            # e = time.time()-start_time
            # print(f'{e:.2f}, pred:{e*32000/4/60:.2f}min/ep')
            if (i+1) % args['print_every'] == 0:
                with torch.no_grad():
                    seqdn, seqdn_mean = post_process_batch(seqdn, seqn, sigma=sigma)
                train_psnr = fastdvdnet_batch_psnr(seq, seqdn)
                train_psnr_mask = fastdvdnet_batch_psnr(seq, seqdn_mean)
                log(args["log_file"], "[epoch {}][{}/{}] loss: {:1.4f} PSNR_train: {:1.4f} PSNR_train_m: {:1.4f}\n".
                    format(epoch + 1, i + 1, int(args['patches_per_epoch'] // args['batch_size']), loss.item(), train_psnr, train_psnr_mask))
            # break
        scheduler.step()

        # evaluating
        model.eval()
        save_model(args, model, optimizer, scheduler, epoch + 1)
        psnr_val = 0
        psnrm_val = 0
        for i, data in enumerate(loader_val):
            seq = data['seq'].cuda()
            # torch.manual_seed(0)
            stdn = torch.FloatTensor([args['val_noiseL']])
            noise_level_map = stdn.expand_as(seq)
            noise = torch.empty_like(seq).normal_(mean=0, std=args['val_noiseL'])
            seqn = seq + noise
            seqn = seqn.contiguous()

            with torch.no_grad():
                seqdn, sigma = model(seqn, noise_level_map)
                sigma = stdn.squeeze() if sigma == None else sigma
                seqdn, seqdn_mean = post_process_batch(seqdn.cpu(), seqn.cpu(), sigma=sigma.cpu())
                psnr = fastdvdnet_batch_psnr(seq, seqdn)

                log(args["log_file"], f"\n[epoch {epoch + 1}] id: {os.path.basename(data['name'][0])} PSNR: {psnr:.4f}")

                psnr_val += psnr
                psnrm_val += fastdvdnet_batch_psnr(seq, seqdn_mean)

        psnr_val = psnr_val / len(dataset_val)
        psnrm_val = psnrm_val / len(dataset_val)
        log(args["log_file"], "\n[epoch %d] PSNR_val: %.4f PSNR_val_m: %.4f, %0.2f hours/epoch\n\n" %
            (epoch + 1, psnr_val, psnrm_val, (time.time()-start_time)/3600))
        del psnr, psnrm_val, sigma, psnr_val, seqdn, seqdn_mean, seqn, seq, stdn, noise_level_map, noise
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    # Model parameters
    parser.add_argument("--model", type=str, default='bdc_2_unify')
    parser.add_argument("--num_resblocks", type=int, default=5)
    parser.add_argument("--train_length", type=int, default=8)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", "--e", type=int, default=10)
    parser.add_argument("--milestones", nargs='*', type=int, default=[2, 4, 6, 8])
    parser.add_argument("--gamma",  type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55])
    parser.add_argument("--val_noiseL", type=float, default=30)
    parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
    parser.add_argument("--patches_per_epoch", "--n", type=int, default=32000, help="Number of patches")

    # Paths
    parser.add_argument("--trainset_dir", type=str)
    parser.add_argument("--valset_dir", type=str)
    parser.add_argument("--log_dir", type=str)
    argspar = parser.parse_args()

    argspar.log_file = os.path.join(argspar.log_dir, 'log.out')
    argspar.train_length = argspar.train_length

    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    if argspar.noise_ival[0] == argspar.noise_ival[1]:
        argspar.noise_ival = [argspar.noise_ival[0] / 255.]
    else:
        argspar.noise_ival[0] /= 255.
        argspar.noise_ival[1] /= 255.

    if not os.path.exists(argspar.log_dir):
        os.makedirs(argspar.log_dir)
    log(argspar.log_file, "\n### Training the denoiser ###\n")
    log(argspar.log_file, "> Parameters:\n")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        log(argspar.log_file, '\t{}: {}\n'.format(p, v))

    main(**vars(argspar))

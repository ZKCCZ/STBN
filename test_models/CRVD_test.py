import sys
sys.path.append('..')
from models.rvidenet.isp import ISP
import argparse
from datasets import CRVDTestDataset
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
import torch
import torch.nn as nn



def test_big_size_raw(input_data, denoiser, patch_h=96, patch_w=96, patch_h_overlap=0, patch_w_overlap=64):

    H = input_data.shape[1]
    W = input_data.shape[2]
    print(H,W)
    test_result = torch.zeros((input_data.shape[0], H, W, 4))
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = torch.zeros((input_data.shape[0], patch_h, W, 4))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1)
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:, h_begin:h_end, w_begin:w_end, :]
            with torch.no_grad():
                test_patch_result = denoiser(test_patch.permute(0,3,1,2).unsqueeze(0)).squeeze(0).permute(0, 2, 3, 1).cpu()
            if w_index == 1:
                test_horizontal_result[:, :, w_begin:w_end, :] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:, :, w_begin+i, :] = test_horizontal_result[:, :, w_begin+i, :] * \
                        (patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:, :, i, :]*i/(patch_w_overlap-1)
                test_horizontal_result[:, :, w_begin+patch_w_overlap:w_end, :] = test_patch_result[:, :, patch_w_overlap:, :]
            w_index += 1

        test_patch = input_data[:, h_begin:h_end, -patch_w:, :]
        with torch.no_grad():
            test_patch_result = denoiser(test_patch.permute(0, 3, 1, 2).unsqueeze(0)).squeeze(0).permute(0, 2, 3, 1).cpu()
        last_range = w_end-(W-patch_w)
        for i in range(last_range):
            test_horizontal_result[:, :, W-patch_w+i, :] = test_horizontal_result[:, :, W-patch_w+i, :] * \
                (last_range-1-i)/(last_range-1)+test_patch_result[:, :, i, :]*i/(last_range-1)
        test_horizontal_result[:, :, w_end:, :] = test_patch_result[:, :, last_range:, :]

        if h_index == 1:
            test_result[:, h_begin:h_end, :, :] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:, h_begin+i, :, :] = test_result[:, h_begin+i, :, :] * \
                    (patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:, i, :, :]*i/(patch_h_overlap-1)
            test_result[:, h_begin+patch_h_overlap:h_end, :, :] = test_horizontal_result[:, patch_h_overlap:, :, :]
        h_index += 1

    test_horizontal_result = torch.zeros((input_data.shape[0], patch_h, W, 4))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_patch = input_data[:, -patch_h:, w_begin:w_end, :]
        with torch.no_grad():
            test_patch_result = denoiser(test_patch.permute(0, 3, 1, 2).unsqueeze(0)).squeeze(0).permute(0, 2, 3, 1).cpu()
        if w_index == 1:
            test_horizontal_result[:, :, w_begin:w_end, :] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:, :, w_begin+i, :] = test_horizontal_result[:, :, w_begin+i, :] * \
                    (patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:, :, i, :]*i/(patch_w_overlap-1)
            test_horizontal_result[:, :, w_begin+patch_w_overlap:w_end, :] = test_patch_result[:, :, patch_w_overlap:, :]
        w_index += 1

    test_patch = input_data[:, -patch_h:, -patch_w:, :]
    with torch.no_grad():
        test_patch_result = denoiser(test_patch.permute(0, 3, 1, 2).unsqueeze(0)).squeeze(0).permute(0, 2, 3, 1).cpu()
    last_range = w_end-(W-patch_w)
    for i in range(last_range):
        test_horizontal_result[:, :, W-patch_w+i, :] = test_horizontal_result[:, :, W-patch_w+i, :] * \
            (last_range-1-i)/(last_range-1)+test_patch_result[:, :, i, :]*i/(last_range-1)
    test_horizontal_result[:, :, w_end:, :] = test_patch_result[:, :, last_range:, :]

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:, H-patch_w+i, :, :] = test_result[:, H-patch_w+i, :, :] * \
            (last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:, i, :, :]*i/(last_last_range-1)
    test_result[:, h_end:, :, :] = test_horizontal_result[:, last_last_range:, :, :]

    return test_result


def reconstruct_image(out):

    B, C4, H2, W2 = out.shape
    C = C4 // 4 

    im = torch.zeros(B, C, H2 * 2, W2 * 2, device=out.device, dtype=out.dtype)

    im[:, :, 1::2, 0::2] = out[:, 0*C:1*C, :, :]  
    im[:, :, 1::2, 1::2] = out[:, 1*C:2*C, :, :]  
    im[:, :, 0::2, 0::2] = out[:, 2*C:3*C, :, :] 
    im[:, :, 0::2, 1::2] = out[:, 3*C:4*C, :, :]

    return im


def raw_ssim(pack1, pack2):
    test_raw_ssim = 0
    for i in range(4):
        test_raw_ssim += structural_similarity(pack1[i], pack2[i])
    return test_raw_ssim / 4


def denoise_seq(seqn, a, b, model):
    T, C, H, W = seqn.shape
    a = a.expand((1, T, 1, H, W)).cuda()
    b = b.expand((1, T, 1, H, W)).cuda()
    seqdn = model(seqn.unsqueeze(0))[0]
    return seqdn

def depack_gbrg_raw(raw):
    H = raw.shape[1]
    W = raw.shape[2]
    output = np.zeros((H*2, W*2))
    for i in range(H):
        for j in range(W):
            output[2*i, 2*j] = raw[0, i, j, 3]
            output[2*i, 2*j+1] = raw[0, i, j, 2]
            output[2*i+1, 2*j] = raw[0, i, j, 0]
            output[2*i+1, 2*j+1] = raw[0, i, j, 1]
    return output

def main(**args):
    dataset_val = CRVDTestDataset(CRVD_path=args['crvd_dir'])
    # dataset_val = CRVDTestDataset(CRVD_path=args['crvd_dir'])
    isp = ISP().cuda()
    isp.load_state_dict(torch.load(args['isp_path'])['state_dict'])

    model_module = __import__(f"models.{args['model']}", fromlist=[args['model']])
    model_class = getattr(model_module, args['model'])
    model = model_class(img_channels=4, num_resblocks=args['num_resblocks'])
    state_temp_dict = torch.load(args['model_file'])['state_dict']
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(state_temp_dict)
    model.eval()



    iso_psnr, iso_ssim = {}, {}
    for data in dataset_val:

        # our channels: RGGB, RViDeNet channels: RGBG. we must pass RGBG pack to ISP as it's pretrained by RViDeNet
        seq = data['seq'].cuda()[:, :, :, :]
        seqn = data['seqn'].cuda()[:, :, :, :]
        iso = data['iso']
        scene_id = data['scene_id']

        with torch.no_grad():
            seqdn = denoise_seq(seqn, data['a'], data['b'], model)

            seqn[:, 2:] = torch.flip(seqn[:, 2:], dims=[1])
            seqdn[:, 2:] = torch.flip(seqdn[:, 2:], dims=[1])
            seq[:, 2:] = torch.flip(seq[:, 2:], dims=[1])

        seq_raw_psnr, seq_srgb_psnr, seq_raw_ssim, seq_srgb_ssim = 0, 0, 0, 0
        for i in range(seq.shape[0]):
            print(i)
            gt_raw_frame = seq[i].cpu().numpy()
            denoised_raw_frame = (np.uint16(seqdn[i].cpu().numpy() * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (2 ** 12 - 1 - 240)
            seq_raw_psnr += compare_psnr(reconstruct_image(gt_raw_frame), reconstruct_image(denoised_raw_frame))
            seq_raw_ssim += raw_ssim(reconstruct_image(gt_raw_frame), reconstruct_image(denoised_raw_frame))

        seq_raw_psnr /= seq.shape[0]
        seq_raw_ssim /= seq.shape[0]

        if (str(data['iso'])+'raw') not in iso_psnr.keys():
            iso_psnr[str(data['iso'])+'raw'] = seq_raw_psnr / 5
            iso_ssim[str(data['iso'])+'raw'] = seq_raw_ssim / 5
        else:
            iso_psnr[str(data['iso'])+'raw'] += seq_raw_psnr / 5
            iso_ssim[str(data['iso']) + 'raw'] += seq_raw_ssim / 5

    dataset_raw_psnr, dataset_srgb_psnr, dataset_raw_ssim, dataset_srgb_ssim = 0, 0, 0, 0
    for iso in [1600, 3200, 6400, 12800, 25600]:
        print('iso %d, raw: %6.4f/%6.4f' % (iso, iso_psnr[str(iso)+'raw'], iso_ssim[str(iso)+'raw'],))
        dataset_raw_psnr += iso_psnr[str(iso)+'raw']
        dataset_raw_ssim += iso_ssim[str(iso)+'raw']

    print('CRVD, raw: %6.4f/%6.4f, srgb: %6.4f/%6.4f' % (dataset_raw_psnr / 5, dataset_raw_ssim / 5, dataset_srgb_psnr / 5, dataset_srgb_ssim / 5))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    parser = argparse.ArgumentParser(description="test raw model")
    parser.add_argument("--model", type=str, default='BR_d_c_raw')
    parser.add_argument("--num_resblocks", type=int, default=5)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--crvd_dir", type=str)
    parser.add_argument("--isp_path", type=str)
    argspar = parser.parse_args()

    print("\n### Testing model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))

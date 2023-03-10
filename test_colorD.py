
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from models.GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from models.Unet import UNetModel
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from new_utils.msgms import MSGMSLoss
from utils.gms import MSGMS_Score
from utils.utils import denormalization
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage.segmentation import mark_boundaries
from skimage import morphology, measure
import matplotlib
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def ColorDifference(imgo,imgr):
    imglabo=cv2.cvtColor(imgo,cv2.COLOR_BGR2LAB)
    imglabr=cv2.cvtColor(imgr,cv2.COLOR_BGR2LAB)
    diff=(imglabr-imglabo)*(imglabr-imglabo)
    RD=diff[:,:,1]
    BD=diff[:,:,2]
    Result=RD+BD
    Result=cv2.blur(Result,(11,11))*0.001
    return Result



def test(args, model, diffusion, test_loader):
    model.eval()
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    msgms_score = MSGMS_Score()
    for (data, label, mask) in tqdm(test_loader):
        test_imgs.extend(data.cpu().numpy())
        gt_list.extend(label.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())
        score = 0
        with torch.no_grad():
            data = data.to(device)
            output = diffusion.forward_backward(model, data, see_whole_sequence=None,
                                                t_distance=args.sample_distance // 2)
            score += msgms_score(data, output) / (args.img_size ** 2)
        score = torch.squeeze(score, dim=1).cpu().numpy()
        # score = score.squeeze().cpu().numpy()
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=7)
            recimg = output.detach().cpu().numpy()[i]
            recimg = np.transpose(recimg, (1, 2, 0)) * 180
            recimg = recimg.astype('uint8')
            oriimg = data.detach().cpu().numpy()[i]
            oriimg = np.transpose(oriimg, (1, 2, 0)) * 180
            oriimg = oriimg.astype('uint8')
            # color

            colorD = ColorDifference(recimg, oriimg)

            # combined
            score[i] = 0.00001*colorD + score[i]


        scores.extend(score)
        recon_imgs.extend(output.cpu().numpy())
    return scores, test_imgs, recon_imgs, gt_list, gt_mask_list

def plot_fig(args, test_img, recon_imgs, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        recon_img = recon_imgs[i]
        recon_img = denormalization(recon_img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('Predicted mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, args.obj + '_{}_png'.format(i)), dpi=100)
        plt.close()


def main(args):
    # 设置随机数
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    checkpoint = '/True' if args.checkpoint else '/False'
    # 设置保存文件夹
    args.save_dir = './save_dir/' + args.data_type + '/' + args.obj + checkpoint + \
                    '/' + args.noise_fn + '/seed_{}'.format(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

        # 加载数据集
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    test_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # 定义与加载模型
    ## 参数定义
    if args.channel_mults == "":
        if args.img_size == 512:
            args.channel_mults = (0.5, 1, 1, 2, 2, 4, 4)
        elif args.img_size == 256:
            args.channel_mults = (1, 1, 2, 2, 4, 4)
        elif args.img_size == 128:
            args.channel_mults = (1, 1, 2, 3, 4)
        elif args.img_size == 64:
            args.channel_mults = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {args.img_size}")
    else:
        args.channel_mults = tuple(int(ch_mult) for ch_mult in args.channel_mults.split(","))

    attention_resolutions = '32,16,8'
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(args.img_size // int(res))

    # 加载模型
    model = UNetModel(
        image_size=args.img_size,
        in_channels=args.in_channels,
        model_channels=args.model_channels,
        out_channels=(3 if not args.learn_sigma else 6),
        num_res_blocks=2,
        attention_resolutions=tuple(attention_ds),
        dropout=args.dropout,
        channel_mult=args.channel_mults,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    #state_dict = torch.load(f'./models/checkpoint/256x256_diffusion_uncond.pt')
    #model.load_state_dict(state_dict)
    state_dict = torch.load(f'{args.save_dir}/save_model/model_best.pt')
    model.load_state_dict(state_dict['ema'])
    model.to(device)
    betas = get_beta_schedule(args.T, args.beta_schedule)

    diffusion = GaussianDiffusionModel(
        args.img_size, betas, loss_weight=args.loss_weight,
        loss_type=args.loss_type, noise=args.noise_fn, img_channels=args.in_channels
    )
    scores, test_imgs, recon_imgs, gt_list, gt_mask_list = test(args, model, diffusion, test_loader)
    scores = np.asarray(scores)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    with open(f'{args.save_dir}/test.txt', 'a') as f:
        print('image ROCAUC: %.3f' % (img_roc_auc), file=f)
    plt.figure()
    plt.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (args.obj, img_roc_auc))
    plt.legend(loc="lower right")

    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    with open(f'{args.save_dir}/test.txt', 'a') as f:
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc), file=f)

    plt.plot(fpr, tpr, label='%s pixel_ROCAUC: %.3f' % (args.obj, per_pixel_rocauc))
    plt.legend(loc="lower right")
    save_dir = args.save_dir + '/' + 'test/' + 'pictures_{:.4f}'.format(threshold)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, args.obj + '_roc_curve.png'), dpi=100)

    plot_fig(args, test_imgs, recon_imgs, scores, gt_mask_list, threshold, save_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='diffusion anomaly detection')
    # 数据集加载相关参数
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='./dataset/mvtec_anomaly_detection')
    parser.add_argument('--obj', type=str, default='bottle', help='bottle, cable, capsule, carpet, grid, '
                                                                  'hazelnut, leather, metal_nut, pill, screw, '
                                                                  'tile, toothbrush, transistor, wood, zipper')
    parser.add_argument('--batch_size', type=int, default=16)

    # 图像相关参数
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--in_channels', type=int, default=3, help='color or grayscale input image')

    # 模型相关参数
    parser.add_argument('--model_channels', type=int, default=256)
    parser.add_argument('--channel_mults', type=str, default='')
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_head_channels', type=int, default=64)
    parser.add_argument('--learn_sigma', type=bool, default=True, help='learn_sigma: True or False')

    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default="linear", help='linear or cosine')

    parser.add_argument('--loss_weight', type=str, default='none', help='prop t / uniform / None')
    parser.add_argument('--loss_type', type=str, default='hybrid', help='l2, l1, hybrid')
    parser.add_argument('--noise_fn', type=str, default='gauss', help='gauss, simplex_randParam, random, simplex')
    parser.add_argument('--train_start', type=bool, default=False)
    parser.add_argument('--sample_distance', type=int, default=1000)

    # 测试相关参数
    parser.add_argument('--seed', type=int, default=2023, help='manual seed')
    parser.add_argument('--checkpoint', type=bool, default=False, help='checkpoint: True or False')

    return parser.parse_args()


if __name__ == '__main__':

    CLASS_NAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
        'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    CLASS_NAMES1 = [
        'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
        'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    for i in CLASS_NAMES:
        torch.cuda.empty_cache()
        args = parse_args()
        args.obj = i
        main(args)

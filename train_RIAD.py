
import os
import argparse
import copy
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from utils.utils import gridify_output, update_ema_params
from utils.ssim import SSIM
from utils.funcs import PSNR, matplot_loss
from models.GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from models.Unet import UNetModel
import matplotlib.pyplot as plt
from matplotlib import animation

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# 训练过程中，更新完参数后，同步更新ema
def train(model,ema,diffusion,train_loader,optimiser,args,epoch,SSIM):
    model.train()
    vlb, psnr, _ssim = [], [], []
    mean_loss = []
    i = 0
    for (x, _, _) in tqdm(train_loader):
        optimiser.zero_grad()

        x = x.to(device)


         # generator mask
        k_value = random.sample(args.k_value, 1)
        Ms_generator = gen_mask(k_value, 3, args.img_size)
        Ms = next(Ms_generator)

        inputs = [x * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
        outputs = [model(x) for x in inputs]
        output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))

        loss, estimates = diffusion.p_loss(model, x, args)

        # x_t以及预测的噪声
        x_t, est = estimates[1], estimates[2]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimiser.step()

        update_ema_params(ema, model)
        mean_loss.append(loss.data.cpu())

        if epoch % 200 == 0 and i == 1:
            row_size = min(8, args.batch_size)
            outputs(diffusion, x, est, x_t, epoch, row_size, save_imgs=args.save_imgs, save_vids=args.save_vids,
                    model=model, args=args, is_train=True)
        if epoch % 200 == 0:
            vlb_terms = diffusion.calc_total_vlb(x, model, args)
            vlb.append(vlb_terms)
            out = diffusion.forward_backward(model, x, see_whole_sequence=None, t_distance=args.sample_distance // 2)
            psnr.append(PSNR(out, x))
            _ssim.append(SSIM(out, x).detach().cpu().numpy())
        i += 1
    if epoch % 200 == 0:
        with open(f'{args.save_dir}/train.txt', 'a') as f:
            print('epoch = %d' % epoch, file=f)
            print(
                f"Train set total VLB: {np.mean([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +- "
                f"{np.std([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])}", file=f
            )
            print(
                f"Train set prior VLB: {np.mean([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +-"
                f" {np.std([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])}", file=f
            )
            print(
                f"Train set vb @ t=200: {np.mean([i['vlb'][0][199].cpu().item() for i in vlb])} "
                f"+- {np.std([i['vlb'][0][199].cpu().item() for i in vlb])}", file=f
            )
            print(
                f"Train set x_0_mse @ t=200: {np.mean([i['x_0_mse'][0][199].cpu().item() for i in vlb])} "
                f"+- {np.std([i['x_0_mse'][0][199].cpu().item() for i in vlb])}", file=f
            )
            print(
                f"Train set mse @ t=200: {np.mean([i['mse'][0][199].cpu().item() for i in vlb])}"
                f" +- {np.std([i['mse'][0][199].cpu().item() for i in vlb])}", file=f
            )
            print(f"Train set PSNR: {np.mean(psnr)} +- {np.std(psnr)}", file=f)
            print(f"Train set SSIM: {np.mean(_ssim)} +- {np.std(_ssim)}\n", file=f)
    return np.mean(mean_loss)

def val(ema,diffusion,val_loader,args,epoch,SSIM):
    ema.eval()
    vlb, psnr, _ssim = [], [], []
    mean_loss = []
    i = 0
    for (x, _, _) in tqdm(val_loader):
        x = x.to(device)
        with torch.no_grad():
            loss, estimates = diffusion.p_loss(ema, x, args)
            # x_t以及预测的噪声
            x_t, est = estimates[1], estimates[2]
            mean_loss.append(loss.data.cpu())

        if epoch % 200 == 0 and i == 1:
            row_size = min(8, args.batch_size)
            outputs(diffusion, x, est, x_t, epoch, row_size, save_imgs=args.save_imgs, save_vids=args.save_vids,
                    model=ema, args=args, is_train=False)
        if epoch % 200 == 0:
            vlb_terms = diffusion.calc_total_vlb(x, ema, args)
            vlb.append(vlb_terms)
            out = diffusion.forward_backward(ema, x, see_whole_sequence=None, t_distance=args.sample_distance // 2)
            psnr.append(PSNR(out, x))
            _ssim.append(SSIM(out, x).detach().cpu().numpy())
        i += 1
    if epoch % 200 == 0:
        with open(f'{args.save_dir}/train.txt', 'a') as f:
            print('epoch = %d' % epoch, file=f)
            print(
                f"Val set total VLB: {np.mean([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +- "
                f"{np.std([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])}", file=f
            )
            print(
                f"Val set prior VLB: {np.mean([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +-"
                f" {np.std([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])}", file=f
            )
            print(
                f"Val set vb @ t=200: {np.mean([i['vlb'][0][199].cpu().item() for i in vlb])} "
                f"+- {np.std([i['vlb'][0][199].cpu().item() for i in vlb])}", file=f
            )
            print(
                f"Val set x_0_mse @ t=200: {np.mean([i['x_0_mse'][0][199].cpu().item() for i in vlb])} "
                f"+- {np.std([i['x_0_mse'][0][199].cpu().item() for i in vlb])}", file=f
            )
            print(
                f"Val set mse @ t=200: {np.mean([i['mse'][0][199].cpu().item() for i in vlb])}"
                f" +- {np.std([i['mse'][0][199].cpu().item() for i in vlb])}", file=f
            )
            print(f"Val set PSNR: {np.mean(psnr)} +- {np.std(psnr)}", file=f)
            print(f"Val set SSIM: {np.mean(_ssim)} +- {np.std(_ssim)}\n", file=f)
    return np.mean(mean_loss)


def outputs(diffusion, x, est, x_t, epoch, row_size, model, args, save_imgs=False, save_vids=False, is_train=True):
    """
    Saves video & images based on args info
    :param diffusion: diffusion model instance
    :param x: x_0 real data value
    :param est: estimate of the noise at x_t (output of the model)
    :param x_t:
    :param epoch:
    :param row_size: rows for outputs into torchvision.utils.make_grid
    :param model: exponential moving average unet for sampling
    :param save_imgs: bool for saving imgs
    :param save_vids: bool for saving diffusion videos
    :param is_train: bool for saving train or val
    :return:
    """
    phase = 'train' if is_train else 'val'
    try:
        os.makedirs(f'{args.save_dir}/{phase}-diffusion-videos')
        os.makedirs(f'{args.save_dir}/{phase}-diffusion-images')
    except OSError:
        pass
    if save_imgs:
        if epoch % 100 == 0:
            # for a given t, output x_0, & prediction of x_(t-1), and x_0
            noise = torch.rand_like(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(model, x_t, t)
            out = torch.cat(
                    (x[:row_size, ...].cpu(), temp["sample"][:row_size, ...].cpu(),
                     temp["pred_x_0"][:row_size, ...].cpu())
                    )
            plt.title(f'real,sample,prediction x_0-{epoch}epoch')
        else:
            # for a given t, output x_0, x_t, & prediction of noise in x_t & MSE
            out = torch.cat(
                    (x[:row_size, ...].cpu(), x_t[:row_size, ...].cpu(), est[:row_size, ...].cpu(),
                     (est - x_t).square().cpu()[:row_size, ...])
                    )
            plt.title(f'real,noisy,noise prediction,mse-{epoch}epoch')
        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size), cmap='gray')
        plt.savefig(f'{args.save_dir}/{phase}-diffusion-images/EPOCH={epoch}.png')
        plt.clf()
    if save_vids:
        fig, ax = plt.subplots()
        if epoch % 50 == 0:
            plt.rcParams['figure.dpi'] = 200
            if epoch % 100 == 0:
                s = 2
            else:
                s = 4
            out = diffusion.forward_backward(model, x, "half", args.sample_distance // s, denoise_fn=args.noise_fn)
            imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
            ani = animation.ArtistAnimation(
                    fig, imgs, interval=30, blit=True,
                    repeat_delay=1000, repeat=False
                    )

            ani.save(f'{args.save_dir}/{phase}-diffusion-videos/EPOCH={epoch}_t={args.sample_distance // s}.gif')
            plt.rcParams['figure.dpi'] = 150
            plt.grid(False)
            plt.imshow(gridify_output(out[-1], row_size), cmap='gray')
            plt.savefig(f'{args.save_dir}/{phase}-diffusion-videos/EPOCH={epoch}_t={args.sample_distance // s}.png')
            plt.clf()

    plt.close('all')

def save(unet, optimiser, args, ema, epoch=1, final=False):
    """
    Save model
    :param unet: unet instance
    :param optimiser: ADAM optim
    :param args: model parameters
    :param ema: ema instance
    :param epoch: epoch for checkpoint
    :return: saved model
    """
    try:
        os.makedirs(f'{args.save_dir}/save_model')

    except OSError:
        pass
    if final:
        torch.save(
            {
                'n_epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                "args": args,
                "ema": ema.state_dict(),
            }, f'{args.save_dir}/save_model/model_last.pt'
        )
    else:
        torch.save(
                {
                    'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    }, f'{args.save_dir}/save_model/model_best.pt'
                )

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
    train_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size)
    img_nums = len(train_dataset)
    valid_num = int(img_nums * args.validation_ratio)
    train_num = img_nums - valid_num
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, **kwargs)

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
    if args.checkpoint:
        state_dict = torch.load(f'./models/checkpoint/256x256_diffusion_uncond.pt')
        model.load_state_dict(state_dict)
    ema = copy.deepcopy(model)
    model.to(device)
    ema.to(device)

    betas = get_beta_schedule(args.T, args.beta_schedule)

    diffusion = GaussianDiffusionModel(
        args.img_size, betas, loss_weight=args.loss_weight,
        loss_type=args.loss_type, noise=args.noise_fn, img_channels=args.in_channels
    )

    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    # 即每个epoch都衰减lr = lr * gamma,即进行指数衰减
    scheduler = lr_scheduler.ExponentialLR(optimiser, gamma=0.96)
    train_losses, val_losses = [], []
    start_time = time.time()
    ssim = SSIM()
    # dataset loop
    for epoch in range(1, args.epochs + 1):
        loss = train(model, ema, diffusion, train_loader, optimiser, args, epoch, ssim)
        train_losses.append(loss)

        loss = val(ema, diffusion, val_loader, args, epoch, ssim)
        if epoch > 50 and loss < min(val_losses):
            save(unet=model, args=args, optimiser=optimiser, ema=ema, epoch=epoch, final=False)
        val_losses.append(loss)
        scheduler.step()
        if epoch % 20 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args.epochs - epoch
            time_per_epoch = time_taken / (epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)
            print(
                f"{args.obj}:time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"est time remaining: {hours}:{mins:02.0f}\r"
            )
    save(unet=model, args=args, optimiser=optimiser, ema=ema, epoch=args.epochs, final=True)
    matplot_loss(train_losses, val_losses, args)

def parse_args():
    parser = argparse.ArgumentParser(description='diffusion anomaly detection')
    # 数据集加载相关参数
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='./dataset/mvtec_anomaly_detection')
    parser.add_argument('--obj', type=str, default='bottle', help='bottle, cable, capsule, carpet, grid, '
                                                                  'hazelnut, leather, metal_nut, pill, screw, '
                                                                  'tile, toothbrush, transistor, wood, zipper')
    parser.add_argument('--batch_size', type=int, default=2)

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
    parser.add_argument('--train_start', type=bool, default=True)
    parser.add_argument('--sample_distance', type=int, default=800)

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100, help='maximum training epochs')
    parser.add_argument('--validation_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.00002, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=2023, help='manual seed')
    parser.add_argument('--checkpoint', type=bool, default=True, help='checkpoint: True or False')

    # 数据保存
    parser.add_argument('--save_imgs', type=bool, default=True, help='save: True or False')
    parser.add_argument('--save_vids', type=bool, default=True, help='save: True or False')
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

import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.Vnet_6DoF import VNet6DOF
from dataset_ct_point import CTPointDataset, CTPoseDataset
import nibabel as nib
from pytorch_msssim import ssim

import torch.nn.functional as F

@torch.no_grad()
def evaluate(net, valloader, mi_loss_fn):

    net.eval()
    total_mi = 0
    count = 0

    for batch in valloader:

        volume = batch['image'].cuda()
        gt_pose = batch['pose'].cuda()

        pred_pose = net(volume)

        pred_slice = extract_slice_torch(volume, pred_pose)
        gt_slice   = extract_slice_torch(volume, gt_pose)

        pred_slice = (pred_slice - pred_slice.min())/(pred_slice.max()-pred_slice.min()+1e-6)
        gt_slice   = (gt_slice - gt_slice.min())/(gt_slice.max()-gt_slice.min()+1e-6)

        mi = -mi_loss_fn(pred_slice, gt_slice)  # convert back

        total_mi += mi.item()
        count += 1

    net.train()
    return total_mi / count

class MutualInformation(torch.nn.Module):
    """
    Differentiable Mutual Information Loss
    """
    def __init__(self, num_bins=64):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, x, y):
        """
        x,y: [B,1,H,W]  in [0,1]
        """
        B = x.shape[0]

        x = x.view(B, -1)
        y = y.view(B, -1)

        mi_total = 0.0

        for b in range(B):
            xi = x[b]
            yi = y[b]

            joint_hist = torch.histc(
                xi * self.num_bins + yi,
                bins=self.num_bins,
                min=0,
                max=self.num_bins
            )

            joint_prob = joint_hist / torch.sum(joint_hist) + 1e-8

            px = torch.sum(joint_prob)
            py = torch.sum(joint_prob)

            mi = torch.sum(joint_prob * torch.log(joint_prob / (px * py)))

            mi_total += mi

        return -mi_total / B   # negative → loss


def pose_to_normal(pose):
    """
    pose: [B,6]
    return normal [B,3]
    """
    angles = pose[:, 3:6] * torch.pi   # 已归一化 [-1,1]

    normal = torch.stack([
        torch.cos(angles[:,0]),
        torch.cos(angles[:,1]),
        torch.cos(angles[:,2])
    ], dim=1)

    normal = F.normalize(normal, dim=1)
    return normal


def extract_slice_torch(volume, pose, size=128):
    """
    volume: [B,1,D,H,W]
    pose: [B,6]
    return slice [B,1,size,size]
    """

    B, _, D, H, W = volume.shape
    device = volume.device

    center = pose[:, 0:3]  # 已归一化
    normal = pose_to_normal(pose)

    # 构造 u,v 基
    tmp = torch.tensor([1.,0.,0.], device=device).expand(B,3)

    dot = torch.abs((tmp * normal).sum(1))
    mask = dot > 0.9
    tmp[mask] = torch.tensor([0.,1.,0.], device=device)

    u = torch.cross(normal, tmp, dim=1)
    u = F.normalize(u, dim=1)

    v = torch.cross(normal, u, dim=1)
    v = F.normalize(v, dim=1)

    # 网格
    coords = torch.linspace(-1,1,size,device=device)
    uu, vv = torch.meshgrid(coords, coords, indexing='ij')

    uu = uu[None,None]
    vv = vv[None,None]

    center = center.view(B,3,1,1)
    u = u.view(B,3,1,1)
    v = v.view(B,3,1,1)

    sample_points = center + u*uu + v*vv

    # grid_sample 需要 [-1,1]
    grid = sample_points.permute(0,2,3,1).unsqueeze(1)

    slice_img = F.grid_sample(
        volume,
        grid,
        mode='bilinear',
        align_corners=True
    )

    return slice_img.squeeze(2)

def save_as_niigz(data, save_path):
    """
    data: torch.Tensor | np.ndarray
    save_path: xxx.nii.gz
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    data = np.asarray(data, dtype=np.float32)

    # affine 用单位矩阵
    affine = np.eye(4)

    nii = nib.Nifti1Image(data[0], affine)
    nib.save(nii, save_path)


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/dof_project/3dct_point_dataset_DS4/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='vnet_3points_BS8_0.0001lr_0.1rotloss', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--num_classes', type=int,  default='3', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    # logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))


    net = VNet6DOF(
        n_channels=1,
        n_filters=16,
        normalization='batchnorm'
    )
    net = net.cuda()

    db_train = CTPoseDataset(
        data_root=args.root_path + '/data',
        label_root=args.root_path + '/ctpoint_label_voxel_revise_DS4',
        split='train'
    )

    os.makedirs("db_train_0", exist_ok=True)

    sample = db_train[0]

    # print("sample size:", len(sample))
    # print("sample size:", sample['pose'])


    # sample_image = sample['image'].unsqueeze(0).cuda()
    # sample_heatmap = sample['heatmap'].unsqueeze(0).cuda()
    # print("加载一个 sample 后显存占用 (MB):", torch.cuda.memory_allocated() / 1024**2)

    # #检查点坐标
    # for k, v in sample.items():
    #     try:
    #         save_as_niigz(v, f"db_train_0/{k}.nii.gz")
    #         print(f"[OK] saved {k}.nii.gz")
    #     except Exception as e:
    #         print(f"[SKIP] {k}: {e}")

    # exit()
    
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
        
    trainloader = DataLoader(
            db_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

    db_val = CTPoseDataset(
        data_root=args.root_path + '/data',
        label_root=args.root_path + '/ctpoint_label_voxel_revise_DS4',
        split='val'
    )

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    best_mi = -1e9
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    mi_loss_fn = MutualInformation().cuda()

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            # print(sampled_batch['image'].shape)
            # print(sampled_batch['pose'].shape)
            volume_batch = sampled_batch['image'].cuda()
            gt_pose = sampled_batch['pose'].cuda()
            
            
            pred_pose = net(volume_batch)

            pred_pose_real = pred_pose.clone()  # 不改原来的 tensor

            # translation
            pred_pose_real[:, 0:3] = pred_pose_real[:, 0:3] * 128.0

            # rotation
            pred_pose_real[:, 3:6] = pred_pose_real[:, 3:6] * 180.0

            # 打印
            # print("pred_pose_real:", pred_pose_real)
            # print("gt_pose_real:", gt_pose)  # 如果 gt_pose 已归一化，也可以乘回

            # -----------------------
            # pose loss (separated)
            # -----------------------
            pred_t = pred_pose[:, 0:3]
            pred_r = pred_pose[:, 3:6]

            gt_t = gt_pose[:, 0:3]
            gt_r = gt_pose[:, 3:6]

            trans_loss = F.mse_loss(pred_t, gt_t)
            rot_loss   = F.mse_loss(pred_r, gt_r)

            lambda_t = 1.0
            lambda_r = 0.1

            pose_loss = lambda_t * trans_loss + lambda_r * rot_loss

            # -----------------------
            # slice extraction
            # -----------------------
            pred_slice = extract_slice_torch(volume_batch, pred_pose)
            gt_slice   = extract_slice_torch(volume_batch, gt_pose)


            # -----------------------
            # MI loss
            # -----------------------
            # normalize slices to [0,1]
            pred_slice_n = (pred_slice - pred_slice.min()) / (pred_slice.max() - pred_slice.min() + 1e-6)
            gt_slice_n   = (gt_slice   - gt_slice.min())   / (gt_slice.max()   - gt_slice.min()   + 1e-6)

            mi_loss = mi_loss_fn(pred_slice_n, gt_slice_n)

            # -----------------------
            # total loss
            # -----------------------
            loss = pose_loss + 0.5 * mi_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            # writer.add_scalar('lr', lr_, iter_num)
            # writer.add_scalar('loss/loss', loss, iter_num)
            # writer.add_scalar('loss/pose', pose_loss, iter_num)
            # writer.add_scalar('loss/ssim', ssim_loss, iter_num)
            # writer.add_scalar('metric/ssim', ssim_val, iter_num)
            # logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 200 == 0:

                val_mi = evaluate(net, valloader, mi_loss_fn)

                print("VAL MI:", val_mi)

                if val_mi > best_mi:
                    best_mi = val_mi

                    save_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(net.state_dict(), save_path)

                    print("⭐ Saved BEST model")
                    
            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
import argparse
import torch
import os
import time
import imageio
import numpy as np
import torchvision.transforms as tfs
import sklearn.cluster as cls
import munch
import yaml

from model.snowflake import TDPNet
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from collections import defaultdict


from datasets.dataset_load import ShapeNet55
from datasets.mv_dataset import ShapeNetPoint
from metrics.evaluation_metrics import distChamferCUDA
from metrics.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from utils import visualize_point_clouds, shapenet_validate
from pointnet.model import PointNetfeat
from datasets.trainer_dataset import build_dataset

_transform = tfs.Compose([
    tfs.CenterCrop(550),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])

_transform_shape = tfs.Compose([
    tfs.CenterCrop(256),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])
def sphere_noise(batch, num_pts, device):
    with torch.no_grad():
        theta = 2 * np.pi * torch.rand(batch, num_pts, device=device)
        phi = torch.acos(1 - 2 * torch.rand(batch, num_pts, device=device))
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)
def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, 0.0001)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def calc_dcd(x, gt, alpha=40, n_lambda=0.1, return_raw=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res
def main(conf):
    # Load 3D Prototype features
    # if conf.prototypes_npy != 'NOF':
    #     proto_corpus = np.load(conf.prototypes_npy)
    #     assert proto_corpus.shape[0] == conf.num_prototypes
    # else:
    #     if not conf.reclustering:
    #         raise RuntimeError('Prototypes are not provided, must re-clustering or train from scratch')
    #
    #     proto_corpus = np.zeros((conf.num_prototypes, 1024), dtype=np.float)

    # Basic setting, make checkpoint folder, initialize dataset, dataloaders and models
    conf.name = 'snowflakedcd'
    checkpoint_path = os.path.join(conf.model_path, conf.name)
    checkpoint_imgs = os.path.join(checkpoint_path, 'images')
    train_log = os.path.join(checkpoint_path, 'train.log')

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.exists(checkpoint_imgs):
        os.mkdir(checkpoint_imgs)

    root, ply_root, tgt_category = conf.root, conf.proot, conf.cat
    tgt_category = tgt_category

    if conf.dataset == 'shapenet':
        # mv_ds = ShapeNet55(root, 'train', transform=_transform_shape)
        # mv_ds_test = ShapeNet55(root, 'test', transform=_transform_shape, number_of_view=12)
        config_path = conf.config
        args = munch.munchify(yaml.safe_load(open(config_path)))
        ds_loader, ds_loader_test = build_dataset(args)
    else:
        raise RuntimeError(f'Dataset is suppose to be [modelnet|shapenet], but {conf.dataset} is given')

    # ds_loader = DataLoader(mv_ds, batch_size=conf.batch_size, drop_last=True, shuffle=True)
    # ds_loader_test = DataLoader(mv_ds_test, batch_size=conf.batch_size)
    # num_classes = len(mv_ds.classes)

    # print(f'Dataset summary : Categories: {mv_ds.classes} with length {len(mv_ds)}')
    # print(f'Num of classes is {len(mv_ds.classes)}')
    with open(train_log, 'a+') as f:
        f.write('Dataset summary :  length {}\n'.format(len(ds_loader)) )
    #     f.write('Num of classes is {}\n'.format(len(mv_ds.classes)))


    # Initialize Model
    model = TDPNet(conf)
    # point_feat_extractor = PointNetfeat()  # Required for re-clustering
    # print("reload model")
    # model.load_state_dict(torch.load(os.path.join(checkpoint_path, f'{conf.name}_iter_101.pt')))
    model.cuda()
    # point_feat_extractor.cuda()


    print('Start Training 2D to 3D -------------------------------------------')
    with open(train_log, 'a+') as f:
        f.write('Start Training 2D to 3D -------------------------------------------\n')

    optimizer = Adam(
        model.parameters(),
        lr=conf.lrate,
        betas=(.9, .999),
        weight_decay=0.00001
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(conf.nepoch / 3), gamma=.5)

    start_time = time.time()

    for i in range(conf.nepoch):
        total_loss = 0.
        print('Start Epoch {}'.format(str(i + 1)))
        with open(train_log, 'a+') as f:
            f.write('Start Epoch {}\n'.format(str(i + 1)))
        # if i == min(50, int(conf.nepoch / 3)):
        #     print('Activated prototype finetune')
        #     with open(train_log, 'a+') as f:
        #         f.write('Activated prototype finetune\n')
        #     model.activate_prototype_finetune()
        # idx = 0
        for idx, data in enumerate(ds_loader):
            # idx+=1
            image = data['image'].cuda()
            pc = data['points'].cuda()
            # print(image.size())
            # print(pc.size())

            # Optimize process
            optimizer.zero_grad()
            # noise = sphere_noise(conf.batch_size, num_pts=2048, device=conf.device)

            svr_pc = model(image)

            corse_pc = svr_pc[0]
            # print("corse:",corse_pc.size())
            choice = np.random.choice(2048, 512)
            down_pc = pc[:, choice, :]
            # print("down:" ,down_pc.size())

            # gen2gr_corse, gr2gen_corse = distChamferCUDA(corse_pc, down_pc)
            # corse_loss = gen2gr_corse.mean(1) + gr2gen_corse.mean(1)
            corse_dcd = calc_dcd(corse_pc, down_pc)
            corse_loss = corse_dcd[0]

            syn_pc = svr_pc[-1]
            # print("fin:" ,syn_pc.size())
            # gen2gr, gr2gen = distChamferCUDA(syn_pc, pc)
            # fin_loss = gen2gr.mean(1) + gr2gen.mean(1)
            fin_dcd = calc_dcd(syn_pc, pc)
            fin_loss = fin_dcd[0]

            cd_loss = corse_loss + fin_loss

            # dcd = calc_dcd(syn_pc, pc)
            # dcd_loss = dcd[1]

            loss = cd_loss.sum()

            total_loss += loss.detach().item()

            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print(
                    'Epoch %d Batch [%2d/%2d] Time [%3.2fs] Recon Nat %.10f' %
                    (i + 1, idx + 1, len(ds_loader), duration, loss.item() / float(conf.batch_size)))
                with open(train_log, 'a+') as f:
                    f.write('Epoch %d Batch [%2d/%2d] Time [%3.2fs] Recon Nat %.10f\n' %
                    (i + 1, idx + 1, len(ds_loader), duration, loss.item() / float(conf.batch_size)))

        print('Epoch {}  -- Recon Nat {}'.format(str(i + 1), total_loss / float(len(ds_loader))))
        with open(train_log, 'a+') as f:
            f.write('Epoch {}  -- Recon Nat {}\n'.format(str(i + 1), total_loss / float(len(ds_loader))))

        # Save model configuration
        if conf.save_interval > 0 and i % opt.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_path,
                                                        '{0}_iter_{1}_2.pt'.format(conf.name, str(i + 1))))

        # Validate the model on test split
        if conf.sample_interval > 0 and i % opt.sample_interval == 0:
            with torch.no_grad():
                cd, emd = shapenet_validate(model, ds_loader_test)
                print("Chamfer Distance  :%s" % cd)
                print("Earth Mover Distance :%s" % emd)
                with open(train_log, 'a+') as f:
                    f.write("Chamfer Distance  :%s\n" % cd)
                    f.write("Earth Mover Distance :%s\n" % emd)

            model.train()

        scheduler.step()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=301, help='number of epochs to train for')
    parser.add_argument('--random_seed', action="store_true", help='Fix random seed or not')
    parser.add_argument('--lrate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=120, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=140, help='learning rate decay 2')
    parser.add_argument('--lr_decay_3', type=int, default=145, help='learning rate decay 2')
    parser.add_argument('--device', type=str, default='cuda', help='Gpu usage')
    parser.add_argument('--dim_template', type=int, default=2, help='Template dimension')

    # Data
    parser.add_argument('-c', '--config', help='path to config file', required=True,default='./cfgs/svr.yaml')
    parser.add_argument('--number_points', type=int, default=2048,
                        help='Number of point sampled on the object during training, and generated by atlasnet')
    parser.add_argument('--prototypes_npy', type=str, default='NOF', help='Path of the prototype npy file')

    # Save dirs and reload
    parser.add_argument('--name', type=str, default="0", help='training name')
    parser.add_argument('--dir_name', type=str, default="", help='name of the log folder.')

    # Network
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
    parser.add_argument('--nb_primitives', type=int, default=16, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SQUARE", choices=["SPHERE", "SQUARE"],
                        help='dim_out_patch')
    parser.add_argument('--number_points_eval', type=int, default=2048,
                        help='Number of points generated by atlasnet (rounded to the nearest squared number) ')
    parser.add_argument("--remove_all_batchNorms", action="store_true", help="Replace all batchnorms by identity")

    parser.add_argument('--bottleneck_size', type=int, default=512, help='dim_out_patch')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='dim_out_patch')
    parser.add_argument('--num_prototypes', type=int, default=8, help='Number of prototypes')
    parser.add_argument('--num_slaves', type=int, default=4, help='Number of slave mlps per prototype')

    # Loss
    parser.add_argument('--no_metro', action="store_true", help='Compute metro distance')

    # Additional arguments
    parser.add_argument('--root', type=str, required=True, help='The path of multi-view dataset')
    parser.add_argument('--proot', type=str, required=True, help='The path of corresponding pc dataset')
    parser.add_argument('--cat', type=str, required=True, help='Target category')
    parser.add_argument('--model_path', type=str, default='../results/c4')

    parser.add_argument('--sample_interval', type=int, default=10, help='The gap between each sampling process')
    parser.add_argument('--save_interval', type=int, default=10, help='The gap between each model saving')

    parser.add_argument('--from_scratch', action="store_true", help='Train the point_feature_extractor from scratch')
    parser.add_argument('--reclustering', action="store_true", help='Flag that controls the re-clustering behavior')
    parser.add_argument('--dataset', type=str, default='shapenet', help='The dataset to use, chose from [modelnet|shapenet]')

    opt = parser.parse_args()
    main(opt)

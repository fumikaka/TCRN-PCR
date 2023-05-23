import matplotlib
import torch

matplotlib.use('Agg')

import numpy as np
import os
import matplotlib.pyplot as plt
import torch.distributed as dist

from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from math import log, pi
from metrics.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from visualize import visualize_pointcloud, visualize_pointcloud_batch


# Most code of this file is borrowed from: https://github.com/stevenygd/PointFlow/blob/master/utils.py
def sphere_noise(batch, num_pts, device):
    with torch.no_grad():
        theta = 2 * np.pi * torch.rand(batch, num_pts, device=device)
        phi = torch.acos(1 - 2 * torch.rand(batch, num_pts, device=device))
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


def gaussian_noise(num, width, height):
    return np.random.normal(0.0, 1.0, size=(num, 1, width, height))


# Visualization
def visualize_point_clouds(pts, gtr, idx, pert_order=[0, 1, 2]):
    pts = pts.cpu().detach().numpy()[:, pert_order]
    gtr = gtr.cpu().detach().numpy()[:, pert_order]

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Sample:%s" % idx)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Ground Truth:%s" % idx)
    ax2.scatter(gtr[:, 0], gtr[:, 1], gtr[:, 2], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


# Original validate function
def validate(model, tloader, image_flag=False):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    for idx, (multi_view, pc, stat) in enumerate(tloader):
        mv = np.stack(multi_view, axis=1).squeeze(1)
        mv = torch.from_numpy(mv)

        multi_view = mv.cuda()

        tr_pc = pc.cuda()

        out_pc = model.reconstruct(multi_view, 2048)

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        all_sample.append(tr_pc)
        all_ref.append(out_pc)

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    result = compute_all_metrics(sample_pcs, ref_pcs, 64, accelerated_cd=True)
    result = {k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in result.items()}

    print("Chamfer Distance  :%s" % cd.item())
    print("Earth Mover Distance :%s" % emd.item())


def validate_shapenet(model, tloader, image_flag=False, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    for idx, (multi_view, pc, _) in enumerate(tloader):
        mv = np.stack(multi_view, axis=1).squeeze(axis=1)
        mv = torch.from_numpy(mv).float()
        mv = mv.cuda()

        tr_pc = pc.cuda()

        if old_version:
            out_pc = model(mv)
        else:
            # out_pc = model(mv, label)       多类版本可能用到
            out_pc = model(mv)

        all_sample.append(tr_pc)
        all_ref.append(out_pc)

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    print("Chamfer Distance  :%s" % cd.item())
    print("Earth Mover Distance :%s" % emd.item())


# Modified validate function for TDPNet
def tdp_validate(model, tloader, image_flag=False, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    for idx, (multi_view, pc, stat, label) in enumerate(tloader):
        mv = np.stack(multi_view, axis=1).squeeze(axis=1)
        mv = torch.from_numpy(mv).float()
        mv = mv.cuda()

        tr_pc = pc.cuda()
        bs = pc.shape[0]

        if old_version:
            # out_pc = model(mv)
            # noise = sphere_noise(bs, num_pts=2048, device='cuda')
            syn_pc = model(mv)
            out_pc = syn_pc[-1]
            # out_pc = model(mv)
        else:
            out_pc = model(mv, label)

        all_sample.append(tr_pc)
        all_ref.append(out_pc)

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item()


def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0), dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, 0.0001)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res


def calc_dcd(x, gt, alpha=100, n_lambda=0.1, return_raw=False, non_reg=False):
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


# Modified validate function for TDPNet
def tdp_validate_dcd(model, tloader, image_flag=False, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list, dcd_list = list(), list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    for idx, (multi_view, pc, stat, label) in enumerate(tloader):
        mv = np.stack(multi_view, axis=1).squeeze(axis=1)
        mv = torch.from_numpy(mv).float()
        mv = mv.cuda()

        tr_pc = pc.cuda()
        bs = pc.shape[0]

        if old_version:
            # out_pc = model(mv)
            noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(mv, noise)
            # out_pc = model(mv)
        else:
            out_pc = model(mv, label)

        all_sample.append(tr_pc)
        all_ref.append(out_pc)

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        dcd_batch = calc_dcd(out_pc, tr_pc)
        # dcd_loss = dcd[0]
        dcd_list.append(dcd_batch[0])

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()
    dcd = torch.cat(dcd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item(), dcd.item()


# Modified validate function for TDPNet
def tdp_validate_npy(model, tloader, save_path, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    o = 0
    for idx, (multi_view, pc, stat, label) in enumerate(tloader):
        mv = np.stack(multi_view, axis=1).squeeze(axis=1)
        mv = torch.from_numpy(mv).float()
        mv = mv.cuda()

        tr_pc = pc.cuda()
        bs = pc.shape[0]

        if old_version:
            # out_pc = model(mv)
            noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(mv, noise)
            pc_cpu = out_pc.cpu()
            visualize_pointcloud(pc[0], out_file=os.path.join(save_path, '{0}-{1}_true.jpg'.format(o,o+11)), elev=10,
                                 azim=200)
            for i, p in enumerate(pc_cpu):
                visualize_pointcloud(p, out_file=os.path.join(save_path, '{0}_.jpg'.format( o)), elev=10,
                                     azim=200)

                # plt.savefig(multi_view[i::],os.path.join(save_path, '{0}_.jpg'.format( o))
                o += 1

            # if idx<=5:
            #     np.save("../pcs/test_{}.npy".format(idx),out_pc.cpu())
        else:
            out_pc = model(mv, label)

        all_sample.append(tr_pc)
        all_ref.append(out_pc)

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item()

def shapenet_validate(model, tloader, image_flag=False, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    for idx, data in enumerate(tloader):
        image = data['image'].cuda()
        tr_pc = data['points'].cuda()
        # print(image[0][0].mean())

        # tr_pc = pc.cuda()
        bs = tr_pc.shape[0]

        if old_version:
            # out_pc = model(mv)
            noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image,noise)
        else:
            # out_pc = model(mv, label)
            # out_pc = model(mv)
            # noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image)[-1]

        all_sample.append(tr_pc)
        all_ref.append(out_pc)
        # print(out_pc.size())
        # print(tr_pc.size())

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item()

def shapenet_validate_dense(model, tloader,img_path,npy_path, overlap=1, image_flag=False, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()
    o=0

    for idx, data in enumerate(tloader):
        image = data['image'].cuda()
        tr_pc = data['points'].cuda()
        # print(image[0][0].mean())
        pc=tr_pc.cpu()

        # tr_pc = pc.cuda()
        bs = tr_pc.shape[0]

        if old_version:
            # out_pc = model(mv)
            out_pc=None
            for count in range(overlap):
                noise = sphere_noise(bs, num_pts=2048, device='cuda')
                instance = model(image,noise)
                if count==0:
                    out_pc = instance
                else:
                    out_pc = torch.cat([out_pc, instance],dim=1)
            pc_cpu = out_pc.cpu()
            for i, p in enumerate(pc_cpu):
                visualize_pointcloud(p, out_file=os.path.join(img_path, '{0}_.jpg'.format(o)), elev=10,
                                     azim=200)
                visualize_pointcloud(pc[i], out_file=os.path.join(img_path, '{0}_true.jpg'.format(o)),
                                     elev=10,
                                     azim=200)
                np.save(os.path.join(npy_path, 'npy_file_{0}.npy'.format(o)), p)
                np.save(os.path.join(npy_path, 'npy_true_{0}.npy'.format(o)), pc[i])
                # print(p.size())
                # print(pc[i].size())

                # plt.savefig(multi_view[i::],os.path.join(save_path, '{0}_.jpg'.format( o))
                o += 1
            # out_pc = model(image,noise)
        else:
            # out_pc = model(mv, label)
            # out_pc = model(mv)
            # noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image)

        all_sample.append(tr_pc)
        all_ref.append(out_pc)
        # print(out_pc.size())
        # print(tr_pc.size())

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item()

def pix3d_validate(model, tloader, image_flag=False, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    for idx, (img,pc) in enumerate(tloader):
        image = img.float().cuda()
        tr_pc = pc.float().cuda()
        # print(type(image))
        # print(type(tr_pc))

        # tr_pc = pc.cuda()
        bs = tr_pc.shape[0]

        if old_version:
            # out_pc = model(mv)
            noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image,noise)
        else:
            # out_pc = model(mv, label)
            # out_pc = model(mv)
            # noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image)[-1]

        all_sample.append(tr_pc)
        all_ref.append(out_pc)
        # print(out_pc.size())
        # print(tr_pc.size())

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item()

def pix3d_validate_npy(model, tloader, img_path, npy_path, image_flag=False, old_version=True,category="no"):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()
    o=0

    for idx, (img,pc) in enumerate(tloader):
        image = img.float().cuda()
        tr_pc = pc.float().cuda()
        # print(type(image))
        # print(type(tr_pc))

        # tr_pc = pc.cuda()
        bs = tr_pc.shape[0]
        # o=0

        if old_version:
            # out_pc = model(mv)
            noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image,noise)
            pc_cpu = out_pc.cpu()
            # visualize_pointcloud(pc[0], out_file=os.path.join(img_path, '{0}-{1}_true.jpg'.format(o, o + 11)), elev=10,
            #                      azim=200)
            # np.save(os.path.join(npy_path, 'npy_file_true_{0}-{1}.npy'.format(o, o + 11)), pc[0])
            for i, p in enumerate(pc_cpu):

                visualize_pointcloud(p, out_file=os.path.join(img_path, '{0}_{1}_.jpg'.format(category,o)), elev=10,
                                azim=200)
                visualize_pointcloud(pc[i], out_file=os.path.join(img_path, '{0}_{1}_true.jpg'.format(category,o)), elev=10,
                                azim=200)
                np.save(os.path.join(npy_path, 'npy_file_{0}_{1}.npy'.format(category,o)), p)
                np.save(os.path.join(npy_path, 'npy_true_{0}_{1}.npy'.format(category, o)), pc[i])

                # plt.savefig(multi_view[i::],os.path.join(save_path, '{0}_.jpg'.format( o))
                o += 1
        else:
            # out_pc = model(mv, label)
            # out_pc = model(mv)
            # noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image)[-1]

            # noise = sphere_noise(bs, num_pts=2048, device='cuda')
            # out_pc = model(mv, noise)
            pc_cpu = out_pc.cpu()
            # visualize_pointcloud(pc[0], out_file=os.path.join(img_path, '{0}-{1}_true.jpg'.format(o, o + 11)), elev=10,
            #                      azim=200)
            # np.save(os.path.join(npy_path, 'npy_file_true_{0}-{1}.npy'.format(o, o + 11)), pc[0])
            for i, p in enumerate(pc_cpu):

                visualize_pointcloud(p, out_file=os.path.join(img_path, '{0}_{1}_.jpg'.format(category, o)),
                                    elev=10,
                                    azim=200)
                visualize_pointcloud(pc[i], out_file=os.path.join(img_path, '{0}_{1}_true.jpg'.format(category, o)),
                                    elev=10,
                                    azim=200)
                # np.save(os.path.join(npy_path, 'npy_file_{}.npy'.format(o)), p)
                np.save(os.path.join(npy_path, 'npy_file_{0}_{1}.npy'.format(category, o)), p)
                np.save(os.path.join(npy_path, 'npy_true_{0}_{1}.npy'.format(category, o)), pc[i])

                # plt.savefig(multi_view[i::],os.path.join(save_path, '{0}_.jpg'.format( o))
                o += 1

        all_sample.append(tr_pc)
        all_ref.append(out_pc)
        # print(out_pc.size())
        # print(tr_pc.size())

        # loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        # cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))
        dcd_batch = calc_dcd(out_pc, tr_pc)
        cd_list.append(dcd_batch[2])

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item()


def shapenet_validate_npy_newwork(model, tloader, img_path, npy_path, old_version=True,category=None):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    # all_sample = list()
    # all_ref = list()
    o = 0
    final_img_path = os.path.join(img_path, 'final/{}'.format(category))
    final_npy_path = os.path.join(npy_path, 'final/{}'.format(category))
    if not os.path.exists(final_img_path):
        os.mkdir(final_img_path)
    if not os.path.exists(final_npy_path):
        os.mkdir(final_npy_path)

    corse_img_path = os.path.join(img_path, 'corse/{}'.format(category))
    corse_npy_path = os.path.join(npy_path, 'corse/{}'.format(category))
    if not os.path.exists(corse_img_path):
        os.mkdir(corse_img_path)
    if not os.path.exists(corse_npy_path):
        os.mkdir(corse_npy_path)

    faise1_img_path = os.path.join(img_path, 'faise1/{}'.format(category))
    faise1_npy_path = os.path.join(npy_path, 'faise1/{}'.format(category))
    if not os.path.exists(faise1_img_path):
        os.mkdir(faise1_img_path)
    if not os.path.exists(faise1_npy_path):
        os.mkdir(faise1_npy_path)

    faise2_img_path = os.path.join(img_path, 'faise2/{}'.format(category))
    faise2_npy_path = os.path.join(npy_path, 'faise2/{}'.format(category))
    if not os.path.exists(faise2_img_path):
        os.mkdir(faise2_img_path)
    if not os.path.exists(faise2_npy_path):
        os.mkdir(faise2_npy_path)


    for idx, data in enumerate(tloader):

        image = data['image'].cuda()
        tr_pc = data['points'].cuda()
        # print(image[0][0].mean())

        # tr_pc = pc.cuda()
        bs = tr_pc.shape[0]

        if old_version:
            # out_pc = model(mv)

            noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image, noise)
            pc_cpu = out_pc.cpu()
            tp = tr_pc.cpu()
            for i, p in enumerate(pc_cpu):
                visualize_pointcloud(tp[i], out_file=os.path.join(img_path, '{0}_true.jpg'.format(o)),
                                     elev=10,
                                     azim=200)
                np.save(os.path.join(npy_path, 'npy_file_true_{0}.npy'.format(o)), tp[i])
                visualize_pointcloud(p, out_file=os.path.join(img_path, '{0}_.jpg'.format(o)), elev=10,
                                     azim=200)
                np.save(os.path.join(npy_path, 'npy_file_{}.npy'.format(o)), p)

                # plt.savefig(multi_view[i::],os.path.join(save_path, '{0}_.jpg'.format( o))
                o += 1
        else:
            # out_pc = model(mv, label)
            # out_pc = model(mv)
            # noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image)
            pc_cpu = out_pc[-1].cpu()
            # print(pc_cpu.size())      #2048
            pc_corse = out_pc[0].cpu()
            # print(pc_corse.size())      #512
            pc_faise1 = out_pc[1].cpu()
            # print(pc_faise1.size())      #512
            pc_faise2 = out_pc[2].cpu()
            # print(pc_faise2.size())        #1024
            tp = tr_pc.cpu()
            # print(tp.size())             #2048
            # visualize_pointcloud(tr_pc[0], out_file=os.path.join(img_path, '{0}-{1}_true.jpg'.format(o, o + 11)), elev=10,
            #                      azim=200)
            # np.save(os.path.join(npy_path, 'npy_file_true_{0}-{1}.npy'.format(o, o + 11)), tr_pc[0])
            for i, p in enumerate(pc_cpu):
                visualize_pointcloud(tp[i], out_file=os.path.join(final_img_path, '{0}_true.jpg'.format(o)),
                                     elev=10,
                                     azim=200)
                np.save(os.path.join(final_npy_path, 'npy_file_true_{0}.npy'.format(o)), tp[i])
                visualize_pointcloud(p, out_file=os.path.join(final_img_path, '{0}_.jpg'.format(o)), elev=10,
                                     azim=200)
                np.save(os.path.join(final_npy_path, 'npy_file_{}.npy'.format(o)), p)

                visualize_pointcloud(tp[i], out_file=os.path.join(corse_img_path, '{0}_true.jpg'.format(o)),
                                     elev=10,
                                     azim=200)
                visualize_pointcloud(pc_corse[i], out_file=os.path.join(corse_img_path, '{0}_.jpg'.format(o)), elev=10,
                                     azim=200)
                np.save(os.path.join(corse_npy_path, 'npy_file_{}.npy'.format(o)), pc_corse[i])
                np.save(os.path.join(corse_npy_path, 'npy_file_true_{0}.npy'.format(o)), tp[i])

                visualize_pointcloud(tp[i], out_file=os.path.join(faise1_img_path, '{0}_true.jpg'.format(o)),
                                     elev=10,
                                     azim=200)
                visualize_pointcloud(pc_faise1[i], out_file=os.path.join(faise1_img_path, '{0}_.jpg'.format(o)), elev=10,
                                     azim=200)
                np.save(os.path.join(faise1_npy_path, 'npy_file_{}.npy'.format(o)), pc_faise1[i])
                np.save(os.path.join(faise1_npy_path, 'npy_file_true_{0}.npy'.format(o)), tp[i])

                visualize_pointcloud(tp[i], out_file=os.path.join(faise2_img_path, '{0}_true.jpg'.format(o)),
                                     elev=10,
                                     azim=200)
                visualize_pointcloud(pc_faise2[i], out_file=os.path.join(faise2_img_path, '{0}_.jpg'.format(o)),
                                     elev=10,
                                     azim=200)
                np.save(os.path.join(faise2_npy_path, 'npy_file_{}.npy'.format(o)), pc_faise2[i])
                np.save(os.path.join(faise2_npy_path, 'npy_file_true_{0}.npy'.format(o)), tp[i])

                # plt.savefig(multi_view[i::],os.path.join(save_path, '{0}_.jpg'.format( o))
                o += 1

        # all_sample.append(tr_pc)
        # all_ref.append(out_pc)

        loss_1, loss_2 = distChamferCUDA(out_pc[-1], tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc[-1], tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    # sample_pcs = torch.cat(all_sample, dim=0)
    # ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item()

def shapenet_validate_npy(model, tloader, img_path, npy_path, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list = list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()
    o=0

    for idx, data in enumerate(tloader):

        image = data['image'].cuda()
        tr_pc = data['points'].cuda()
        # print(image[0][0].mean())

        # tr_pc = pc.cuda()
        bs = tr_pc.shape[0]




        if old_version:
            # out_pc = model(mv)

            noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image, noise)
            pc_cpu = out_pc.cpu()
            tp = tr_pc.cpu()
            for i, p in enumerate(pc_cpu):
                visualize_pointcloud(tp[i], out_file=os.path.join(img_path, '{0}_true.jpg'.format(o)),
                                     elev=10,
                                     azim=200)
                np.save(os.path.join(npy_path, 'npy_file_true_{0}.npy'.format(o)), tp[i])
                visualize_pointcloud(p, out_file=os.path.join(img_path, '{0}_.jpg'.format(o)), elev=10,
                                     azim=200)
                np.save(os.path.join(npy_path, 'npy_file_{}.npy'.format(o)), p)

                # plt.savefig(multi_view[i::],os.path.join(save_path, '{0}_.jpg'.format( o))
                o += 1
        else:
            # out_pc = model(mv, label)
            # out_pc = model(mv)
            # noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(image)
            pc_cpu = out_pc[-1].cpu()
            pc_corse = out_pc[0].cpu()
            pc_faise1 = out_pc[1].cpu()
            pc_faise2 = out_pc[2].cpu()
            tp = tr_pc.cpu()
            # visualize_pointcloud(tr_pc[0], out_file=os.path.join(img_path, '{0}-{1}_true.jpg'.format(o, o + 11)), elev=10,
            #                      azim=200)
            # np.save(os.path.join(npy_path, 'npy_file_true_{0}-{1}.npy'.format(o, o + 11)), tr_pc[0])
            for i, p in enumerate(pc_cpu):
                visualize_pointcloud(tp[i], out_file=os.path.join(img_path, '/final/{0}_true.jpg'.format(o)),
                                     elev=10,
                                     azim=200)
                np.save(os.path.join(npy_path, '/final/npy_file_true_{0}.npy'.format(o)), tp[i])
                visualize_pointcloud(p, out_file=os.path.join(img_path, '/final/{0}_.jpg'.format(o)), elev=10,
                                     azim=200)
                np.save(os.path.join(npy_path, '/final/npy_file_{}.npy'.format(o)), p)



                # plt.savefig(multi_view[i::],os.path.join(save_path, '{0}_.jpg'.format( o))
                o += 1

        all_sample.append(tr_pc)
        all_ref.append(out_pc)

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item()


def shapenet_validate_dcd(model, tloader, image_flag=False, old_version=True):
    from metrics.evaluation_metrics import emd_approx, distChamferCUDA, compute_all_metrics, \
        jsd_between_point_cloud_sets

    model.eval()
    cd_list, emd_list, dcd_list = list(), list(), list()
    ttl_samples = 0

    all_sample = list()
    all_ref = list()

    for idx, (multi_view, pc, _) in enumerate(tloader):
        mv = np.stack(multi_view, axis=1).squeeze(axis=1)
        mv = torch.from_numpy(mv).float()
        mv = mv.cuda()

        tr_pc = pc.cuda()
        bs = pc.shape[0]

        if old_version:
            # out_pc = model(mv)
            noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(mv, noise)
        else:
            # out_pc = model(mv, label)
            # out_pc = model(mv)
            noise = sphere_noise(bs, num_pts=2048, device='cuda')
            out_pc = model(mv, noise)

        all_sample.append(tr_pc)
        all_ref.append(out_pc)

        loss_1, loss_2 = distChamferCUDA(out_pc, tr_pc)
        cd_list.append(loss_1.mean(dim=1) + loss_2.mean(dim=1))

        emd_batch = emd_approx(out_pc, tr_pc)
        emd_list.append(emd_batch)

        dcd_batch = calc_dcd(out_pc, tr_pc)
        # dcd_loss = dcd[0]
        dcd_list.append(dcd_batch[0])

        ttl_samples += int(tr_pc.size(0))

    cd = torch.cat(cd_list).mean()
    emd = torch.cat(emd_list).mean()
    dcd = torch.cat(dcd_list).mean()

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    return cd.item(), emd.item(), dcd.item()


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from model import HCL
from data_loader import *
import argparse
import numpy as np
import torch
import random
import faiss
import sklearn.metrics as skm
import torch_geometric


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad'])
    parser.add_argument('-DS', help='Dataset', default='BZR')
    parser.add_argument('-DS_ood', help='Dataset', default='COX2')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0)

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)


def run_kmeans(x, args):

    results = {}

    d = x.shape[1]
    k = args.num_cluster
    clus = faiss.Clustering(d, k)
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 3

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False

    try:
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        clus.train(x, index)
    except:
        print('Fail to cluster with GPU. Try CPU...')
        index = faiss.IndexFlatL2(d)
        clus.train(x, index)

    D, I = index.search(x, 1)
    im2cluster = [int(n[0]) for n in I]

    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d

    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 30),
                           np.percentile(density, 70))
    density = density / density.mean() + 0.5

    centroids = torch.Tensor(centroids).cuda()
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results['centroids'] = centroids
    results['density'] = density
    results['im2cluster'] = im2cluster

    return results


def get_cluster_result(dataloader, model, args):
    model.eval()
    b_all = torch.zeros((n_train, model.embedding_dim))
    for data in dataloader:
        with torch.no_grad():
            data = data.to(device)
            b = model.get_b(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
            b_all[data.idx] = b.detach().cpu()
    cluster_result = run_kmeans(b_all.numpy(), args)
    return cluster_result


if __name__ == '__main__':
    setup_seed(0)
    args = arg_parse()

    if args.exp_type == 'ad':
        if args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
        else:
            splits = get_ad_split_TU(args, fold=args.num_trial)

    aucs = []
    for trial in range(args.num_trial):
        setup_seed(trial + 1)

        if args.exp_type == 'oodd':
            dataloader, dataloader_test, meta = get_ood_dataset(args)
        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_TU(args, splits[trial])

        dataset_num_features = meta['num_feat']
        n_train = meta['num_train']

        if trial == 0:
            print('================')
            print('Exp_type: {}'.format(args.exp_type))
            print('DS: {}'.format(args.DS_pair if args.DS_pair is not None else args.DS))
            print('num_features: {}'.format(dataset_num_features))
            print('num_structural_encodings: {}'.format(args.dg_dim + args.rw_dim))
            print('hidden_dim: {}'.format(args.hidden_dim))
            print('num_gc_layers: {}'.format(args.num_layer))
            print('================')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HCL(args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim+args.rw_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.num_epoch + 1):
            if args.is_adaptive:
                if epoch == 1:
                    weight_b, weight_g, weight_n = 1, 1, 1
                else:
                    weight_b, weight_g, weight_n = std_b ** args.alpha, std_g ** args.alpha, std_n ** args.alpha
                    weight_sum = (weight_b  + weight_g  + weight_n) / 3
                    weight_b, weight_g, weight_n = weight_b/weight_sum, weight_g/weight_sum, weight_n/weight_sum

            cluster_result = get_cluster_result(dataloader, model, args)

            model.train()
            loss_all = 0
            if args.is_adaptive:
                loss_b_all, loss_g_all, loss_n_all = [], [], []

            for data in dataloader:
                data = data.to(device)
                optimizer.zero_grad()
                b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                loss_g = model.calc_loss_g(g_f, g_s)
                loss_b = model.calc_loss_b(b, data.idx, cluster_result)
                loss_n = model.calc_loss_n(n_f, n_s, data.batch)
                if args.is_adaptive:
                    loss = weight_b * loss_b.mean() + weight_g * loss_g.mean() + weight_n * loss_n.mean()
                    loss_b_all = loss_b_all + loss_b.detach().cpu().tolist()
                    loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                    loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
                else:
                    loss = loss_b.mean() + loss_g.mean() + loss_n.mean()
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / n_train))

            if args.is_adaptive:
                mean_b, std_b = np.mean(loss_b_all), np.std(loss_b_all)
                mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
                mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)

            if epoch % args.eval_freq == 0:

                cluster_result_eval = get_cluster_result(dataloader, model, args)
                model.eval()

                y_score_all = []
                y_true_all = []
                for data in dataloader_test:
                    data = data.to(device)
                    b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                    y_score_b = model.scoring_b(b, cluster_result_eval)
                    y_score_g = model.calc_loss_g(g_f, g_s)
                    y_score_n = model.calc_loss_n(n_f, n_s, data.batch)
                    if args.is_adaptive:
                        y_score = (y_score_b - mean_b)/std_b + (y_score_g - mean_g)/std_g + (y_score_n - mean_n)/std_n
                    else:
                        y_score = y_score_b + y_score_g + y_score_n
                    y_true = data.y

                    y_score_all = y_score_all + y_score.detach().cpu().tolist()
                    y_true_all = y_true_all + y_true.detach().cpu().tolist()

                auc = skm.roc_auc_score(y_true_all, y_score_all)

                print('[EVAL] Epoch: {:03d} | AUC:{:.4f}'.format(epoch, auc))

        print('[RESULT] Trial: {:02d} | AUC:{:.4f}'.format(trial, auc))
        aucs.append(auc)

    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print('[FINAL RESULT] AVG_AUC:{:.4f}+-{:.4f}'.format(avg_auc, std_auc))
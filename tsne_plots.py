import sys
sys.path.append('..')
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data import ImageFolder
import numpy as np
from data import load_data, MEANS, STDS
from PIL import Image
import matplotlib.pyplot as plt
from torch import optim
from collections import defaultdict
import torchvision.datasets as datasets
import torch.nn.functional as F
from train import define_model
# from torchviz import make_dot
from skimage import io
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
from train import cudnn, train, define_model
import pickle

from sklearn.metrics import confusion_matrix

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE



def load_labels(args):
    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    else:
        file_list = './misc/class100.txt'
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))
    return class_labels

def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    return diffusion, vae

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def get_loader(args, return_path=False, mode_id_file=None):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform, nclass=args.nclass,
                        ipc=args.finetune_ipc, spec=args.spec, phase=args.phase,
                        seed=0, return_origin=True, return_path=return_path, mode_id_file=mode_id_file)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        # sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return loader


def mode_ids_to_csv(path_per_class, mode_id_per_class):
    keys = path_per_class.keys()
    image_ids = list()
    mode_label = list()
    class_ids = list()
    for k in keys:
        paths = path_per_class[k]
        mode_ids = mode_id_per_class[k]
        for p, mid in zip(paths, mode_ids):
            image_ids.append(p.split("/")[-1])
            mode_label.append(mid)
            class_ids.append(k)
    return pd.DataFrame({"image_id": image_ids, "class_id": class_ids, "mode_id": mode_label})


def get_features_per_class(args, loader, model):
    features_per_class = defaultdict(list)
    path_per_class = defaultdict(list)
    for img, lb, orig_labels, paths in tqdm(loader):

        if args.use_vae:
            features = model.encode(img.to(device)).latent_dist.mean * 0.18215
            features = features.detach().cpu()
        else:
            features = model.get_feature(img.to(device), 5, 5)[0]
            features = features.detach().cpu()
        # print(paths.shape)
        for feat, label, path in zip(features, lb, paths):
            features_per_class[label.item()].append(feat.flatten().numpy())
            path_per_class[label.item()].append(path)
    return features_per_class, path_per_class

def get_prediction_per_class(loader, model):
    features_per_class = defaultdict(list)
    path_per_class = defaultdict(list)
    for img, lb, orig_labels, paths in tqdm(loader):
        pred = model(img.to(device)).argmax()
        pred = pred.detach().cpu().item()

        features_per_class[lb.item()].append(pred)
        path_per_class[lb.item()].append(paths[0])
    return features_per_class, path_per_class


def train_model(args):
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True
    print(f"ImageNet directory: {args.imagenet_dir[0]}")
    _, train_loader, val_loader, nclass = load_data(args)

    best_acc_l = []
    acc_l = []
    # plotter = Plotter(args.save_dir, args.epochs, idx=i)
    model = define_model(args, nclass)

    best_acc, acc = train(args, model, train_loader, val_loader)
    best_acc_l.append(best_acc)
    acc_l.append(acc)

    print(f'\nBest, last acc: {np.mean(best_acc_l):.1f} {np.std(best_acc_l):.1f}')
    return model, np.mean(best_acc_l)



def main(args):
    original_data_path = args.imagenet_dir
    # original_finetune_ipc = args.finetune_ipc
    
    if args.use_vae:
        model = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    else:
        ckpt = torch.load("/home/jchan/MinimaxDiffusion/weights/model_best.pth.tar")
        args.net_type = ckpt['arch']
        model = define_model(args, args.nclass).to(device)
        model.load_state_dict(ckpt['state_dict'])
    model.eval()

    args.data_path = os.path.join(original_data_path[1], 'train')
    args.finetune_ipc = -1
    original_loader = get_loader(args, return_path=True)
    # CONDENSED_DATASET_PATH = original_data_path[0]

    args.data_path =  original_data_path[0]
    args.finetune_ipc = args.ipc
    condensed_loader = get_loader(args, return_path=True)

    args.imagenet_dir = original_data_path
    condensed_model, acc = train_model(args)

    original_features_per_class, original_paths = get_features_per_class(args, original_loader, model)
    condensed_features_per_class, _ = get_features_per_class(args, condensed_loader, model)
    cm_predictions_per_class, condensed_paths = get_prediction_per_class(original_loader, condensed_model)
    # exit()

    for c in original_paths.keys():
        for opath, cpath in zip(original_paths[c], condensed_paths[c]):
            if opath != cpath:
                print("not same")
                print(opath)
                print(cpath)
                print("Something wrong with the datasets")
                exit()

    original_features = list()
    original_labels = list()
    predicted_labels = list()
    for label, features in original_features_per_class.items():
        original_features += features
        original_labels += [label] * len(features)
        predicted_labels += cm_predictions_per_class[label]
    original_labels = np.array(original_labels)
    predicted_labels = np.array(predicted_labels)
    correct =  original_labels  == predicted_labels

    condensed_features = list()
    condensed_labels = list()
    for label, features in condensed_features_per_class.items():
        condensed_features += features
        condensed_labels += [label] * len(features)

    X_original = np.stack(original_features)
    X_condensed = np.stack(condensed_features)
    X_c = np.concatenate([X_original, X_condensed])
    tsne = TSNE(n_components=2, learning_rate='auto', random_state=42)
    X_embedded = tsne.fit_transform(X_c)

    N_original = X_original.shape[0]
    N_condensed = X_condensed.shape[0]


    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].scatter(x=X_embedded[:N_original, 0], y=X_embedded[:N_original, 1], s=10, cmap='tab10', c=original_labels, label="original", alpha=0.2)
    axes[0].scatter(x=X_embedded[N_original:, 0], y=X_embedded[N_original:, 1], s=50, cmap='tab10',  marker="X", c=condensed_labels, label="condensed", edgecolors='black', alpha=0.5)
    # axes[0].scatter(x=X_embedded[:N_original, 0], y=X_embedded[:N_original, 1], s=10, cmap='tab10', c="white", label="original", alpha=0.2)
    # axes[0].set_title("Correct")
    axes[0].set_xlabel("Model trained in full dataset")
    axes[0].legend()



    axes[1].scatter(x=X_embedded[:N_original, 0][correct], y=X_embedded[:N_original, 1][correct], s=10, cmap='tab10', c=original_labels[correct], label="original", alpha=0.2)
    axes[1].scatter(x=X_embedded[:N_original, 0][~correct], y=X_embedded[:N_original, 1][~correct], s=10, cmap='tab10', c="white", alpha=0.2)
    # axes[0].scatter(x=X_embedded[N_original:, 0], y=X_embedded[N_original:, 1], s=50, cmap='tab10',  marker="X", c=condensed_labels, label="condensed", edgecolors='black', alpha=0.2)
    axes[1].scatter(x=X_embedded[N_original:, 0], y=X_embedded[N_original:, 1], s=50, cmap='tab10',  marker="X", c=condensed_labels, label="condensed", edgecolors='black', alpha=0.5)
    axes[1].set_title("Correct")
    axes[1].set_xlabel("Condensed Model")
    axes[1].legend()

    axes[2].scatter(x=X_embedded[:N_original, 0][~correct], y=X_embedded[:N_original, 1][~correct], s=10, cmap='tab10', c="red", label="original", alpha=0.2)
    axes[2].scatter(x=X_embedded[:N_original, 0][correct], y=X_embedded[:N_original, 1][correct], s=10, cmap='tab10', c="white", alpha=0.2)
    axes[2].scatter(x=X_embedded[N_original:, 0], y=X_embedded[N_original:, 1], s=50, cmap='tab10',  marker="X", c=condensed_labels, label="condensed", edgecolors='black', alpha=0.5)
    axes[2].set_title("Incorrect")
    axes[2].set_xlabel("Condensed Model")
    axes[2].legend()
    fig.suptitle(f"Accuracy {np.round(acc, 2)}")
    plt.savefig(os.path.join(args.save_dir_plot, "condensed_plots.png"))
    plt.close()


    matrix = confusion_matrix(original_labels, predicted_labels)
    accuracy_per_class = matrix.diagonal()/matrix.sum(axis=1)



    IPC=args.ipc
    mode_id_per_class = defaultdict(list)
    clusters_centers = dict()


    for i, c in enumerate(tqdm(original_features_per_class.keys())):
        X = np.stack(original_features_per_class[c])
        kmeans = KMeans(n_clusters=IPC, random_state=0, n_init="auto").fit(X)
        kmeans_labels = kmeans.labels_
        mode_id_per_class[c] = kmeans_labels
        clusters_centers[c] = kmeans.cluster_centers_

    fig, ax = plt.subplots(2, 5, figsize=(25, 8))
    ax = ax.ravel()
    for i, c in enumerate(tqdm(original_features_per_class.keys())):
        X_original = np.stack(original_features_per_class[c])
        X_condensed = np.stack(condensed_features_per_class[c])
        X_c = np.concatenate([X_original, X_condensed, clusters_centers[c]])

        N_original = X_original.shape[0]
        N_condensed = X_condensed.shape[0]
        N_modes = IPC
        # X_mean = X.mean(0)
        # X_directions = compute_direction(X_mean, X)
        tsne = TSNE(n_components=2, learning_rate='auto', n_jobs=8, random_state=42)
        X_embedded = tsne.fit_transform(X_c)
        X_orginal_emb = X_embedded[:N_original]
        X_condensed_emb = X_embedded[N_original:N_original + N_condensed]
        X_mode_center = X_embedded[N_original + N_condensed:]
        ax[i].scatter(x=X_orginal_emb[:-IPC, 0], y=X_orginal_emb[:-IPC, 1], cmap = 'Accent', s=10, color="gray", label="original")
        ax[i].scatter(x=X_mode_center[-IPC:, 0], y=X_mode_center[-IPC:, 1], cmap = 'Accent', s=50,  marker="*",  color="red", label="Mode Center")
        ax[i].scatter(x=X_condensed_emb[-IPC:, 0], y=X_condensed_emb[-IPC:, 1], cmap = 'Accent', s=50,  marker="X",  color="blue", label="Condensed")
        ax[i].legend()
        ax[i].set_title(f"Class: {c}: acc = {np.round(accuracy_per_class[c],2) }")
    fig.suptitle(f"Accuracy {np.round(acc, 2)}")
    plt.savefig(os.path.join(args.save_dir_plot, "PlotsPerClass.png"))
    plt.close()


    

    


if __name__ == '__main__':
    from misc.utils import Logger
    from argument import args

    args.global_batch_size = 1
    args.image_size = 256

    # os.makedirs(args.save_dir, exist_ok=True)
    # logger = Logger(args.save_dir)
    # logger(f"Save dir: {args.save_dir_plot}")

    # os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir_plot, exist_ok=True)
    print(args)
    main(args)
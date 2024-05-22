"""
Sample new images from a pre-trained DiT.
"""
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
from tsne_plots import define_model, get_loader, get_features_per_class
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import pickle
from sklearn.decomposition import PCA
from torchvision import io
import time

def compute_distance(x1, x2):
    return torch.sqrt(torch.sum((x2-x1)**2))


def get_closest_distance(previous_batch, current_sample, cosine=False):

    if cosine:
        min_dist = -10000000
    else:
        min_dist = 10000000
    # min_z = 0
    if len(previous_batch) == 0:
        return 1
    for x in previous_batch:
        if cosine:
            # print(x.shape, x.reshape(1, -1))
            # print(current_sample.shape, current_sample.reshape(1, -1))
            curr_dist = torch.nn.functional.cosine_similarity(x.reshape(1, -1), current_sample.reshape(1, -1), dim=1, eps=1e-8).item()
            # print(curr_dist)
            if curr_dist > min_dist:
                min_dist = curr_dist
        else:
            curr_dist = compute_distance(x, current_sample)
            if curr_dist < min_dist:
                min_dist = curr_dist
    return min_dist




def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == 'imagenet100':
        file_list = './misc/class100.txt'
    elif args.spec == 'imagenet1k':
        file_list = './misc/class_indices.txt'
    else:
        raise ValueError("Invalid spec")
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
    # vae_ckpt = '/home/jchan/MinimaxDiffusion/logs/test/004-vae-vae_condensed_euclid_distance/checkpoints/0002900.pt'
    if args.vae_ckpt is not None:
        vae_ckpt = args.vae_ckpt
        ckpt = torch.load(vae_ckpt)
        vae.load_state_dict(ckpt['model'])

    batch_size = 1

    args.dataset = 'imagenet'
    args.net_type = 'resnet_ap'
    args.depth = 10
    args.width = 1.0
    args.norm_type = 'instance'
    args.nch = 3
    # args.imagenet_dir = ['/mnt/ImageNet']
    args.data_path = os.path.join(args.imagenet_dir,'train')
    args.size = 224
    args.aug_type = 'color_crop_cutout'
    args.augment = True
    args.dseed = 0
    args.global_batch_size = 1
    args.use_vae = True
    # original_finetune_ipc = args.finetune_ipc
    args.finetune_ipc = -1



    if args.cluster_file is None:
        original_loader = get_loader(args, return_path=True)
        original_features_per_class, original_paths = get_features_per_class(args, original_loader, vae)
        IPC=args.num_samples
        mode_id_per_class = defaultdict(list)
        clusters_centers = dict()
        clusters_centers_path = dict()

    
        if args.use_pca:
            for i, c in enumerate(tqdm(original_features_per_class.keys())):
                X = np.stack(original_features_per_class[c])
                pca = PCA(n_components=4)
                X_pca = pca.fit_transform(X)
                kmeans = KMeans(n_clusters=IPC, random_state=0, n_init="auto").fit(X_pca)
                kmeans_labels = kmeans.labels_
                mode_id_per_class[c] = kmeans_labels
                # get closest sample to centroids
                closest_points = list()
                closest_points_path = list()
                kmeans_centers = kmeans.cluster_centers_
                for center in kmeans_centers:
                    closest_point = X[np.argmin(np.sum((X_pca - center)**2, axis=1))]
                    center_path = original_paths[c][np.argmin(np.sum((X - center)**2, axis=1))]
                    closest_points.append(closest_point)
                    closest_points_path.append(center_path)
                clusters_centers[c] = np.stack(closest_points)
                clusters_centers_path[c] = closest_points_path

                

        else:
            for i, c in enumerate(tqdm(original_features_per_class.keys())):
                start_time = time.time()
                X = np.stack(original_features_per_class[c])
                kmeans = KMeans(n_clusters=IPC, random_state=0, n_init="auto").fit(X)
                kmeans_labels = kmeans.labels_
                mode_id_per_class[c] = kmeans_labels
                clusters_centers[c] = kmeans.cluster_centers_
                if args.closest_point:
                    closest_points = list()
                    closest_points_path = list()
                    for center in kmeans.cluster_centers_:
                        closest_point = X[np.argmin(np.sum((X - center)**2, axis=1))]
                        center_path = original_paths[c][np.argmin(np.sum((X - center)**2, axis=1))]
                        closest_points.append(closest_point)
                        closest_points_path.append(center_path)
                    clusters_centers[c] = np.stack(closest_points)
                    clusters_centers_path[c] = closest_points_path
                end_time = time.time()
                print(f"Time taken for class {c}: {end_time - start_time}")
    else:
        with open(args.cluster_file, "rb") as f:
            clusters_centers = pickle.load(f)

        if args.real:
            raise ValueError("Real data not supported with cluster file")
    print(clusters_centers.keys())



    
    if args.real:
        for i in range(args.num_datasets):
            save_dir = os.path.join(args.save_dir, f"dataset_{i}")
            # breakpoint()
            for class_label, sel_class in zip(class_labels, sel_classes):
                print(class_label, sel_classes)
                index = sel_classes.index(sel_class)
                os.makedirs(os.path.join(save_dir, sel_class), exist_ok=True)
                for shift in range(args.num_samples // batch_size):
                    
                    image_path = clusters_centers_path[index][shift]
                    image = io.read_image(image_path).float()/255.0
                    
                    image = image.unsqueeze(0)
                    save_image(image, os.path.join(save_dir, sel_class,
                                                f"{shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(0, 1))
        
        
    else:
        for i in range(args.num_datasets):
            save_dir = os.path.join(args.save_dir, f"dataset_{i}")
        
            noises = list()
            if args.use_same_noise:
                for i in range(args.num_samples // batch_size):
                    z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
                    noises.append(z)
            # breakpoint()
            for class_label, sel_class in zip(class_labels, sel_classes):
                print(class_label, sel_classes)
                index = sel_classes.index(sel_class)
                os.makedirs(os.path.join(save_dir, sel_class), exist_ok=True)
                previous_batch = list()
                start_time = time.time()
                for shift in tqdm(range(args.num_samples // batch_size)):
                    # Create sampling noise:
                    if args.use_same_noise:
                        z = noises[shift]
                    else:
                        z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
                    y = torch.tensor([class_label], device=device)

                    # Setup classifier-free guidance:
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * batch_size, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

                    # Sample images:
                    if args.guidance:
                        samples = diffusion.p_sample_loop_with_mode_guidance(
                            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device,
                            mode_features=torch.tensor(clusters_centers[index][shift].reshape(1, 4, 32, 32), device=device),
                            stop_t=args.stop_t, mode_guidance_scale=args.mode_guidance_scale)
                    else:
                        samples = diffusion.p_sample_loop(
                            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                        )
                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                    # min_dist = get_closest_distance(previous_batch, samples)
                    # print(f"Eucledean Distance")
                    # print(f"min_dist: {min_dist}")
                    # if min_dist != 0:
                    #     print(f"1/min_dist: {1/min_dist}")

                    # print(f"Cosine Distance")
                    # min_dist = get_closest_distance(previous_batch, samples, cosine=True)
                    # print(f"min_dist: {min_dist}")
                    # if min_dist != 0:
                    #     print(f"1/min_dist: {1/min_dist}")
                    # previous_batch.append(samples)
                    # breakpoint()
                    samples = vae.decode(samples / 0.18215).sample

                    # Save and display images:
                    for image_index, image in enumerate(samples):
                        save_image(image, os.path.join(save_dir, sel_class,
                                                    f"{image_index + shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(-1, 1))
                end_time = time.time()
                print(f"Time taken for class {sel_class}: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    parser.add_argument("--guidance", action="store_true", default=False, help="use guidance")
    parser.add_argument("--stop_t", type=int, default=0, help="stop t from 50 to 0")
    parser.add_argument("--mode_guidance_scale", type=float, default=0.1, help="guidance scale")
    parser.add_argument('--imagenet_dir', type=str, default='/ssd_data/imagenet/')
    parser.add_argument("--use_same_noise", action="store_true", default=False, help="use same noise across classes")
    parser.add_argument("--cluster_file", type=str, default=None, help="Use a cluster file")
    parser.add_argument("--num-datasets", type=int, default=5, help="number of generated datasets")
    parser.add_argument("--vae-ckpt", type=str, default=None, help="vae checkpoint")
    parser.add_argument("--use_pca", action="store_true", default=False, help="use pca for clustering")
    parser.add_argument("--closest_point", action="store_true", default=False, help="use closest point to cluster center")
    parser.add_argument("--real", action="store_true", default=False, help="use real data")
    args = parser.parse_args()
    main(args)

import torch

def stage1(diffusion, model, vae, img, label, device, args):
    with torch.no_grad():
        latents = vae.encode(img.to(device)).latent_dist.sample().mul_(0.18215)
    y = torch.tensor([label], device=device)
    y_null = torch.tensor([1000] * 1, device=device)
    y = torch.cat([y, y_null], 0)
    z = torch.cat([latents, latents], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    x_t = diffusion.ddim_reverse_sample_loop(model.forward_with_cfg, z.to(device), model_kwargs=model_kwargs, clip_denoised=True, progress=False, device=device)
    x_t, _ = x_t.chunk(2, dim=0)  # Remove null class samples
    return x_t

def stage2(diffusion, model, vae, x_t, label, device, args):
    y = torch.tensor([label], device=device)
    y_null = torch.tensor([1000] * 1, device=device)
    y = torch.cat([y, y_null], 0)
    x_t = torch.cat([x_t, x_t], 0).to(device)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    img = diffusion.ddim_sample_loop(
                    model.forward_with_cfg, x_t.shape, x_t, clip_denoised=True, model_kwargs=model_kwargs, progress=False, device=device
                )
    img, _ = img.chunk(2, dim=0)  # Remove null class samples
    img = vae.decode(img / 0.18215).sample
    return img

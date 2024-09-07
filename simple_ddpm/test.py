# -*- coding: utf-8 -*-
"""
@File:        test.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 27, 2024
@Description:
"""
import torch

import numpy as np


# from models import DDPM


def generate_images(model, n_samples, device, frame_pre_gif, gif_name, c=1, h=28, w=28):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = model.device

        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(model.n_steps))[::-1]):  # T 999 998 ~ 0
            time_tensor = (torch.ones() * t)

            # predict noise
            eta_theta = model.predict_noise(x, time_tensor)

            alpha_t = model.alphas[t]
            alpha_t_bar = model.alphas_bar[t]

            # denoise the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 1:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = model.betas[t]
                sigma_t = beta_t.sqrt()

                # option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                x = x + sigma_t * z
            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)
        # Storing the gif
        with imageio.get_writer(gif_name, mode="I") as writer:
            for idx, frame in enumerate(frames):
                # Convert grayscale frame to RGB
                rgb_frame = np.repeat(frame, 3, axis=-1)
                writer.append_data(rgb_frame)
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(rgb_frame)
        return x


if __name__ == '__main__':
    # Loading the trained model
    best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    print("Model loaded")

    # In[ ]:

    print("Generating new images")
    generated = generate_new_images(
        best_model,
        n_samples=100,
        device=device,
        gif_name="fashion.gif" if fashion else "mnist.gif"
    )
    show_images(generated, "Final result")

    # In[ ]:

    from IPython.display import Image

    Image(open('fashion.gif' if fashion else 'mnist.gif', 'rb').read())
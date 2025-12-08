import os
import numpy as np
import matplotlib.pyplot as plt

def show_all_projections(proj_dir):
    proj_files = sorted([f for f in os.listdir(proj_dir) if f.endswith('.npy')][:25])
    n_proj = len(proj_files)
    cols = int(np.ceil(np.sqrt(n_proj)))
    rows = int(np.ceil(n_proj / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()

    for idx, fname in enumerate(proj_files):
        arr = np.load(os.path.join(proj_dir, fname))
        if idx == 1:
            print(f"Shape of {fname}: {arr.shape}")
        axes[idx].imshow(arr, cmap='gray')
        axes[idx].set_title(fname)
        axes[idx].axis('off')

    # Hide unused axes
    for idx in range(n_proj, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    proj_dir = "data/synthetic_dataset/Case6_4DCT/Case6_T00_s_cone/proj_train"
    show_all_projections(proj_dir)



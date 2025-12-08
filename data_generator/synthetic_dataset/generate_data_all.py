import os
import os.path as osp
import glob
import argparse
import numpy as np
from scipy.ndimage import zoom



#===================================================================================#
#Extra code for padding function to pad volumes to (256,256,256) if needed
#===================================================================================#
def pad_volume(vol, target_shape=(256, 256, 256)):
    # Calculate required padding for each axis
    pad_x = target_shape[0] - vol.shape[0]
    pad_y = target_shape[1] - vol.shape[1]
    pad_z = target_shape[2] - vol.shape[2]
    # Pad at the end of each axis if needed
    pad_width = (
        (0, max(pad_x, 0)),
        (0, max(pad_y, 0)),
        (0, max(pad_z, 0))
    )
    #return np.pad(vol, pad_width, mode='constant')
    return vol # No padding applied

def stretch_volume(vol, target_shape=(256, 256, 256)):
    factors = (
        target_shape[0] / vol.shape[0],
        target_shape[1] / vol.shape[1],
        target_shape[2] / vol.shape[2]
    )
    vol_stretched = zoom(vol, factors, order=1)  # order=1: linear interpolation
    return vol_stretched
#===================================================================================#
#end of extra code
#===================================================================================#


def main(args):
    vol_dataset_path = args.vol
    output_path = args.output
    scanner_path = args.scanner
    n_train = args.n_train
    n_test = args.n_test
    device = args.device


    #===================================================================================#
    #EXTRA CODE TO CONVERT .img FILES TO .npy FILES IF NEEDED
    #===================================================================================#

    # Check for .img files and convert to .npy if needed
    img_file_paths = sorted(glob.glob(osp.join(vol_dataset_path, "*.img")))
    if img_file_paths:
        print(f"Found {len(img_file_paths)} .img files. Converting to .npy...")
        # Set your image shape and dtype here
        img_shape = (94, 256, 256)  # TODO: update to match your data
        dtype = np.uint16           # TODO: update to match your data
        for img_path in img_file_paths:
            arr = np.fromfile(img_path, dtype=dtype).reshape(img_shape)
            arr_padded = stretch_volume(arr, (256, 256, 256))
            npy_path = img_path.replace(".img", ".npy")
            np.save(npy_path, arr_padded)
            print(f"Converted {img_path} to {npy_path}")
    
    #===================================================================================#
    #END OF EXTRA CODE
    #===================================================================================#


    vol_file_paths = sorted(glob.glob(osp.join(vol_dataset_path, "*.npy")))

    if len(vol_file_paths) == 0:
        raise ValueError("{} find no *.npy file!".format(vol_file_paths))

    for vol_file_path in vol_file_paths:
        cmd = f"CUDA_VISIBLE_DEVICES={device} python data_generator/synthetic_dataset/generate_data.py --vol {vol_file_path} --scanner {scanner_path} --output {output_path}  --n_train {n_train} --n_test {n_test}"
        os.system(cmd)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol", default="data_generator/volume_gt", type=str, help="Path to vol dataset.")
    parser.add_argument("--scanner", default="data_generator/synthetic_dataset/scanner/cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/4DCT_STRETCHED", type=str, help="Path to output.")
    parser.add_argument("--n_train", default=50, type=int, help="Number of projections for training.")
    parser.add_argument("--n_test", default=100, type=int, help="Number of projections for test.")
    parser.add_argument("--device", default=0, type=int, help="GPU device.")
    # fmt: on

    args = parser.parse_args()
    main(args)

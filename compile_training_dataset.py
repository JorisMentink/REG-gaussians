import os
import random
import numpy as np
import sys
import json
import argparse



def generate_dataset(input_folder, output_path, train_size,test_size):

    phasebin_folders = sorted(os.listdir(input_folder))
    phases = range(0, len(phasebin_folders))

    data_output_path = output_path

    train_proj_path = os.path.join(data_output_path, "proj_train") 
    test_proj_path = os.path.join(data_output_path, "proj_test")

    os.makedirs(data_output_path, exist_ok=True)  #Creates output folder if it doesn't exist yet.
    os.makedirs(train_proj_path, exist_ok=True)  #Create subfolder to store train projections
    os.makedirs(test_proj_path, exist_ok=True)  #Create subfolder to store test projections

    #Create subfolder to store ground truth volmumes
    output_subfolder_gt = os.path.join(data_output_path, "GroundTruthVols")
    os.makedirs(output_subfolder_gt, exist_ok=True)

    gt_volumes = []

    #Create organized ground truth volume folder.
    for phasebin in phasebin_folders:
        
        phasebin_path = os.path.join(input_folder, phasebin)
        
        if not os.path.isdir(phasebin_path):
            continue  # Skip non-directory files

        #Load 3D volume
        gt_volume = np.load(os.path.join(phasebin_path, "vol_gt.npy"))
        output_gt_path = os.path.join(output_subfolder_gt, f"{phasebin}_gt.npy")
        np.save(output_gt_path, gt_volume)
        
        print(f"Saved ground truth volume for {phasebin} to {output_gt_path}")
        
        gt_volumes.append(os.path.relpath(output_gt_path,data_output_path))

    #Creating variables to keep track of sampled projections
    sampled_train_projections = 0
    sampled_test_projections = 0

    proj_train_metadata = []
    proj_test_metadata = []

    for phase_num in phases:

        phase_num = phase_num
        phasebin_name = phasebin_folders[phase_num]

        print(f"currently working on phase bin {phasebin_name}")
        
        samplefolder = os.path.join(input_folder, phasebin_name)
            
        #Load metadata from phase bin folder for lookup of angles etc.
        print(f"Loading metadata from {os.path.join(samplefolder, 'meta_data.json')}")
        with open(os.path.join(samplefolder, "meta_data.json"), "r") as f:
            metadata = json.load(f)


        print(f"Sampling {train_size} train and {test_size} test projections from phase bin {phasebin_name}.")
        #Sample projections for every timepoint, both train and test according to specified sizes.
        train_proj_files = [f for f in os.listdir(os.path.join(samplefolder, "proj_train")) if f.endswith('.npy')]
        chosen_train_projs = random.sample(train_proj_files, train_size)
        print(f"Chosen train projections: {chosen_train_projs}")

        for proj in chosen_train_projs:
            arr = np.load(os.path.join(samplefolder, "proj_train", proj))
            print(f"Loaded projection f{proj}")
            output_proj_path = os.path.join(train_proj_path, f"proj_train_{sampled_train_projections:04d}.npy")
            relative_proj_path = os.path.relpath(output_proj_path, data_output_path)
            np.save(output_proj_path, arr)
            sampled_train_projections += 1

            angle = None
            for entry in metadata["proj_train"]:
                if os.path.basename(entry["file_path"]) == proj:
                    angle = entry["angle"]
                    break

            # Store metadata for this projection
            proj_train_metadata.append({
                "phasebin": phasebin_name,
                "original_file_path": proj,
                "file_path": relative_proj_path,
                "angle": angle,
                "timepoint": phase_num,
                "gt_vol": os.path.join("GroundTruthVols", f"{phasebin_name}_gt.npy")
            })


        test_proj_files = [f for f in os.listdir(os.path.join(samplefolder, "proj_test")) if f.endswith('.npy')]
        chosen_test_projs = random.sample(test_proj_files, test_size)
        
        for proj in chosen_test_projs:
            arr = np.load(os.path.join(samplefolder, "proj_test", proj))
            output_proj_path = os.path.join(test_proj_path, f"proj_test_{sampled_test_projections:04d}.npy")
            relative_proj_path = os.path.relpath(output_proj_path, data_output_path)
            np.save(output_proj_path, arr)
            sampled_test_projections += 1

            angle = None
            for entry in metadata["proj_test"]:
                if os.path.basename(entry["file_path"]) == proj:
                    angle = entry["angle"]
                    break

            # Store metadata for this test projection
            proj_test_metadata.append({
                "phasebin": phasebin_name,
                "original_file_path": proj,
                "file_path": relative_proj_path,
                "angle": angle,
                "timepoint": phase_num,
                "gt_vol": os.path.join("GroundTruthVols", f"{phasebin_name}_gt.npy")
            })


    combined_metadata = {
        "scanner": metadata["scanner"],
        "bbox": [[-1, -1, -1], [1, 1, 1]],
        "gt_vols": gt_volumes,
        "proj_train": proj_train_metadata,
        "proj_test": proj_test_metadata,
        "phase_bin": phasebin
    }

    combined_metadata_path = os.path.join(data_output_path, "meta_data.json")
    with open(combined_metadata_path, "w", encoding="utf-8") as f:
        json.dump(combined_metadata, f, indent=4)


#Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trainable dataset from 4DCT data")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input folder containing phase bin data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output folder for combined dataset")
    parser.add_argument("--train_size", type=int, default=30, help="Number of training projections to sample per phase (default: 30)")
    parser.add_argument("--test_size", type=int, default=30, help="Number of test projections to sample per phase (default: 30)")
    
    args = parser.parse_args()
    
    print(f"Input folder: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    
    generate_dataset(
        input_folder=args.input_path,
        output_path=args.output_path,
        train_size=args.train_size,
        test_size=args.test_size
    )
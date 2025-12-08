import os
import random
import numpy as np
import sys
import json


#UTILS:
def find_phase_folder(input_folder, phasebin):
    for folder in os.listdir(input_folder):
        if folder.startswith(str(phasebin)):
            return os.path.join(input_folder, folder)
    return None





def find_closest_phase(timepoint, breathing_cycle_period, time_per_phase):
    """
    Function that determines the closest phase bin for a given timepoint, breathing cycle period and time per phase bin.

    Input parameters:
    timepoint:                  Timepoint in seconds (float or int)
    breathing_cycle_period:     Period of one breathing cycle in seconds (float or int)
    time_per_phase:             Time duration of one phase bin in seconds (float or int)

    Output:
    phase:                      Closest phase bin (int)
    """

    #Handle end of cycle case (TODO: bake this into equation if possible. this sucks lmfao)
    if abs(timepoint % breathing_cycle_period) < 1e-6:
        return (breathing_cycle_period / time_per_phase -1) #Returns last phase bin

    amount_of_cycles = timepoint // breathing_cycle_period #Number of complete breathing cycles that have passed
    phase = round(int((timepoint - (amount_of_cycles * breathing_cycle_period) - 0.001 )/(time_per_phase))) #Find phase in current breathing cycle, use -0/001 to always round half down

    return int(phase)

def sample_points(case_folder,total_time=10,nr_of_timepoints=300,breathing_cycle_period=4,imgshape=(94,256,256),dtype=np.uint16,save_as_file=True,verbose=True):
    """
    Function that samples time-based 4DCT from multiple phase binned 3DCT volumes.

    Input parameters:
    case_folder:                Path to case folder (string)
    total_time:                 Total time duration to sample in seconds (int or float, default=60)
    nr_of_timepoints:           Number of timepoints to sample (int, default=300)
    breathing_cycle_period:     Period of one breathing cycle in seconds (int or float, default=4)
    imgshape:                   Shape of one 3D volume (tuple, default=(94,256,256))
    dtype:                      Data type of the image (numpy dtype, default=np.uint16)
    save_as_file:               Whether to save the sampled phases as a .npy file (boolean, default=True)

    Output:
    sampled_phases:             Numpy array of shape (nr_of_timepoints, 2) with columns timepoint and phase bin.
    """
    
    casename = os.path.basename(case_folder)

    #Opening folder containing 3D volumes
    images_folder = case_folder + "/Images"
    img_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".img")])

    amount_of_phases = len(img_files) #Reads amount of phase bins present in folder

    if verbose:
        print(f"Found {amount_of_phases} 3D volumes:")
        for f in img_files:
            print(f)

    starting_point = total_time / nr_of_timepoints #Use starting point >0 to avoid finding phase 0
    timepoints = np.linspace(starting_point, total_time, nr_of_timepoints) #Timepoints in seconds
    time_per_phase = breathing_cycle_period / amount_of_phases #Time per phase in seconds

    sampled_phases = []

    for t in timepoints: #iterate find_closest_phase for all timepoints
        phase_num = int(find_closest_phase(t, breathing_cycle_period, time_per_phase))
        phasebin_name = os.path.splitext(img_files[phase_num])[0]
        sampled_phases.append((np.round(t,2),phase_num, phasebin_name))

    if save_as_file:
        filename=os.path.join(case_folder, f"sampled_phases{casename}.npy")
        np.save(filename,sampled_phases)
        
        if verbose:
            print(f"Saved sampled phases to {filename}")
            print(sampled_phases)
    
    return sampled_phases

def generate_dataset(input_folder, output_path, sampled_phases, train_size,test_size):

    base_data_folder = os.path.dirname(input_folder)
    phasebin_folders = sorted(os.listdir(input_folder))

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

    for datapoint in sampled_phases:

        timepoint, phase_num, phasebin_name = datapoint

        print(f"currently working on timepoint {timepoint} seconds, phase bin {phasebin_name}")


        for folder in os.listdir(input_folder):
            print(f"Checking folder: {folder}, looking for {phasebin_name}")
            if folder.lower().startswith(str(phasebin_name.lower())):
                samplefolder = os.path.join(input_folder, folder) #TODO: Add error handling if folder not found
                print(f"Found folder: {samplefolder}!")
                break

            
        #Load metadata from phase bin folder for lookup of angles etc.
        print(f"Loading metadata from {os.path.join(samplefolder, 'meta_data.json')}")
        with open(os.path.join(samplefolder, "meta_data.json"), "r") as f:
            metadata = json.load(f)


        print(f"Sampling {train_size} train and {test_size} test projections from phase bin {phasebin_name} for timepoint {timepoint} seconds.")
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
    if len(sys.argv) < 2:
        print("Usage: python sample_4DCT.py <path>")
        exit(1)
    
    #Change default parameters if needed
    total_time = 4  # Total time duration to sample in seconds
    nr_of_timepoints = 10  # Number of timepoints to sample
    breathing_cycle_period = 4  # Period of one breathing cycle in seconds
    imgshape = (94, 256, 256)  # Shape of one 3D volume
    
    selected_folder = sys.argv[1]
    print(f"Using folder: {selected_folder}")
    sampled_phases = sample_points(selected_folder,total_time=total_time,nr_of_timepoints=nr_of_timepoints,breathing_cycle_period=breathing_cycle_period,imgshape=imgshape) #TODO: Implement argument parsing for other parameters
    generate_dataset(input_folder="data/synthetic_dataset/Downsampled_2x_4DCT", output_path="data/combined_data/Downsampled_2x_4DCT_SMALL", sampled_phases=sampled_phases, train_size=30, test_size=30)
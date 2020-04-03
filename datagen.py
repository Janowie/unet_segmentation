# class SequenceDataGenerator(keras.utils.Sequence):
import os
import numpy as np
import SimpleITK as sitka
import random


def test():
    print("Import succes")


def get_image_array(file_path):
    return sitka.GetArrayFromImage(sitka.ReadImage(file_path, sitka.sitkFloat32))


def normalise(tensor, image_size):

    for idx_channel, channel in enumerate(tensor):

        b_rows, b_cols = np.nonzero(channel.astype(bool))
        brain_values = []
        for coords in zip(b_rows, b_cols):
            y, x = coords
            brain_values.append(channel[y][x])
        brain = np.array(brain_values)

        if np.count_nonzero(brain): # only normalise slices with a brain region
            b_mean = brain.mean()
            b_std = brain.std()
            if b_mean and b_std:
                brain = (brain - b_mean) / b_std

        channel = np.zeros((image_size, image_size)) # Reset background
        for idx,coords in enumerate(zip(b_rows, b_cols)):
            y, x = coords
            channel[y, x] = brain[idx]
        
        tensor[idx_channel] = channel
        
    return tensor


def select_tumor_region(mask, tumor_region):
        if tumor_region != -1: # only apply if specific labels are requested
            for i, channel in enumerate(mask):
                mask[i][mask[i] >= tumor_region] = tumor_region
                mask[i][mask[i] < tumor_region] = 0
        return mask


def get_slice_idxs_w_tumor(mask, num, slices_w_tumor_only):     
    mask_idxs = []
    slice_offset = 20
    for i, s in enumerate(mask[slice_offset:]):
        if np.count_nonzero(s > 0) > slices_w_tumor_only:
            mask_idxs.append(i + slice_offset)
        if len(mask_idxs) == num:
            return mask_idxs
    return mask_idxs


def get_images_masks(patient_dir, slices_from_patient, image_size, slices_w_tumor_only, slices_w_less_brain, mode):
        
        imaging_sequence_sufixes = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]
        mask_sufix = "_seg.nii"
        num_slices_per_image = 159
        # Slices with expected high ratio of brain campared with its background
        slices_from = 60
        slices_to = 100
        
        # Declare and initialize numpy arrays to return
        tensor = np.zeros((slices_from_patient, len(imaging_sequence_sufixes), 240, 240))
        mask = np.zeros((slices_from_patient, 1, 240, 240))
        
        # Open patients images
        mask_img = get_image_array(patient_dir + "/" + patient_dir[patient_dir.rfind('\\') + 1:] + mask_sufix)
        tensors = []
        for sufix in imaging_sequence_sufixes:
            tensors.append(get_image_array(patient_dir + "/" + patient_dir[patient_dir.rfind('\\') + 1:] + sufix))
        
        # Create field to iterate through
        slice_indexes = np.random.choice(range(slices_from, slices_to), slices_from_patient, replace=False)
        
        # include slices with less brain - picks 2 or given number of slices from range 30-60 and (or) 100-120
        if slices_w_less_brain:
            if type(slices_w_less_brain) == int and slices_w_less_brain > 0:
                for i in range(slices_w_less_brain):
                    slice_indexes[i] = random.randint(*random.choice([(slices_from-20, slices_from), (slices_to, slices_to+20)]))
        
        # if selected, replace all slices without tumor with slices that contain tumor
        if slices_w_tumor_only and mode == "training":
            indexes_w_tumor = get_slice_idxs_w_tumor(mask_img, len(slice_indexes), slices_w_tumor_only)
            for i, slice_i in enumerate(slice_indexes):
                if np.count_nonzero(mask_img[slice_i] > 0) < slices_w_tumor_only:
                    if i < len(indexes_w_tumor): 
                        slice_indexes[i] = indexes_w_tumor[i]
                    else:
                        break
#             slice_indexes = indexes_w_tumor
        
        # Loop through slice indexes and fill values in to the "to be returned" tensor
        for final_tensor_i, slice_i in enumerate(slice_indexes):
            for tensors_i in range(len(imaging_sequence_sufixes)):
                tensor[final_tensor_i][tensors_i] = tensors[tensors_i][slice_i]
        
            mask[final_tensor_i] = mask_img[slice_i]
            
            # Normalise tensor
            tensor[final_tensor_i] = normalise(tensor[final_tensor_i], image_size)
            
        return tensor, mask


# Main function
def get_dataset(slices_from_patient, 
                file_path='../dataset/', 
                mode='training', 
                glioma_type=['HGG'], 
                tumor_region=-1, 
                slices_w_tumor_only=False,
                slices_w_less_brain=None,
                image_size=240,
                train_HGG_patients=239):
    
    print("Getting dataset")

    dir_paths = []
    for glioma in glioma_type:
        for directory in os.listdir(file_path + mode + "/" + glioma):
            dir_paths.append(os.path.join(file_path, mode, glioma, directory))  

    train_data_x = []
    train_data_y = []

    for patient_dir in dir_paths:
        tensors, masks = get_images_masks(patient_dir, slices_from_patient, image_size, slices_w_tumor_only, slices_w_less_brain, mode)
        for j in range(len(tensors)):
            train_data_x.append(tensors[j])
            train_data_y.append(masks[j])

    train_data_x = np.array(train_data_x)
    train_data_y = np.array(train_data_y)

    randomize = np.arange(len(train_data_x))
    np.random.shuffle(randomize)
    train_data_x = train_data_x[randomize]
    train_data_y = train_data_y[randomize]

    print("Shape of x_train, y_train", train_data_x.shape, ",", train_data_y.shape)
    return train_data_x, train_data_y

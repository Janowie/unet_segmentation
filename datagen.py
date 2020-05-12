# class SequenceDataGenerator(keras.utils.Sequence):
import os
import numpy as np
import SimpleITK as sitka
import random
import keras
import random
import time


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


def normalize_tensor(tensor):
    if not np.count_nonzero(tensor): 
        return tensor
    brain_mean = tensor[tensor > 0].mean()
    brain_std = tensor[tensor > 0].std()
    tensor[tensor > 0] -= brain_mean
    tensor[tensor > 0] /= brain_std
    return tensor


def get_patients_tensors_and_mask(patient_dir, mask_sufix, imaging_sequence_sufixes):
    mask_img = get_image_array(patient_dir + "/" + patient_dir[patient_dir.rfind('\\') + 1:] + mask_sufix)
    tensors = []
    for sufix in imaging_sequence_sufixes:
        tensors.append(get_image_array(patient_dir + "/" + patient_dir[patient_dir.rfind('\\') + 1:] + sufix))
    
    return np.swapaxes(np.array(tensors), 0, 1), np.swapaxes(np.array([mask_img]), 0, 1)


def get_images_masks(patient_dir, slices_from, slices_to, slices_from_patient, image_size, slices_w_tumor_only, mode, tumor_region):
    
    imaging_sequence_sufixes = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]
    mask_sufix = "_seg.nii"
    num_slices_per_image = 159

    # Open patients images
    tensors, mask_img = get_patients_tensors_and_mask(patient_dir, mask_sufix, imaging_sequence_sufixes)

    # Create field to iterate through
    if mode == "training" and tumor_region:
        slice_indexes = np.random.choice(np.unique(np.nonzero(mask_img == tumor_region)[0]) or range(slices_from, slices_to), slices_from_patient)
    else:
        slice_indexes = np.random.choice(range(slices_from, slices_to), slices_from_patient, replace=False)
    
    tensor = tensors[slice_indexes]
    mask = mask_img[slice_indexes]
    
    for i, t in enumerate(tensor):
        tensor[i] = normalise(t, image_size)

    return tensor, mask


def load_patient(patient_dir, image_size):
    imaging_sequence_sufixes = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]
    mask_sufix = "_seg.nii"
    
    # Open patients images
    tensors, mask = get_patients_tensors_and_mask(patient_dir, mask_sufix, imaging_sequence_sufixes)
    
    # Normalize
    for i in range(len(tensors)):
        tensors[i] = normalise(tensors[i], image_size)

    return np.array(tensors), np.array([mask])
    

def convert_labels(masks, tumor_region, image_size):
    # Change all labels to tumor region specified
    if tumor_region == 0:
        raise ValueError('Invalid tumor_region value')
    binary_masks = []
    tumor_val = 1
    for i, mask_slice in enumerate(masks):
        binary_mask = np.zeros((1, image_size, image_size))
        binary_mask[mask_slice >= tumor_region] = tumor_val
        binary_masks.append(binary_mask)
    return np.array(binary_masks)


def get_dir_paths(file_path, mode, glioma_type):
    dir_paths = []
    for glioma in glioma_type:
        for directory in os.listdir(file_path + mode + "/" + glioma):
            dir_paths.append(os.path.join(file_path, mode, glioma, directory))
    return dir_paths


# Main function
def get_dataset(slices_from_patient, 
                slices_from, slices_to,
                file_path='../dataset/', 
                mode='training', 
                glioma_type=['HGG'],
                randomize_slices=False,
                slices_w_tumor_only=False,
                slices_w_less_brain=None,
                tumor_region=None,
                image_size=240,
                train_HGG_patients=239):

    x = []
    y = []
    dir_paths = get_dir_paths(file_path, mode, glioma_type)
    
    # Get tensors and masks
    for i, patient_dir in enumerate(dir_paths):
        tensors, masks = get_images_masks(patient_dir, slices_from, slices_to, slices_from_patient, image_size, slices_w_tumor_only, mode, tumor_region)
        for j in range(len(tensors)):
            x.append(tensors[j])
            y.append(masks[j])
        if i == train_HGG_patients:
            break
    
    # Randomize slice order
    x = np.array(x)
    y = np.array(y)
    if randomize_slices:
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]

    print(mode, "data with the shape:", x.shape, y.shape)

    return np.array(x), np.array(y)


def get_whole_patient(number=1, 
                      file_path='../dataset/', 
                      mode='training', 
                      glioma_type=['HGG'], 
                      image_size=240                     
                     ):
    x = []
    y = []
    dir_paths = get_dir_paths(file_path, mode, glioma_type)
    random.shuffle(dir_paths)
    for i, patient_dir in enumerate(dir_paths):
        if i == number: break
        
        tensors, masks = load_patient(patient_dir, image_size)
        x.append(tensors)
        y.append(masks)

    return np.array(x), np.array(y)


class AugmentationDatagen(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, augmentations):
        self.x, self.y = x_set, y_set
        self.x_aug, self.y_aug = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        
        batch_x = np.array(self.x_aug[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_y = np.array(self.y_aug[idx * self.batch_size:(idx + 1) * self.batch_size])            
        return batch_x, batch_y
    
    def on_epoch_end(self):
        data_x = []
        data_y = []
        
        if self.augment:
            for img, mask in zip(self.x, self.y):
                
                if np.count_nonzero(mask): # Final control if mask contains tumor labels
                    
                    try:
                        augmented = self.augment(image=img, mask=mask)
                    except Exception as e:
                        f, axarr = plt.subplots(1,2)
                        axarr[0].imshow(img[0]) #, cmap='gray', vmin=0
                        axarr[1].imshow(mask[0], alpha=1)
                    data_x.append(augmented["image"])
                    data_y.append(augmented["mask"])
                else:
                    data_x.append(img)
                    data_y.append(mask)
        else:
            data_x, data_y = selg.x_aug, selg.y_aug
            
        self.x_aug = np.array(data_x)
        self.y_aug = np.array(data_y)

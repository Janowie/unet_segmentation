# class SequenceDataGenerator(keras.utils.Sequence):
import os
import numpy as np
import SimpleITK as sitka
import random
import keras


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
                m_slice = mask_img[slice_i]
                if np.count_nonzero(m_slice >= 0) < slices_w_tumor_only:
                    if i < len(indexes_w_tumor):
                        slice_indexes[i] = indexes_w_tumor[i]
                    else:
                        break        
        
        # Loop through slice indexes and fill values in to the "to be returned" tensor
        for final_tensor_i, slice_i in enumerate(slice_indexes):
            for tensors_i in range(len(imaging_sequence_sufixes)):
                tensor[final_tensor_i][tensors_i] = tensors[tensors_i][slice_i]
        
            mask[final_tensor_i] = np.nan_to_num(mask_img[slice_i]) # replace all nan values with 0
            
            # Normalise tensor
            tensor[final_tensor_i] = normalise(tensor[final_tensor_i], image_size)
                
        return tensor, mask
    

def convert_labels(masks, tumor_region):
    # Change all labels to tumor region specified
    if tumor_region == 0:
        raise ValueError('Invalid tumor_region value')
    
    background_val = 0
    tumor_val = 1
    for i, mask_slice in enumerate(masks):
        masks[i][mask_slice < tumor_region] = background_val
        masks[i][mask_slice >= tumor_region] = tumor_val
    return masks    


# Main function
def get_dataset(slices_from_patient, 
                file_path='../dataset/', 
                mode='training', 
                glioma_type=['HGG'],
                randomize_slices=False,
                slices_w_tumor_only=False,
                slices_w_less_brain=None,
                image_size=240,
                train_HGG_patients=239):

    dir_paths = []
    for glioma in glioma_type:
        for directory in os.listdir(file_path + mode + "/" + glioma):
            dir_paths.append(os.path.join(file_path, mode, glioma, directory))  

    train_data_x = []
    train_data_y = []

    for i, patient_dir in enumerate(dir_paths):
        tensors, masks = get_images_masks(patient_dir, slices_from_patient, image_size, slices_w_tumor_only, slices_w_less_brain, mode)
        for j in range(len(tensors)):
            train_data_x.append(tensors[j])
            train_data_y.append(masks[j])
        if i == train_HGG_patients:
            break

    train_data_x = np.array(train_data_x)
    train_data_y = np.array(train_data_y)
    if randomize_slices:
        randomize = np.arange(len(train_data_x))
        np.random.shuffle(randomize)
        train_data_x = train_data_x[randomize]
        train_data_y = train_data_y[randomize]

    print(mode, "data with shape:", train_data_x.shape, train_data_y.shape)

    return train_data_x, train_data_y


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
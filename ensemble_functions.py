import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import cv2
import datagen
import model as unet_model_script
import random
import matplotlib.pyplot as plt
import importlib
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve
import copy


def get_models(path, name, a, b, verbose=0):
    """
    ### get_models

    Loads and returns models based on path and name.

    :param path: path used to load models  
    :param name: model name appended to path    
    :param a: tumor region label from  
    :param b: tumor region label to  
    :param verbose: prints informative texts

    :return: models
    """
    models = {}
    for tumor_region in range(a, b+1):
        if verbose==1: print("Loading model", path + name + "_" + "{}.h5".format(tumor_region))
        models[tumor_region] = keras.models.load_model(path + name + "_" + "{}.h5".format(tumor_region))
    return models


def get_predictions(models, x, a, b, verbose=0):
    """
    ### get_predictions

    Loads models and returns predictions based on x.
    
    :param models: dict with loaded models
    :param x: data to use for predictions  
    :param a: tumor region label from  
    :param b: tumor region label to  
    :param verbose: prints informative texts

    :return: predictions
    """
    predictions = {}
    for tumor_region in range(a, b+1):
        if verbose==1: print("Model {} predictions".format(tumor_region))
        predictions[tumor_region] = models[tumor_region].predict(x, verbose=verbose)
    return predictions


def create_mask(predictions, num_slices, image_size, p=0.5):
    """
    ### create_mask

    :param predictions: dict of prediction lists for each tumor class (1..4), where each list contains all slices segmented  
    :param num_slices: integer, describes the length of each list in predictions  
    :param image_size: integer, describes image size (x, y)  
    :param p: float, sets the threshold value of which pixels to use in the final mask  

    :return: final mask as numpy array with dimensions (num_slices, image_size, image_size)
    """
    final_mask = np.zeros((num_slices, 1, image_size, image_size))
    b_d_thresholds = [0.5695, 0.4209, 0.6089, 0.5738]
    
    for k in predictions.keys():
        if p == None:
            final_mask[predictions[k] >= b_d_thresholds[k-1]] = k
        else:
            final_mask[predictions[k] >= p] = k

    return np.nan_to_num(final_mask) # replace all nan values with 0


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2 * |X & Y|)/ (|X| + |Y|)
         =  2  *sum(|A * B|)/(sum(A ^ 2)+sum(B ^ 2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    source: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
    """
    # Change dtype
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    axis=None
    intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),axis) + K.sum(K.square(y_pred),axis) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def evaluate_tumor_region(y, f, image_size, tumor_region):
    return dice_coef(datagen.convert_labels_eval(y, tumor_region, image_size), datagen.convert_labels_eval(f, tumor_region, image_size)).numpy()


def evaluate_ensemble(x_val, y_val, path, model_name, num_patients, image_size, final_mask_threshold, verbose):
    # Evaluation pipeline:
    
    tumor_regions = ["enhancing", "whole", "core"]

    evaluations = {
        "enhancing": [],
        "whole": [],
        "core": []
    }

    models = get_models(path, model_name, 1, 4, verbose=verbose)
    if verbose:
        print("Models loaded")

    for i in range(num_patients):
        if verbose:
            print("Evaluating patient {}".format(i))
        predictions = get_predictions(models, x_val[i], 1, 4, verbose=verbose)
        final_mask = create_mask(predictions, 155, image_size, p=final_mask_threshold)
        
        for tumor_region in tumor_regions:
            evaluations[tumor_region].append(evaluate_tumor_region(y_val[i][0], final_mask, image_size, tumor_region))

    results = {}
    for tumor_region in tumor_regions:
        results[tumor_region] = {
            "mean": round(np.array(evaluations[tumor_region]).mean() * 100, 2),
            "stdDev": round(np.std(np.array(evaluations[tumor_region])) * 100, 2),
            "median": round(np.median(np.array(evaluations[tumor_region])) * 100, 2)
        }

    # Cleanup
    del models
    del predictions
    
    return results


def cross_val_avg(tumor_regions, results):
    stats_dict = {
            "mean": 0,
            "stdDev": 0,
            "median": 0
        }
    results_avg = {}
    for t in tumor_regions:
        results_avg[t] = copy.deepcopy(stats_dict)
        
    # Add numbers
    for r in results:
        for tumor_part in r.keys():
            for sm in r[tumor_part].keys():
                results_avg[tumor_part][sm] += r[tumor_part][sm]
    # Calc average:
    for r in results_avg.keys():
        for sm in results_avg[r].keys():
            results_avg[r][sm] = round(results_avg[r][sm]/3, 2)
    
    return results_avg   


def evaluate_cv_ensemble(path, model_name, num_patients, image_size, final_mask_threshold, verbose):
    tumor_regions=["enhancing", "whole", "core"]
    results=[]
    
    x_val, y_val = datagen.get_whole_patient(number=num_patients, 
                    file_path='../dataset/', 
                    mode='test', 
                    glioma_type=['HGG'], 
                    image_size=240                     
                    )
    if verbose:
        print("Data loaded")
        
    for cv in range(1, 4):
        results.append(evaluate_ensemble(x_val, y_val, path, "{}_{}".format(cv, model_name), num_patients, image_size, final_mask_threshold, verbose))
    return cross_val_avg(tumor_regions, results)

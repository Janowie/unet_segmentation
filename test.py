def per_class_accuracy(y_preds, y_true, class_labels):
    return_arr = []
    for class_label in class_labels:
        if y_true[pred_idx] == int(class_label):
            mean_val = []
            for pred_idx, y_pred in enumerate(y_preds):
                mean_val.append(y_true[pred_idx] == np.round(y_pred))

        return_arr.append(np.mean([mean_val]))

    return return_arr
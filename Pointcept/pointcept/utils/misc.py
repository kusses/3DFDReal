"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import warnings
from collections import abc
import numpy as np
import torch
from importlib import import_module
from sklearn.metrics import precision_recall_curve, auc, average_precision_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    #output = output.reshape(output.size).copy()
    output = output.view(-1).cpu().numpy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    #print(f"Debug: Output shape: {output.shape}, Target shape: {target.shape}")
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

#Wonjoo
def calculate_precision_recall(output, target, K, iou_threshold, ignore_index=-1):
    """
    Calculate precision and recall at a specific IoU threshold across all classes.
    """
    area_intersection, area_union, area_target = intersection_and_union(output, target, K, ignore_index)

    precision_list = []
    recall_list = []

    for class_id in range(K):
        if area_union[class_id] == 0:
            # No prediction or target for this class, precision and recall are not available, so pass.
            #precision_list.append(np.nan)
            #recall_list.append(np.nan)
            precision_list.append(0)
            recall_list.append(0)

        else:

            tp = area_intersection[class_id] >= iou_threshold * area_union[class_id]
            fp = (area_union[class_id] > 0) & ~tp
            fn = (area_target[class_id] > 0) & ~tp

            precision = np.sum(tp) / (np.sum(tp) + np.sum(fp) + 1e-10)
            recall = np.sum(tp) / (np.sum(tp) + np.sum(fn) + 1e-10)

            precision_list.append(precision)
            recall_list.append(recall)

    return precision_list, recall_list

def calculate_ap_per_class(output, target, K, iou_threshold, ignore_index=-1):
    """
    Calculate Average Precision (AP) for each class.
    """
    # set the length = K for ap_per_class calculation
    ap_per_class = np.full(K, np.nan)  # Set the initial ap value
    precision_list, recall_list = calculate_precision_recall(output, target, K, iou_threshold, ignore_index)
    # ap_per_class = precision_list
    # In this context, AP is the same as precision for a single IoU threshold
    # Apply actual precision value for each class
    for class_id in range(len(precision_list)):
        if not np.isnan(precision_list[class_id]):
            ap_per_class[class_id] = precision_list[class_id]

    return ap_per_class

def calculate_true_ap_per_class_with_threshold(probabilities, targets, num_classes, ignore_index=-1, threshold=0.5):
    """
    Calculate per-class AP with confidence thresholding.
    Only predictions with confidence >= threshold are considered.
    """
    ap_per_class = np.full(num_classes, np.nan)
    max_probs = np.max(probabilities, axis=1)
    pred_classes = np.argmax(probabilities, axis=1)

    for class_id in range(num_classes):
        valid_mask = (targets != ignore_index) & (max_probs >= threshold)
        class_probs = probabilities[valid_mask, class_id]
        class_targets = (targets[valid_mask] == class_id).astype(int)

        if np.sum(class_targets) == 0:
            continue

        from sklearn.metrics import precision_recall_curve, auc
        precision, recall, _ = precision_recall_curve(class_targets, class_probs)
        ap = auc(recall, precision)
        ap_per_class[class_id] = ap
    return ap_per_class

def calculate_soft_ap_per_class(probabilities, targets, num_classes, ignore_index=-1):
    """
    Calculates per-class average precision using sklearn.metrics.average_precision_score.
    This is a proper soft-mAP implementation.
    """
    ap_per_class = np.full(num_classes, np.nan)
    valid_mask = targets != ignore_index
    targets = targets[valid_mask]
    probs = probabilities[valid_mask]

    for class_id in range(num_classes):
        # Binary targets for current class
        binary_gt = (targets == class_id).astype(int)
        if np.sum(binary_gt) == 0:
            continue  # No positive sample for this class

        scores = probs[:, class_id]
        try:
            ap = average_precision_score(binary_gt, scores)
        except ValueError:
            ap = np.nan  # Fail-safe
        ap_per_class[class_id] = ap

    return ap_per_class



def calculate_true_ap_per_class(probabilities, targets, num_classes, ignore_index=-1):
    ap_per_class = np.full(num_classes, np.nan)
    for class_id in range(num_classes):
        valid_mask = targets != ignore_index
        class_probs = probabilities[valid_mask, class_id]
        class_targets = (targets[valid_mask] == class_id).astype(int)
        if np.sum(class_targets) == 0:
            continue
        precision, recall, _ = precision_recall_curve(class_targets, class_probs)
        ap = auc(recall, precision)
        ap_per_class[class_id] = ap
    return ap_per_class



def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class DummyClass:
    def __init__(self):
        pass

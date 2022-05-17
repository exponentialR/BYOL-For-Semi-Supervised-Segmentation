import torch
import numpy as np
import torch.nn.functional as F
import default_param

def get_confusion_matrix(output: torch.Tensor, target: torch.Tensor)-> np.ndarray:
    """

    :param output: output logits
    :param target: targets
    :return: returns the confusion matric of shape CxC betweeb prediction and tagrets
    """
    n_classes = output.shape[1]
    softmax_out = F.softmax(output, dim=1)
    pred = torch.argmax(softmax_out, dim=1).squeeze(1).cpu().numpy()
    target = target.cpu().numpy()
    hist = fast_hist(pred.flatten(), target.flatten(), n_classes)
    return hist

def mIOU(c_matrix: np.ndarray) -> float:
    """
    calculates the mIOU for a given confusion matrix
    :param c_matrix: Squared matrix
    :return: effection mIOU
    """
    if type(c_matrix) != np.ndarray:
        return 0
    class_iu = per_class_iu(c_matrix)
    m_iou = np.nanmean(class_iu)
    return m_iou

def fast_hist(pred: torch.Tensor, label:torch.Tensor, n:int) -> np.ndarray:
    """

    :param pred: softmax-applied prediction of shape[N, H, W]
    :param label: Labels of shape [N, H, W]
    :param n: number of classes
    :return: a  square matrix where row i represents the count of each class in prediction when the actual class of label is i

    """
    k = (label >=0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength = n**2).reshape(n,n)

def per_class_iu(hist:np.ndarray) -> np.array:
    """

    :param hist: bin count matrix of size CxC where C is the no of output classes
    :return: list of IOU per each class
    """
    return np.diag(hist)/(hist.sum(1) + hist.sum(0) - np.diag(hist))

def pixel_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """

    :param output:
    :param target:
    :return:
    """
    softmax_out = F.softmax(output, dim=1)
    pred= torch.argmax(softmax_out, dim =1).squeeze(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target!=default_param.CITYSCAPES_IGNORED_INDEX]
    correct = correct.view(-1)
    score = correct.float().sum(0) /correct.size(0)
    return score.item()


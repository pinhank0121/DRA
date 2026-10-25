import torch

def accuracy(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    # print(f"DEBUG: preds.shape={preds.shape}, true.shape={true.shape}")
    try:
        accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float()/float(true.size(0))
    except Exception as e:
        print(f"DEBUG: preds.shape={preds.shape}, true.shape={true.shape}")
        raise e
    return accuracy.item()

import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    """
    Calculates the VQA criterion for a batch of predictions and answers.

    Args:
        batch_pred (torch.Tensor): Tensor containing the predicted answers for each sample in the batch.
        batch_answers (torch.Tensor): Tensor containing the ground truth answers for each sample in the batch.

    Returns:
        float: The VQA criterion value for the batch.

    """
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)
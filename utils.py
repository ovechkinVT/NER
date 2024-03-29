# python 3
import torch


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    """Фукнция преобразует набор токенов в тензор из их id."""
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)



def log_sum_exp(vec):
    """Compute log sum exp in a numerically stable way for the forward algorithm. """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

import torch

def total_elem(net,per_layer = False):
    total = 0
    for name,p in net.named_parameters():
        if per_layer:
            print(name,p.numel())
        total += p.numel()
    print(total)
    return total



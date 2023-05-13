import torch.nn.utils.prune as prune
import torch.nn as nn
import torch

def Lottery_Ticket_OneShot(net,amount):
    '''
    Prunes the supplied network by an amount where amound is between 0 and 1 (percentage)
    One shot so we do not retrain the network but keep training?
    '''
    for n,c in net.named_children():
        for na,module in c.named_modules():
            if hasattr(module,'weight') and isinstance(module,nn.Conv2d) and (na == 'conv2') and amount[0] is not None:
                print(na)
                prune.ln_structured(module, name="weight", amount=amount[0], n=1, dim=0)
            elif hasattr(module,'weight') and isinstance(module,nn.Conv2d) and (na == 'conv3') and amount[1] is not None:
                print(na)
                prune.l1_unstructured(module, name="weight", amount=amount[1])



def Lottery_Ticket_Iter_Prune(i,net,p:float,n:float):
    '''
    iteratively Prunes the supplied network by a total amount, p, which is between 0 and 1 (percentage)
    Each round i.e. call of this function will prune the network by p^(1/n)

    '''
     
    if i >= n: # Dont Prune if i >= n
        return None
    # Convert to percentage first
    prune_pct = ((p*100)**(1/n))**(i+1)/100
    for child_name,child in net.named_children():
        for na,module in child.named_modules():
            if hasattr(module,'weight') and isinstance(module,nn.Conv2d) and na == 'conv2':
                prune.ln_structured(module, name="weight", amount= prune_pct, n=1, dim=0)
    
    # Reset the weights to original. For the final iteration, dont reset the weights
    if i < n-1:
        for n,c in net.named_children():
            for na,module in c.named_modules():
                if hasattr(module,'weight') and isinstance(module,nn.Conv2d) and na == 'conv2':
                    module.weight = module.weight_orig*module.weight_mask
import logging
import torch


def load_my_state_dict(model: torch.nn.Module, state_dict, lambda_own=lambda x: x):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        own_name = lambda_own(name)
        # own_name = '.'.join(name.split('.')[1:])
        if own_name not in own_state:
            logging.warn('Not found in checkpoint %s %s' % (name, own_name))
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if param.size() != own_state[own_name].size():
            logging.warn('size not match %s %s %s' % (
                name, str(param.size()), str(own_state[own_name].size())))
            continue
        own_state[own_name].copy_(param)



def to_cuda(data, device='cuda'):
    new_data = {}
    for key in data:
        if hasattr(data[key], 'cuda'):
            new_data[key] = data[key].to(device)
        else:
            new_data[key] = data[key]
    return new_data

def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
import logging
import torch


def load_my_state_dict(model: torch.nn.Module, state_dict, lambda_own=lambda x: x):
    own_state = model.state_dict()
    record = {}
    missing_keys, unexpected_keys, mismatch_keys = [], [], []
    for name, param in state_dict.items():
        own_name = lambda_own(name)
        record[own_name] = 0
        # own_name = '.'.join(name.split('.')[1:])
        if own_name not in own_state:
            unexpected_keys.append(f'{name}->{own_name}')
            logging.warn('Unexpected key from checkpoint %s %s' % (name, own_name))
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if param.size() != own_state[own_name].size():
            logging.warn('size not match %s %s %s' % (
                name, str(param.size()), str(own_state[own_name].size())))
            mismatch_keys.append(own_name)
            continue
        own_state[own_name].copy_(param)

    for n in own_state:
        if n not in record:
            missing_keys.append(n)
    
    logging.warn('Unexpected keys' + str(unexpected_keys))
    logging.warn('Missing keys' + str(missing_keys))
    logging.warn('Size mismatched keys' + str(mismatch_keys))
    return missing_keys, unexpected_keys, mismatch_keys
    

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
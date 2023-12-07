import importlib
from omegaconf import OmegaConf
import logging
import torch


def get_obj_from_str(string: str, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)




def load_from_checkpoint(ckpt, cfg_file=None, legacy=False, cfg=None):
    if cfg_file is None:
        cfg_file = ckpt.split('checkpoints')[0] + '/config.yaml'
    print('use cfg file', cfg_file)
    if cfg is None:
        cfg = OmegaConf.load(cfg_file)
    print(cfg)
    if 'resume_ckpt' in cfg.model:
        cfg.model.resume_ckpt = None  # save time to load base model :p
    # legacy issue
    if legacy:
        model = instantiate_from_config(cfg.model, cfg=cfg)

    else:
        module = cfg.model.module
        model_name = cfg.model.model
        module = importlib.import_module(module)
        model_cls = getattr(module, model_name)
        model = model_cls(cfg, )
        model.init_model()

    print('loading from checkpoint', ckpt)    
    # import pdb; pdb.set_trace()
    weights = torch.load(ckpt)['state_dict']
    load_my_state_dict(model, weights)
    return model


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
    
    if unexpected_keys: logging.warn('Unexpected keys' + str(unexpected_keys))
    if missing_keys: logging.warn('Missing keys' + str(missing_keys))
    if mismatch_keys: logging.warn('Size mismatched keys' + str(mismatch_keys))
    return missing_keys, unexpected_keys, mismatch_keys
    
def deep_to(data, device='cuda'):
    if hasattr(data, 'to'):
        return data.to(device)
    if isinstance(data, list):
        for i, d in enumerate(data):
            data[i] = deep_to(d, device)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = deep_to(v, device)
    return data


def to_cuda(data, device='cuda'):
    if hasattr(data, 'to'):
        return data.to(device)
    if isinstance(data, list):
        for i, d in enumerate(data):
            data[i] = to_cuda(d, device)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = to_cuda(v, device)
    elif isinstance(data, tuple):
        data = tuple(to_cuda(d, device) for d in data)
    return data



def zero_grad(params, set_to_none=False):
    for p in params:
        if p.grad is not None:
            if set_to_none:
                p.grad = None
            else:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()
                

def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True
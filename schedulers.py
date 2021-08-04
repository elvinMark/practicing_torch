import torch

def create_lr_scheduler(optim,args):
    if args.sched == "step":
        return torch.optim.lr_scheduler.StepLR(optim,args.step_size,gamma=args.gamma)
    elif args.sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim,args.T_max,eta_min=args.eta_min)
    else:
        return None

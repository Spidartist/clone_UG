import math

def get_warmup_cosine_lr(base_lr, max_lr, cur, total_steps, steps_per_epoch, warmup_epochs=2):
    """ warmup in first 2 epochs, then lr is calculated using cosine function
    """

    if cur <= warmup_epochs * steps_per_epoch:
        lr = base_lr + cur * (max_lr - base_lr)/(warmup_epochs*steps_per_epoch)
    else:
        step = cur - warmup_epochs * steps_per_epoch
        decayed_steps = total_steps - warmup_epochs * steps_per_epoch
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decayed_steps))
        lr = max_lr * cosine_decay
    
    return max(0., lr)
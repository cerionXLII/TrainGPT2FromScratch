import math

class LRScheduler:
    def __init__(self, max_lr=3e-4, min_lr=3e-5, warmup_steps=10, max_steps=50):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
    
    def get_lr(self, step):

        # Linear warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        
        # If we are past the cosine decay we return the minumum lr
        if step >= self.max_steps:
            return self.min_lr

        # Cosine decay
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)

        coeff = 0.5 * (1 + math.cos(math.pi * progress)) #Coeff starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
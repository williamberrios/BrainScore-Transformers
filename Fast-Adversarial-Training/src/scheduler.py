# +
class Linear_LR:
    def __init__(self,lr_init,lr_final,nsteps):
        self.lr_init  = lr_init
        self.lr_final = lr_final
        self.nsteps   = nsteps
        self.delta    = (lr_final - lr_init)/(nsteps)
    def __call__(self,i):
        return self.lr_init + self.delta*i

def custom_lr_scheduler_v1(epoch,i,nsteps): 
    if epoch < 1:
        lr_scheduler = Linear_LR(8e-5,2e-4,nsteps)
        return lr_scheduler(i)
    elif epoch < 2:
        lr_scheduler = Linear_LR(2e-4,1e-4,nsteps)
        return lr_scheduler(i)
    elif epoch < 3:
        lr_scheduler = Linear_LR(1e-4,5e-5,nsteps)
        return lr_scheduler(i)
    elif epoch < 4:
        lr_scheduler = Linear_LR(5e-5,2.5e-5,nsteps)
        return lr_scheduler(i)
    elif epoch < 5:
        lr_scheduler = Linear_LR(2.5e-5,1.25e-5,nsteps)
        return lr_scheduler(i)
    elif epoch < 6:
        lr_scheduler = Linear_LR(1.25e-5,6.25e-6,nsteps)
        return lr_scheduler(i)
    elif epoch < 7:
        lr_scheduler = Linear_LR(6.25e-6,3.125e-06,nsteps)
        return lr_scheduler(i)

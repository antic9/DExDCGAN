from types import SimpleNamespace

_common = {
    'ngpu':        1,     # Number of GPUs available. Use 0 for CPU mode.
}

_cnn = {
    'image_size':  64,
    'nc':          3,     # inchannels (RGB)
    'nz':          256,   # Latent space dim
    'ngf':         128,   # Number of feature maps in generator
    'ndf':         128,   # Number of feature maps in discriminator
    'num_epoch':   100, 
    'lr':          0.002, # Learning rate
    'beta1':       0.5,   # Beta1 hyperparam for Adam optimizers
}

HPS = {
    'cnn':                  SimpleNamespace(**(_common | _cnn)),
}
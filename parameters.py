from types import SimpleNamespace

_common = {
    'ngpu':        1,     # Number of GPUs available. Use 0 for CPU mode.
}

_cnn = {
    'image_size':  64,
    'nc':          3,     # inchannels (RGB)
    'nz':          128,   # Latent space dim
    'ngf':         64,   # Number of feature maps in generator
    'ndf':         64,   # Number of feature maps in discriminator
    'num_epochs':   100, 
    'lr':          0.002, # Learning rate
    'beta1':       0.5,   # Beta1 hyperparam for Adam optimizers
    'batch_size':   256,
}

HPS = {
    'cnn':                  SimpleNamespace(**(_common | _cnn)),
}
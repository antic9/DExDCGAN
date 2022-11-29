from types import SimpleNamespace

_common = {
    'ngpu':        1,     # Number of GPUs available. Use 0 for CPU mode.
}

_cnn = {
    'image_size':  64,
    'nc':          3,     # inchannels (RGB)
    'nz':          100,   # Latent space dim
    'ngf':         64,   # Number of feature maps in generator
    'ndf':         64,   # Number of feature maps in discriminator
    'num_epochs':   100, 
    'lr':          0.02, # Learning rate
    'beta1':       0.5,   # Beta1 hyperparam for Adam optimizers
    'batch_size':   1024,
}

HPS = {
    'cnn':                  SimpleNamespace(**(_common | _cnn)),
}
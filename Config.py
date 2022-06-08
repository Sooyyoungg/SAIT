class Config:
    ## dataset parameters
    train_list = '../SAIT_Data/train_list.csv'
    val_list = '../SAIT_Data/valid_list.csv'
    test_list = '../SAIT_Data/test_list.csv'
    train_root = '../SAIT_Data/Train/'
    valid_root = '../SAIT_Data/Validation/'
    test_root = '../SAIT_Data/Test/'

    ## basic parameters
    gpu_ids = []
    n_epoch = 100
    n_iter = 100
    n_iter_decay = 100
    batch_size = 32
    lr = 0.0002
    lr_policy = 'linear'
    lr_decay_iters = 50
    beta1 = 0.5
    pool_size = 50
    image_display_iter = 100
    gan_mode = 'lsgan'

    input_nc = 1
    output_nc = 1
    ngf = 64
    ndf = 16
    netG = 'unet_256'
    netD = 'basic'
    n_layers_D = 3
    norm = 'instance'   # [instance | batch | none]
    init_type = 'normal' # [normal | xavier | kaiming | orthogonal]
    init_gain = 0.02    # scaling factor for normal, xavier and orthogonal
    no_dropout = 'store_true'   # no dropout for generator
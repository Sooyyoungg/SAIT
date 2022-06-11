class Config:
    ## dataset parameters
    data_name = 'SEM'
    train_list = './Dataset/train_list.csv'
    valid_list = './Dataset/val_list.csv'
    test_list = './Dataset/test_list.csv'
    train_root = './Dataset/Train/'
    valid_root = './Dataset/Validation/'
    test_root = './Dataset/Test/'

    # output directory
    log_dir = 'Pix2Pix/log'
    img_dir = 'Pix2Pix/Generated_images'
    test_img_dir = 'Pix2Pix/Tested_images'
    self_supervised_dir = './Self_log'

    ## basic parameters
    gpu_ids = [5]
    n_epoch = 100
    n_iter = 100
    n_iter_decay = 100
    batch_size = 64
    lr = 0.0005
    lr_policy = 'step'
    lr_decay_iters = 50
    beta1 = 0.5
    pool_size = 50
    image_display_iter = 100
    gan_mode = 'wgangp'

    # model parameters
    input_nc = 1
    output_nc = 1
    ngf = 64
    ndf = 64
    netG = 'unet_256'
    netD = 'basic'
    n_layers_D = 3
    norm = 'instance'   # [instance | batch | none]
    init_type = 'normal' # [normal | xavier | kaiming | orthogonal]
    init_gain = 0.02    # scaling factor for normal, xavier and orthogonal
    no_dropout = 'store_true'   # no dropout for generator
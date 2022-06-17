class Config:
    ## dataset parameters
    data_name = 'SEM'
    train_list = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/train_list.csv'
    valid_list = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/val_list.csv'
    test_list = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/test_list.csv'
    train_root = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/Train/'
    valid_root = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/Validation/'
    test_root = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/Test/'

    # output directory
    log_dir = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Pix2Pix/log/wgangp'
    img_dir = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Pix2Pix/Generated_images/wgangp'
    test_img_dir = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Pix2Pix/Tested_images'

    ## basic parameters
    gpu_ids = [7]
    n_epoch = 100
    n_iter = 100
    n_iter_decay = 100
    batch_size = 64
    lr = 0.0002
    lr_policy = 'step'
    lr_decay_iters = 50
    beta1 = 0.5
    pool_size = 50
    image_display_iter = 100
    gan_mode = 'wgangp'
    lambda_L1 = 100

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
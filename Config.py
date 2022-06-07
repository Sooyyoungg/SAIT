class Config:
    ## dataset parameters
    train_list = '../SAIT_Data/train_list.csv'
    val_list = '../SAIT_Data/valid_list.csv'
    test_list = '../SAIT_Data/test_list.csv'
    train_root = '../SAIT_Data/Train/'
    valid_root = '../SAIT_Data/Validation/'
    test_root = '../SAIT_Data/Test/'

    ## basic parameters
    n_epoch = 100
    batch_size = 32
    lr = 0.00005
    gpu_ids = [0]
    image_display_iter = 100

    input_nc = 1
    output_nc = 1
    ngf = 16
    netG = 'unet_256'
    netD = 'basic'
    n_layers_D = 3
    norm = 'instance'   # [instance | batch | none]
    init_type = 'normal' # [normal | xavier | kaiming | orthogonal]
    init_gain = 0.02    # scaling factor for normal, xavier and orthogonal
    no_dropout = 'store_true'   # no dropout for generator
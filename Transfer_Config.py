class Transfer_Config:
    ## dataset parameters
    data_name = 'SEM'
    train_list = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/train_list.csv'
    valid_list = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/val_list.csv'
    valid_half_list = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/val_half_list.csv'
    test_list = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/test_list.csv'
    train_root = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/Train/'
    valid_root = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/Validation/'
    test_root = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Dataset/Test/'

    # output directory
    log_dir = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Pix2Pix/log'
    img_dir = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Pix2Pix/Generated_images'
    test_img_dir = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Pix2Pix/Tested_images'
    self_supervised_dir = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Self_log'

    gpu_ids = [3]
    model_transfer = 1

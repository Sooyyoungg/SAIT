#selects different type of CNN model. From https://github.com/czbiohub/noise2self/blob/master/models/models.py

from models.babyunet import BabyUnet
from models.dncnn import DnCNN
from models.singleconv import SingleConvolution
from models.unet import Unet
import Pix2Pix.networks

def get_model(name, in_channels, out_channels, **kwargs):
    if name == "unet":
        return Unet(in_channels, out_channels)
    if name == "baby-unet":
        return BabyUnet(in_channels, out_channels)
    if name == "dncnn":
        return DnCNN(in_channels, out_channels)
    if name == "convolution":
        return SingleConvolution(in_channels, out_channels, kwargs["width"])
    if name == "pix2pix":
        return Pix2Pix.networks.UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

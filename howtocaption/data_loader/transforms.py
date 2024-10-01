from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from howtocaption.data_loader.utils import MyRandomResizedCrop


def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        grayscale_p=0,
                        norm_mean=(0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711),
                        antialias=True,
                        use_old_wrong_version=False):
    if use_old_wrong_version:
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)
        antialias = False

    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            MyRandomResizedCrop(input_res, scale=randcrop_scale, interpolation=InterpolationMode.BICUBIC, antialias=antialias),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            transforms.RandomGrayscale(p=grayscale_p),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_res, interpolation=InterpolationMode.BICUBIC, antialias=antialias),
            transforms.CenterCrop(input_res),
            normalize,
        ]),
        'test_resize': transforms.Compose([  # TODO: this might be just test
            transforms.Resize(center_crop, interpolation=InterpolationMode.BICUBIC, antialias=antialias),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res, interpolation=InterpolationMode.BICUBIC, antialias=antialias),
            normalize,
        ]),
        'visualization': transforms.Compose([
            transforms.Resize(240, interpolation=InterpolationMode.BICUBIC, antialias=antialias),
            transforms.CenterCrop((240, 320)),
            # normalize,
        ]),
        'none': None,
    }
    return tsfm_dict
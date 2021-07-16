import albumentations
from albumentations.pytorch import ToTensorV2
def load_training_augmentations(image_size,configuration):
    transformTrain=None
    if configuration=="Base":
        transformTrain = albumentations.Compose(
        [
            albumentations.Resize(int(image_size+image_size*0.1), int(image_size+image_size*0.1)),
            albumentations.RandomCrop(image_size, image_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            albumentations.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            albumentations.RandomBrightnessContrast(p=0.5),
            albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    elif configuration=="Huge":
        transformTrain = albumentations.Compose(
    [
        albumentations.Resize(int(image_size+image_size*0.1), int(image_size+image_size*0.1)),
        albumentations.RandomCrop(image_size, image_size),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
      #  A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5),
        albumentations.Blur(p=0.5),
        albumentations.MedianBlur(p=0.5),
#        A.CLAHE(p=0.5),
        albumentations.Downscale(p=0.5),
        albumentations.ElasticTransform(p=0.5),
        albumentations.OpticalDistortion(p=0.5),
        albumentations.GridDistortion(p=0.5),
        albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)
            
    return(transformTrain)
def load_valid_augmentations(image_size,configuration):
    transformValid=None
    if configuration=="Base":
        transformValid = albumentations.Compose(
            [
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
    elif configuration=="Huge":
        transformValid = albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    return(transformValid)
from monai.transforms import Compose, Activationsd, AsDiscreted, EnsureTyped


def get_post_transforms():
    return Compose([
        EnsureTyped(keys=["pred"]),
        Activationsd(keys=["pred"], softmax=True),
        AsDiscreted(keys=["pred"], argmax=True),
    ])

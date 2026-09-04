"""U-Net de segmentation avec encodeur pré-entraîné (transfer learning)."""
import segmentation_models_pytorch as smp


def build_model(encoder_name="resnet34", encoder_weights="imagenet"):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )

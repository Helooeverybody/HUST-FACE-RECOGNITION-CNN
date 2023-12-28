import cv2
from pathlib import Path


def get_representation(label):
    """Return an image representation of a label"""

    img = cv2.imread("./Assets/UI/unknown.png")
    img_path = Path(f"./Assets/Faces/{label}.jpg")
    if img_path.exists():
        img = cv2.imread(str(img_path))
    return img

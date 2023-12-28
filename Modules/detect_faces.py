import cv2
from pathlib import Path
from typing import NewType
import numpy as np

MatLike = NewType("MatLike", np.array)

cascade_path = (
    Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
)
classifier = cv2.CascadeClassifier(str(cascade_path))


def detect_faces(
    frame: MatLike,
    scale_factor=1.3,
    min_neighbors=5,
    min_size=(30, 30),
):
    """Return all detected faces."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        image=gray_frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces

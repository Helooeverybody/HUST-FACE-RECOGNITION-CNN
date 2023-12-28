from pathlib import Path
from Models import CNN_Model
import cv2

labels = CNN_Model().labels
out = "./face_check.txt"
face_dir = Path("./Assets/Faces")


with open(out, "w") as file:
    for label in labels:
        img_path = face_dir / f"{label}.jpg"
        exist = img_path.exists()
        shape = None
        if exist:
            img_shape = cv2.imread(str(img_path)).shape
            if img_shape[0] == img_shape[1]:
                shape = "square"
            else:
                shape = "not square"
        file.write(f"{label:20}{str(exist):10}{shape}\n")

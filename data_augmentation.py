from keras import Sequential
from keras.layers import RandomBrightness, RandomFlip, RandomZoom, RandomRotation
import cv2
import matplotlib.pyplot as plt

data_augmentation = Sequential(
    [
        RandomBrightness((-0.3, 0.3)),
        RandomFlip("horizontal"),
        RandomZoom((-0.3, 0.3)),
        RandomRotation(0.12),
    ],
    name="augmentation",
)

plt.figure(figsize=(10, 10))
image = cv2.imread("./Assets/Faces/Duy Dat.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for i in range(9):
    augmented_images = data_augmentation([image])
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
plt.show()

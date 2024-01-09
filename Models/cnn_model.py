from typing import NewType
from keras.layers import RandomZoom, RandomBrightness
from keras.models import load_model
import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt

MatLike = NewType("MatLike", np.array)

LABELS = [
    "Alex Lawther",
    "Amber Heard",
    "Angelina Jolie",
    "Avicii",
    "Ben Affleck",
    "Benzema",
    "Bill Gates",
    "Brie Larson",
    "Calvin Harris",
    "Chipu",
    "Chris Evans",
    "Chris Martin",
    "Chris Pratt",
    "Cillian Murphy",
    "Claudia Salas",
    "Cristiana Ronaldo",
    "Dakota Johnson",
    "Dat TaHo",
    "David Beckham",
    "Doug McMillion",
    "Drake",
    "Duy Dat",
    "Elizabeth Olsen",
    "Elle Fanning",
    "Elon Musk",
    "Emilia Clarke",
    "Emily Blunt",
    "Gal Gadot",
    "Gordon Ramsey",
    "Hanni NewJeans",
    "Henry Cavill",
    "Ho Ngoc Ha",
    "Irene",
    "JKRowling",
    "Jack Ma",
    "Jackie Chan",
    "Jason Momoa",
    "Jeff Bezos",
    "Jenna Ortega",
    "Jennie Blackpink",
    "Jennifer Lawrence",
    "Jessica Barden",
    "Jisoo Blackpink",
    "Joe Biden",
    "Karen Gillan",
    "Keanu Reeves",
    "Kit Harington",
    "LeBron James",
    "Leonardo Dicaprio",
    "Lewis Hamilton",
    "Lily Collins",
    "Lisa Blackpink",
    "Marc Marquez",
    "Marie Curie",
    "Mark Wahlberg",
    "Megan Fox",
    "Messi",
    "Micheal B Jordan",
    "Mina Twice",
    "Mohamed Salah",
    "Nancy Momoland",
    "Nick Vujicic",
    "Olivia Rodrigo",
    "Oprah Winfrey",
    "Ozil",
    "Quan Nguyen",
    "Quynh Nguyen",
    "Robert Pattinson",
    "Rose Blackpink",
    "Rose Leslie",
    "Sam Claflin",
    "Shakira",
    "Sophie Turner",
    "Steven Spielberg",
    "Suzy Bae",
    "Taylor Swift",
    "Thang Doi",
    "Tom Cruise",
    "Tom Holland",
    "Tzuyu Twice",
    "Ursula Corbero",
    "Vladimir V Putin",
    "Willem Dafoe",
    "Yoona",
    "Yuna Itzy",
    "Zedd",
    "Cong Phuong",
    "Do My Linh",
    "Khanh Vy",
    "Luu Diec Phi",
    "Mai Phuong Thuy",
    "My Tam",
    "Ngo Bao Chau",
    "Nguyen Phu Trong",
    "Park Hang Seo",
    "Park Seo Joon",
    "Pham Nhat Vuong",
    "Tran Thanh",
    "Vu Duc Dam",
]
IMAGE_SIZE = (224, 224)
COLOR = cv2.COLOR_BGR2GRAY


# Input augmentation
def zoom(img, zoom_level):
    return (
        RandomZoom(
            (zoom_level,) * 2,
            fill_mode="reflect",
        )(img)
        .numpy()
        .astype("uint8")
    )


def brightness(img, brightness_level):
    return (
        RandomBrightness(
            (brightness_level,) * 2,
        )(img)
        .numpy()
        .astype("uint8")
    )


# Convert input image a color type
def convert(img: MatLike, color) -> MatLike:
    if color == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[None, :, :]


class CNN_Model:
    def __init__(self, json_path) -> None:
        """Create a CNN model instance from a json file."""
        with open(json_path, "r") as data:
            model_info = json.load(data)
            self.model = load_model(model_info["model_path"])
            self.name = model_info["name"]
            self.img_size = model_info["input_size"]
            self.img_color = model_info["color_mode"]
            self.labels = LABELS[:]

    def summary(self):
        """Return the summary of the model."""
        return self.model.summary()

    def predict(
        self,
        image: MatLike,
        zoom_depth: int = 3,
        brightness_levels: int = 3,
        threshold: float = 0.2,
        verbose=0,
    ):
        """Return the label and uncertainty of an image."""
        # Preprocess input to match the model's input size
        image = cv2.resize(image, self.img_size[:2])
        image = convert(image, self.img_color)

        # Perform prediction on multiple zoom levels
        img_batch = []
        for zoom_level in np.arange(0, 1, 1 / zoom_depth):
            for brightness_level in np.linspace(-0.2, 0.2, brightness_levels):
                augmented_image = zoom(image, zoom_level)
                augmented_image = brightness(augmented_image, brightness_level)
                img_batch.append(augmented_image)

        # Perform prediction and calculate matching label and uncertainty
        predict_list = []
        plt.figure(figsize=(brightness_levels, zoom_depth))
        for i, img in enumerate(img_batch):
            prediction = self.model.predict(img, verbose=verbose)
            arg_max = np.argmax(prediction)

            # The model will be able to recognize an image when the uncertainty
            # is less than the threshold
            uncertainty = sum(-p * math.log(p, 99) for p in prediction[0])
            label = "Unknown"
            if uncertainty <= threshold:
                label: str = LABELS[arg_max]
            print(label, uncertainty)
            predict_list.append((label, uncertainty))

            plt.subplot(zoom_depth, brightness_levels, i + 1)
            plt.imshow(img[0])
            plt.axis("off")
        plt.savefig("aug_input_images")
        return min(predict_list, key=lambda x: x[1])


if __name__ == "__main__":
    model = CNN_Model("./Models/JSON/cnn_model_3.json")
    model.summary()
    print("Number of labels:", len(LABELS))
    test_image = cv2.imread("./Assets/Faces/Thang Doi.jpg")
    prediction = model.predict(test_image, 5, 3)
    print(">>>", *prediction)
    while True:
        test_image = cv2.resize(test_image, (224, 224))
        cv2.imshow("Display", test_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

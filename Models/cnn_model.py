from typing import NewType
from keras.models import load_model
import numpy as np
import cv2
import json
import math

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
            self.img_size = model_info["input_size"]
            self.img_color = model_info["color_mode"]
            self.labels = LABELS

    def summary(self):
        """Return the summary of the model."""
        return self.model.summary()

    def predict(self, image: MatLike, threshold=0.2):
        """Return the label and uncertainty of an image."""
        # Preprocess input to match the model's input size
        image = cv2.resize(image, self.img_size[:2])
        image = convert(image, self.img_color)
        # Perform prediction and calculate matching label and uncertainty
        prediction = self.model.predict(image)
        arg_max = np.argmax(prediction)
        label = "Unknown"
        res = 0
        for p in prediction[0]:
            res -= p * math.log(p, 99)
        # The model will be able to recognize an image when the uncertainty
        # is less than some threshold
        if res <= threshold:
            label: str = LABELS[arg_max]
        return label, res


if __name__ == "__main__":
    model = CNN_Model("./Models/JSON/cnn_model_2.json")
    print("Number of labels:", len(LABELS))
    test_image = cv2.imread("./Assets/Faces/Keanu Reeves.jpg")
    prediction = model.predict(test_image)
    print(prediction)
    while True:
        test_image = cv2.resize(test_image, (224, 224))
        cv2.imshow("Display", test_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

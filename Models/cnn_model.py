from typing import NewType
from keras.models import load_model
import numpy as np
import cv2

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


class CNN_Model:
    def __init__(self, model_path="./Models/cnn_model_2.keras") -> None:
        self.model = load_model(model_path)
        self.labels = LABELS

    def summary(self):
        return self.model.summary()

    def predict(self, image: MatLike):
        image = cv2.resize(image, IMAGE_SIZE)
        image = cv2.cvtColor(image, COLOR)
        prediction = self.model.predict(image[None, :, :])
        arg_max = np.argmax(prediction)
        label: str = LABELS[arg_max]
        confidence: float = prediction[0][arg_max]
        return label, confidence


if __name__ == "__main__":
    model = CNN_Model()
    print("Number of labels:", len(LABELS))
    test_image = cv2.imread("./Assets/Faces/Drake.jpg")
    print(model.predict(test_image))
    while True:
        test_image = cv2.resize(test_image, (224, 224))
        cv2.imshow("Display", test_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

import cv2
from Models import CNN_Model
from Modules import detect_faces

model = CNN_Model("./Models/JSON/cnn_model_3.json")


def main_program(video_feed, flip):
    camera = cv2.VideoCapture(video_feed)

    while True:
        _, frame = camera.read()
        if flip:
            frame = cv2.flip(frame, 1)

        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord("q"):
            break

        faces = detect_faces(frame)
        for x, y, w, h in faces:
            detected_face = frame[y : y + h, x : x + w]
            label, uncertainty = model.predict(detected_face, 1, 1, 0.25)
            text = label + " " + f"{uncertainty :.2%}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img=frame,
                text=text,
                org=(x, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1,
            )
        cv2.imshow("Live Recognition", frame)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam_id = 1
    main_program(cam_id, flip=False)

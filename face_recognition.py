from Models import CNN_Model
from Modules import detect_faces
from Modules import get_representation
import cv2
import cvzone


def draw_result(frame, label, uncertainty):
    """Handle drawing result."""
    # Draw label
    cv2.putText(
        img=frame,
        text="Predict: " + label,
        org=(543, 325),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 255, 255),
        thickness=2,
    )
    # Draw confidence
    cv2.putText(
        img=frame,
        text="Uncertainty: " + f"{uncertainty :.2%}",
        org=(543, 375),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 255, 255),
        thickness=2,
    )


def main_program(video_feed=1, zoom_depth=5, threshold=0.2):
    # Set up camera and window
    camera = cv2.VideoCapture(video_feed)

    # Import UI assets
    ui_base = cv2.imread("./Assets/UI/base.png")
    ui_front = cv2.imread("./Assets/UI/bg.png", cv2.IMREAD_UNCHANGED)
    ui_dialog = cv2.imread("./Assets/UI/dialog.png", cv2.IMREAD_UNCHANGED)

    # Programme parameters
    dialoge_count = 0
    predict_img = False
    flip_image = True
    show_face = False

    # Import model
    model = CNN_Model("./Models/JSON/cnn_model_2.json")

    # Default output
    extracted_face = cv2.imread("./Assets/UI/unknown.png")
    box_img = cv2.imread("./Assets/UI/unknown.png")
    predict_label = "None"
    uncertainty = 0

    while True:
        # Capture face using cv2 face recognition
        _, frame = camera.read()
        if flip_image:
            frame = cv2.flip(frame, 1)

        # Detect faces from camera
        faces = detect_faces(frame, 1.3, 5, (100, 100))
        num_faces = 0

        # Draw rectangles around the detected faces
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            num_faces += 1

        # Basic control
        pressed_key = cv2.waitKey(1) & 0xFF
        # Quit the program
        if pressed_key == ord("q"):
            break
        # Flip camera
        elif pressed_key == ord("f"):
            flip_image = not flip_image
        # Capture face from camera
        elif pressed_key == ord("c"):
            if num_faces > 0:
                extracted_face = frame[y : y + h, x : x + w]
                predict_img = True
                if dialoge_count < 5:
                    dialoge_count += 1
        # Show extracted face
        elif pressed_key == ord("s"):
            show_face = not show_face

        # Make prediction on extracted face
        if predict_img:
            predict_label, uncertainty = model.predict(
                extracted_face,
                zoom_depth,
                threshold,
            )
            print(">", predict_label, uncertainty)
            predict_img = False

        # Draw UI
        frame = cv2.resize(frame, (362, 283))
        ui_base[106 : 106 + 283, 42 : 42 + 362] = frame
        if show_face:
            box_img = extracted_face
        else:
            box_img = get_representation(predict_label)
        box_img = cv2.resize(box_img, (196, 196))
        ui_base[86 : 86 + 196, 531 : 531 + 196] = box_img
        ui_base = cvzone.overlayPNG(ui_base, ui_front)
        if dialoge_count < 5:
            ui_base = cvzone.overlayPNG(ui_base, ui_dialog)
        draw_result(ui_base, predict_label, uncertainty)

        cv2.imshow("Face Recognition", ui_base)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_feed = 1
    zoom_depth = 5
    threshold = 0.4
    main_program(video_feed, zoom_depth, threshold)

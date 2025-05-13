from ultralytics import YOLO
import cv2
import os

#C:\MPR\dataset\test\images\armas--344-_jpg.rf.5ffa59743c7dd35cc4668f66f2701a36.jpg
#C:\MPR\test_video.mp4

def detect_img(source_path, model):

    results = model.predict(source=source_path, conf=0.5, save=True, project=r'results', stream=True)
    saved_to = []

    for r in results:
        boxes = r.boxes
        masks = r.masks
        keypoints = r.keypoints
        probs = r.probs
        obb = r.obb
        output_path = r.save_dir
        saved_to.append(output_path)
        print("Detection Completed!!\n")
        print('File Saved to:-', output_path)

    return saved_to

def detect_webcam(model):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot acess the webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the Frame!!")

        results = model.predict(source=frame, conf=0.5, stream=True)

        for r in results:
            annotated_frame = r.plot()
            cv2.imshow('Weapon Detection', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return None

def main():
    model = YOLO('models/detect/train/weights/best.pt')
    
    print("1.Any Img, Video or Youtube Video Link:-\n2.Webcam\n")
    source = int(input("Select a Source:-"))

    if source == 1:
        source_path = input("Enter Source Path:-")
        try:
            saves = detect_img(source_path=source_path, model=model)
            print("Images Saved Locations:-", saves)
        except FileNotFoundError:
            print("File not Found!")
        except Exception as e:
            print("Error:-", e)
    elif source == 2:
        detect_webcam(model=model)
    else:
        return print("Wrong Input!!")


if __name__ == '__main__':
    main()
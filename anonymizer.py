import mediapipe as mp
import cv2
import os
import argparse

def process_img(img, face_detection):
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Clip coordinates to avoid out-of-bounds errors
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            img[y1:y2, x1:x2] = cv2.blur(img[y1:y2, x1:x2], (30, 30))
    return img

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="webcam", help="Mode: image, video, or webcam")
parser.add_argument("--filePath", default="data/karan.mp4", help="Path to image or video file")
args = parser.parse_args()

# Load face detection
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    os.makedirs('output_dir', exist_ok=True)

    if args.mode == "image":
        img = cv2.imread(args.filePath)
        if img is None:
            raise ValueError(f"Image not found at path: {args.filePath}")
        img = process_img(img, face_detection)
        cv2.imwrite(os.path.join('output_dir', 'output.jpg'), img)

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.filePath)
        if not cap.isOpened():
            raise ValueError(f"Video not found or cannot be opened: {args.filePath}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_vid = cv2.VideoWriter(
            os.path.join('output_dir', 'output_vid.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_img(frame, face_detection)
            output_vid.write(processed_frame)

        cap.release()
        output_vid.release()

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        print("Press 'q' to quit webcam...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_img(frame, face_detection)
            cv2.imshow("Webcam Face Blur", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        raise ValueError("Invalid mode. Use 'image', 'video', or 'webcam'.")

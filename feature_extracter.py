import cv2
import csv
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("training/runs/detect/train10/weights/best.pt")  # Replace with your model if needed

# Path to the video file
video_path = "runs/detect/predict7/wet_book.avi"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Open files to save detections
txt_file_path = "detections.txt"
csv_file_path = "detections.csv"

with open(txt_file_path, "w") as txt_file, open(csv_file_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow(["Frame", "Label", "Confidence", "X1", "Y1", "X2", "Y2"])

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames are available

        # Run object detection on the frame
        results = model.predict(source=frame, save=False, save_txt=False)

        # Extract detections from results
        for result in results:
            for detection in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Top-left and bottom-right coordinates
                confidence = float(detection.conf[0])  # Confidence score
                label = result.names[int(detection.cls[0])]  # Class label

                # Save the data to the TXT file
                txt_file.write(f"Frame {frame_index}: Label={label}, Confidence={confidence}, BBox=({x1}, {y1}, {x2}, {y2})\n")

                # Save the data to the CSV file
                csv_writer.writerow([frame_index, label, confidence, x1, y1, x2, y2])

                # Optional: Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Optional: Show the frame with detections
        '''
         cv2.imshow("Detections", frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''

        frame_index += 1

cap.release()
cv2.destroyAllWindows()

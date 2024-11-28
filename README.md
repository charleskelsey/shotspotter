<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tracker Installation Guide</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            background-color: #f9f9f9;
            color: #333;
        }
        h1, h2, h3 {
            color: #0056b3;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 4px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-left: 3px solid #ddd;
            overflow-x: auto;
        }
        a {
            color: #0056b3;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .note {
            background-color: #fff3cd;
            border-left: 5px solid #ffeeba;
            padding: 10px;
            margin: 10px 0;
        }
        .troubleshooting {
            background-color: #f8d7da;
            border-left: 5px solid #f5c6cb;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Tracker Installation Guide</h1>
    <p>This guide provides instructions on how to set up and use the <code>Tracker</code> class for object tracking and detection in video frames. The code is built using the <strong>YOLO</strong> model from Ultralytics and integrates with <strong>supervision</strong> for advanced tracking functionality.</p>

    <h2>Prerequisites</h2>
    <ul>
        <li>Python 3.8 or later</li>
        <li><code>pip</code> for managing Python packages</li>
        <li>A supported GPU (recommended for YOLO model inference)</li>
        <li>OpenCV (for image processing)</li>
    </ul>

    <h2>Installation</h2>
    <ol>
        <li><strong>Clone or Download the Repository</strong></li>
        <li><strong>Install Required Dependencies:</strong></li>
        <pre><code>pip install ultralytics supervision opencv-python-headless numpy pandas</code></pre>
        <li><strong>Install Additional System Libraries:</strong></li>
        <ul>
            <li>Ubuntu/Debian:
                <pre><code>sudo apt update
sudo apt install libgl1-mesa-glx</code></pre>
            </li>
            <li>Windows:
                <p>Ensure your Python environment is properly configured to handle OpenCV and YOLO.</p>
            </li>
        </ul>
        <li><strong>Include Utility Functions:</strong>
            <p>Ensure the <code>utils.py</code> file is present in the project directory. It should contain the following helper functions:</p>
            <ul>
                <li><code>get_bbox_width</code></li>
                <li><code>get_center_of_bbox</code></li>
            </ul>
        </li>
        <li><strong>Verify the Installation:</strong></li>
        <pre><code>from ultralytics import YOLO
import cv2
print("Setup successful!")</code></pre>
    </ol>

    <h2>Usage</h2>
    <ol>
        <li><strong>Initialize the Tracker:</strong></li>
        <pre><code>tracker = Tracker(model_path='path/to/your/yolo/model.pt')</code></pre>
        <li><strong>Process Video Frames:</strong></li>
        <ul>
            <li><strong>Load Video Frames:</strong></li>
            <pre><code>import cv2

video_frames = []
cap = cv2.VideoCapture('input_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)
cap.release()</code></pre>
            <li><strong>Detect and Track Objects:</strong></li>
            <pre><code>tracks = tracker.get_object_tracks(video_frames)</code></pre>
        </ul>
        <li><strong>Annotate and Export Frames:</strong></li>
        <ul>
            <li><strong>Draw Annotations:</strong></li>
            <pre><code>annotated_frames = tracker.draw_annotations(video_frames, tracks)</code></pre>
            <li><strong>Save the Output Video:</strong></li>
            <pre><code>out = cv2.VideoWriter(
    'output_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (video_frames[0].shape[1], video_frames[0].shape[0])
)
for frame in annotated_frames:
    out.write(frame)
out.release()</code></pre>
        </ul>
    </ol>

    <h2>Sample Workflow</h2>
    <pre><code>from Tracker import Tracker
import cv2

# Initialize the Tracker
tracker = Tracker(model_path='yolov8n.pt')

# Load video frames
video_frames = []
cap = cv2.VideoCapture('input_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)
cap.release()

# Detect and track objects
tracks = tracker.get_object_tracks(video_frames)

# Annotate video
annotated_frames = tracker.draw_annotations(video_frames, tracks)

# Save the output
out = cv2.VideoWriter(
    'output_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (video_frames[0].shape[1], video_frames[0].shape[0])
)
for frame in annotated_frames:
    out.write(frame)
out.release()</code></pre>


</body>
</html>

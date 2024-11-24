# Import necessary libraries
import pandas as pd
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt

# Configuration
VIDEO_PATH = 'wet_book.mp4'  # Input video file path
DETECTIONS_CSV = 'detections.csv'  # Input detections CSV file
OUTPUT_CSV = 'trajectories.csv'  # Output trajectory CSV file
OUTPUT_VIDEO = 'trajectory_overlay.mp4'  # Output video file with trajectory

# Initialize DeepSORT
deepsort = DeepSort(
    max_age=80,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.7,
    nn_budget=None,
    embedder='mobilenet',
    embedder_model_name='mobilenet_v2',
    half=True,
    embedder_gpu=False,
    polygon=False
)

# Load detections
try:
    detections_df = pd.read_csv(DETECTIONS_CSV)
    print(f"Detections loaded from '{DETECTIONS_CSV}'")
except FileNotFoundError:
    print(f"Error: File '{DETECTIONS_CSV}' not found.")
    exit()

# Filter detections to include only 'basketball'
basketball_detections_df = detections_df[detections_df['Label'] == 'basketball']
print(f"Filtered detections: {len(basketball_detections_df)} basketball detections found.")

# Initialize list to store trajectories
trajectories = []

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Unable to open video file '{VIDEO_PATH}'")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video resolution: {frame_width}x{frame_height} at {fps} FPS")
print(f"Total frames in video: {total_frames}")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# Initialize variables for trajectory visualization
trajectory_points = {}

current_frame = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Retrieve detections for the current frame
    frame_detections = basketball_detections_df[basketball_detections_df['Frame'] == current_frame]

    detections = []
    for _, det in frame_detections.iterrows():
        try:
            x1, y1, x2, y2 = det['X1'], det['Y1'], det['X2'], det['Y2']
            confidence = det['Confidence']
            if pd.notnull(x1) and pd.notnull(y1) and pd.notnull(x2) and pd.notnull(y2) and pd.notnull(confidence):
                detections.append([float(x1), float(y1), float(x2), float(y2), float(confidence)])
        except Exception as e:
            print(f"Error processing detection at frame {current_frame}: {e}")

    # Filter low-confidence detections
    CONFIDENCE_THRESHOLD = 0.3
    detections = [d for d in detections if d[4] >= CONFIDENCE_THRESHOLD]

    # Transform detections for DeepSORT
    dets_transformed = [
        ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(confidence), 'basketball')
        for x1, y1, x2, y2, confidence in detections
    ]

    # Update tracker
    tracks = deepsort.update_tracks(dets_transformed, frame=frame)

    # Process tracks and store trajectory data
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb(orig=True)
        x1, y1, x2, y2 = bbox
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Add point to trajectory points
        if track_id not in trajectory_points:
            trajectory_points[track_id] = []
        trajectory_points[track_id].append((center_x, center_y))

        # Add data to trajectories list
        trajectories.append({
            'Frame': current_frame,
            'TrackID': track_id,
            'X1': x1,
            'Y1': y1,
            'X2': x2,
            'Y2': y2
        })

    # Draw the trajectory on the frame
    for track_id, points in trajectory_points.items():
        for i in range(1, len(points)):
            # Draw line connecting trajectory points
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)  # Green line for trajectory
            # Draw current point as a red circle
            cv2.circle(frame, points[i], 5, (0, 0, 255), -1)

    # Write the frame with the trajectory to the output video
    out.write(frame)

    # Print progress
    if current_frame % 100 == 0:
        print(f"Processed frame {current_frame}/{total_frames}")

    current_frame += 1

# Release resources
cap.release()
out.release()

# Save trajectory data to CSV
trajectories_df = pd.DataFrame(trajectories)
trajectories_df.to_csv(OUTPUT_CSV, index=False)
print(f"Trajectories saved to '{OUTPUT_CSV}'")
print(f"Trajectory video saved to '{OUTPUT_VIDEO}'")

# Plot trajectories
plt.figure(figsize=(10, 8))
for track_id in trajectories_df['TrackID'].unique():
    track_data = trajectories_df[trajectories_df['TrackID'] == track_id]
    plt.plot(track_data['X1'], track_data['Y1'], label=f'Track ID {track_id}')

plt.title("Trajectories of Tracked Basketballs")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
plt.show()

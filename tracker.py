from ultralytics import YOLO
import supervision as sv
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox

class Tracker:

    #The initialization fuction
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.score1 = 0
        self.score2 = 0
        self.net_states = {}  # To keep track of shot detection per net
        self.frame_width = None  # Will be set when processing frames

    # Interpolating the ball position based on the ball movement and position using the built-in interpolate and filling functions
    def interpolate_ball(self, ball_positions):
        #Converts the basketball positions into a pandas DataFrame.
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        #Uses interpolation (interpolate) to fill missing values and backward filling (bfill) to handle edge cases.
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        #ball_positions is a list of interpolated ball positions.
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    #Detects objects in a batch of video frames using the YOLO model.
    def detect_frames(self, frames):
        batch_size = 30 #The amount of video frames processed at a time
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1) #For detection low confidence is enough
            detections += detections_batch #Returns all detections for the input frames.
        return detections

    #Draws a triangle on the frame for marking the ball or net.
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox) #Taken from util files

        triangle_points = np.array([ #Defining the triangle vertices
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def get_object_tracks(self, frames): #Tracks objects (layers, referees, basketball, nets) across video frames.
        detections = self.detect_frames(frames) 
        self.frame_width = frames[0].shape[1]  # Set frame width based on first frame

        tracks = {
            "player": [],
            "referee": [],
            "basketball": [],
            "net": []
        }

        for frame_num, detection in enumerate(detections): #Runs object detection on all frames using detect_frames.
            cls_name = detection.names
            cls_names_inv = {v: k for k, v in cls_name.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            print(detection_with_tracks)

            #The detections are formatted into IDs to keep track of them
            tracks["player"].append({})
            tracks["basketball"].append({})
            tracks["referee"].append({})
            tracks["net"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['net']:
                    tracks["net"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['basketball']:
                    tracks["basketball"][frame_num][1] = {"bbox": bbox}

        return tracks

    #Draws an ellipse around a tracked object and optionally displays its track_id. To give it a video game look
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        #The values were assigned after trial and error for most asthetically pleasing style. 
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width / 2), int(0.35 * width / 2)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        #Used for tracking the players and referres. 
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame
    
    #Checks if two bounding boxes intersect.
    def bbox_intersects(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        #Computes the overlap in the x and y directions and returns true if the box intersects
        x_intersect = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_intersect = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

        return x_intersect > 0 and y_intersect > 0

    #Enlarges a bounding box by a specified scale factor. Set to 1 but was changed during trials
    def enlarge_bbox(self, bbox, scale=1):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        new_width = width * scale
        new_height = height * scale
        x1_new = x_center - new_width / 2
        x2_new = x_center + new_width / 2
        y1_new = y_center - new_height / 2
        y2_new = y_center + new_height / 2
        return [x1_new, y1_new, x2_new, y2_new]

    def detect_shots(self, tracks):
        #Tracks the ball's position across frames and stores its bounding box and center.
        ball_positions = []
        for frame_num in range(len(tracks['basketball'])):
            ball_frame = tracks['basketball'][frame_num]
            if ball_frame:
                ball_bbox = ball_frame[1]['bbox']
                ball_center_x, ball_center_y = get_center_of_bbox(ball_bbox)
                ball_positions.append({'frame_num': frame_num, 'center_y': ball_center_y, 'bbox': ball_bbox})
            else:
                ball_positions.append(None)

        net_states = {}  # {'left_net': state, 'right_net': state}

        tracks['shot_events'] = []

        window_size = 5  
        for frame_num in range(len(tracks['basketball'])):
            ball_data = ball_positions[frame_num]
            if not ball_data:
                continue

            window_start = max(0, frame_num - window_size)
            window_end = min(len(ball_positions), frame_num + window_size)
            ball_window = ball_positions[window_start:window_end]
            ball_window = [b for b in ball_window if b is not None]

            if len(ball_window) < 2:
                continue
            #Each net is inialized based on if it is on the right or left side of the camera screen. 
            dy_list = []
            for i in range(1, len(ball_window)):
                dy = ball_window[i]['center_y'] - ball_window[i - 1]['center_y']
                dy_list.append(dy)
            avg_dy = np.mean(dy_list)

            net_frame = tracks['net'][frame_num]
            for net_id, net_data in net_frame.items():
                net_bbox = net_data['bbox']
                enlarged_net_bbox = self.enlarge_bbox(net_bbox, scale=1.5)  

                intersects = self.bbox_intersects(ball_data['bbox'], enlarged_net_bbox)

                # Decide whether this net is 'net1' (right half) or 'net2' (left half)
                net_center_x, _ = get_center_of_bbox(net_bbox)
                frame_width = self.frame_width

                if net_center_x > frame_width / 2:
                    net_key = 'net1'  # Right half of the screen
                else:
                    net_key = 'net2'  # Left half of the screen

                if net_key not in net_states:
                    net_states[net_key] = {'state': 'idle', 'buffer': []}

                net_state = net_states[net_key]

                net_state['buffer'].append({'frame_num': frame_num, 'intersects': intersects, 'avg_dy': avg_dy})
                if len(net_state['buffer']) > window_size * 2 + 1:
                    net_state['buffer'].pop(0)
                #The ball trajectory is taken into account. The ball should be on the way down for it to be considered a shot made
                intersects_list = [b['intersects'] for b in net_state['buffer']]
                dy_list = [b['avg_dy'] for b in net_state['buffer'] if b['intersects']]
                if not dy_list:
                    continue
                
                #The score is updated for both the nets
                avg_dy_intersects = np.mean(dy_list)
                if net_state['state'] == 'idle':
                    if any(intersects_list):
                        if avg_dy_intersects > 0:
                            net_state['state'] = 'ball_intersecting'
                elif net_state['state'] == 'ball_intersecting':
                    if avg_dy_intersects > 0:
                        net_y2 = net_bbox[3]
                        if ball_data['center_y'] > net_y2:
                            net_state['state'] = 'shot_made'
                            tracks['shot_events'].append({'net_key': net_key, 'event': 'shot_made', 'frame_num': frame_num})
                            if net_key == 'net1':
                                self.score1 += 1
                            else:
                                self.score2 += 1
                            net_state['state'] = 'idle'
                            net_state['buffer'] = []
                    else:
                        net_state['state'] = 'shot_missed'
                        tracks['shot_events'].append({'net_key': net_key, 'event': 'shot_missed', 'frame_num': frame_num})
                        net_state['state'] = 'idle'
                        net_state['buffer'] = []
    #This function draws the annotations which are ellipses and triangles.
    def draw_annotations(self, video_frames, tracks):
        output_vid_frames = []
        shot_events = {event['frame_num']: event for event in tracks.get('shot_events', [])}
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            #Each array of object is given a name and this is done for each frame. 
            player_dict = tracks["player"][frame_num]
            ball_dict = tracks["basketball"][frame_num]
            ref_dict = tracks["referee"][frame_num]
            net_dict = tracks["net"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)

            # Draw referees
            for ref_id, referee in ref_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), ref_id)

            # Draw ball
            for ball_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw nets
            for hoop_id, hoop in net_dict.items():
                frame = self.draw_triangle(frame, hoop["bbox"], (0, 0, 255))

            # Draw shot events
            for offset in range(-5, 6):  # Consider frames before and after
                event_frame_num = frame_num + offset
                if event_frame_num in shot_events:
                    event = shot_events[event_frame_num]
                    if event['event'] == 'shot_made':
                        label = 'Shot Made'
                        color = (0, 255, 0)
                    elif event['event'] == 'shot_missed':
                        label = 'Shot Missed'
                        color = (0, 0, 255)
                    else:
                        label = ''
                        color = (255, 255, 255)
                    # Draw label on the frame
                    if label:
                        ball_frame = tracks['basketball'][frame_num]
                        if ball_frame:
                            ball_bbox = ball_frame[1]['bbox']
                            x, y = get_center_of_bbox(ball_bbox)
                            cv2.putText(frame, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Draw scores
            cv2.putText(frame, f'Score1: {self.score1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Score2: {self.score2}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            output_vid_frames.append(frame)

        return output_vid_frames
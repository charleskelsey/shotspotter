from ultralytics import YOLO
import supervision as sv
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox

class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.score1 = 0
        self.score2 = 0
        self.net_states = {}  # To keep track of shot detection per net

    # Interpolating the ball position
    def interpolate_ball(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 30
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        tracks = {
            "player": [],
            "referee": [],
            "basketball": [],
            "net": []
        }

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_names_inv = {v: k for k, v in cls_name.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            print(detection_with_tracks)

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

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

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

    def bbox_intersects(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        x_intersect = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_intersect = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

        return x_intersect > 0 and y_intersect > 0

    def enlarge_bbox(self, bbox, scale=1.5):
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
        ball_positions = []
        for frame_num in range(len(tracks['basketball'])):
            ball_frame = tracks['basketball'][frame_num]
            if ball_frame:
                ball_bbox = ball_frame[1]['bbox']
                ball_center_x, ball_center_y = get_center_of_bbox(ball_bbox)
                ball_positions.append({'frame_num': frame_num, 'center_y': ball_center_y, 'bbox': ball_bbox})
            else:
                ball_positions.append(None)

        net_ids = set()
        for frame_nets in tracks['net']:
            net_ids.update(frame_nets.keys())

        net_states = {net_id: {'state': 'idle', 'last_ball_y': None, 'buffer': []} for net_id in net_ids}

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

            dy_list = []
            for i in range(1, len(ball_window)):
                dy = ball_window[i]['center_y'] - ball_window[i - 1]['center_y']
                dy_list.append(dy)
            avg_dy = np.mean(dy_list)

            net_frame = tracks['net'][frame_num]
            for net_id, net_data in net_frame.items():
                net_state = net_states[net_id]
                net_bbox = net_data['bbox']
                enlarged_net_bbox = self.enlarge_bbox(net_bbox, scale=1.5)  

                intersects = self.bbox_intersects(ball_data['bbox'], enlarged_net_bbox)

                net_state['buffer'].append({'frame_num': frame_num, 'intersects': intersects, 'avg_dy': avg_dy})
                if len(net_state['buffer']) > window_size * 2 + 1:
                    net_state['buffer'].pop(0)

                intersects_list = [b['intersects'] for b in net_state['buffer']]
                dy_list = [b['avg_dy'] for b in net_state['buffer'] if b['intersects']]
                if not dy_list:
                    continue

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
                            tracks['shot_events'].append({'net_id': net_id, 'event': 'shot_made', 'frame_num': frame_num})
                            if net_id == 1:
                                self.score1 += 1
                            else:
                                self.score2 += 1
                            net_state['state'] = 'idle'
                            net_state['buffer'] = []
                    else:
                        net_state['state'] = 'shot_missed'
                        tracks['shot_events'].append({'net_id': net_id, 'event': 'shot_missed', 'frame_num': frame_num})
                        net_state['state'] = 'idle'
                        net_state['buffer'] = []

    def draw_annotations(self, video_frames, tracks):
        output_vid_frames = []
        shot_events = {event['frame_num']: event for event in tracks.get('shot_events', [])}
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["player"][frame_num]
            ball_dict = tracks["basketball"][frame_num]
            ref_dict = tracks["referee"][frame_num]
            net_dict = tracks["net"][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)

            for ref_id, referee in ref_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), ref_id)

            for ball_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            for hoop_id, hoop in net_dict.items():
                frame = self.draw_triangle(frame, hoop["bbox"], (0, 0, 255))

            for offset in range(-5, 6):  
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
                    if label:
                        ball_frame = tracks['basketball'][frame_num]
                        if ball_frame:
                            ball_bbox = ball_frame[1]['bbox']
                            x, y = get_center_of_bbox(ball_bbox)
                            cv2.putText(frame, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(frame, f'Score1: {self.score1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Score2: {self.score2}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            output_vid_frames.append(frame)

        return output_vid_frames

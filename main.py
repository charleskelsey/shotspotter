from utils import read_video, save_video
from tracker import Tracker
import torch

torch.cuda.empty_cache()

def main():
    #Read the input mp4 video
    video_frames = read_video('chunk_002.mp4')
    
    #The model is trained for detection by the best epoch of the training model
    tracker = Tracker('training/runs/detect/train10/weights/best.pt')

    #The objects are called and put into their lists for each video frame
    tracks = tracker.get_object_tracks(video_frames)
    tracker.detect_shots(tracks)

    #The basketball is interpolated using interpolate_ball function in tracker.py
    tracks["basketball"] = tracker.interpolate_ball(tracks["basketball"])

    #Game like annotations are drawn for each object in each frame and the output video is saved.
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, 'output_vid/infer3_27oct.avi')

if __name__ == '__main__':
    main()

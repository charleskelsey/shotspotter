from utils import read_video, save_video
from tracker import Tracker
import torch

torch.cuda.empty_cache()

def main():
    video_frames = read_video('input_vid/wet_book.mp4')
    
    tracker = Tracker('training/runs/detect/train10/weights/best.pt')

    tracks = tracker.get_object_tracks(video_frames)
    tracker.detect_shots(tracks)

    tracks["basketball"] = tracker.interpolate_ball(tracks["basketball"])

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, 'output_vid/test_book_made.avi')

if __name__ == '__main__':
    main()
# Import dependencies
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, CourtLineTracker
from mini_court import MiniCourt

def main():
    # Read input video
    input_video_path = "input_video/aus_open2024.mp4"
    video_frames = read_video(input_video_path)
    
    # Optional: Add mask

    # CNN model 1 detects players (player_yolov8m.pt)
    player_model_path = "models/player_yolov8m.pt"
    player_detection_stub_path = "tracker_stubs/player_detections.pkl"
    
    model_1 = PlayerTracker(player_model_path)
    player_detections = model_1.detect_frames(video_frames,
                                              read_from_stub = True,
                                              stub_path = "tracker_stubs/player_detections_aus_open2024.pkl"
                                              )

    # CNN model 2 detects tennis ball (ball_yolov8n_best.pt)
    ball_model_path = "models/ball_yolov8n_best.pt"
    ball_detection_stub_path = "tracker_stubs/ball_detections.pkl"
    
    model_2 = BallTracker(ball_model_path)
    ball_detections = model_2.detect_frames(video_frames,
                                            read_from_stub = True,
                                            stub_path = "tracker_stubs/ball_detections_aus_open2024.pkl"
                                            )
    interpolate_ball_detections = model_2.interpolate_ball_positions(ball_detections)

    # CNN model 3 detects keypoints (tennis_court_keypoints_resnet50.pth)
    court_model_path = "models/tennis_court_keypoints_resnet50.pth"
    
    model_3 = CourtLineTracker(court_model_path)
    keypoints = model_3.predict_on_video(video_frames)

    # Filter players to detect (2 players on court)
    filtered_player_detection = model_1.choose_and_filter_players(keypoints, player_detections)

    # Draw bbox of players
    output_video_frames = model_1.draw_bboxes(video_frames, filtered_player_detection)

    # Draw bbox of tennis ball
    output_video_frames = model_2.draw_bboxes(output_video_frames, interpolate_ball_detections)

    # Draw keypoints of tennis court
    output_video_frames = model_3.draw_kps_on_video(output_video_frames, keypoints)

    # Draw miniature tennis court
    mini_court = MiniCourt(output_video_frames[0])
    mini_court.get_mini_court_coordinates(interpolate_ball_detections, filtered_player_detection, keypoints)
    output_video_frames = mini_court.draw_on_video(output_video_frames)

    # Draw frame number on video
    import cv2
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save output video
    output_video_path = "output_video/output_video.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()
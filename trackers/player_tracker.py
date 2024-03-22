from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append("../")
from utils import get_bbox_midpoint, distance_between_points

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results = self.model.track(frame, persist = True)
        
        player_dict = {}
        id_name_dict = results[0].names
        for box in results[0].boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
                
        return player_dict
    
    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        player_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
            
        return player_detections
    
    def draw_bboxes(self, frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                
            output_video_frames.append(frame)
            
        return output_video_frames
    
    # def choose_players(self, kps, player_dict):
    #     """
    #     Chooses players based on their absolute minimum distance to a nearest keypoint.
    #     """
    #     distances = []
    #     for track_id, bbox in player_dict.items():
    #         bbox_mid = get_bbox_midpoint(bbox)
            
    #         min_distance = float("inf")
    #         for i in range(0, len(kps), 2):
    #             keypoint = (kps[i], kps[i+1])
    #             distance = distance_between_points(bbox_mid, keypoint)
    #             if distance < min_distance:
    #                 min_distance = distance
    #         distances.append((track_id, min_distance))
        
    #     distances.sort(key = lambda x: x[1])
    #     chosen_players_id = [distances[0][0], distances[1][0]]
        
    #     return chosen_players_id

    # def choose_and_filter_players(self, kps, player_detections):
    #     """
    #     Chooses and filters players based on the first frame.
    #     """
    #     player_detection_first_frame = player_detections[0]
    #     kps_first_frame = kps[0]
    #     chosen_player_id = self.choose_players(kps_first_frame, player_detection_first_frame)
        
    #     filtered_player_detections = []
    #     for player_detected in player_detections:            
    #         filtered_player_detections.append({id: bbox for id, bbox in player_detected.items() if id in chosen_player_id})
        
    #     return filtered_player_detections
    
    def choose_players(self, kps, player_dict):
        """
        Chooses players based on their total distance to all keypoints.
        """
        distances = []
        for track_id, bbox in player_dict.items():
            bbox_mid = get_bbox_midpoint(bbox)
            total_distance = 0
            for i in range(0, len(kps), 2):
                keypoint = (kps[i], kps[i+1])
                distance = distance_between_points(bbox_mid, keypoint)
                total_distance += distance
            distances.append((track_id, total_distance))
        
        distances.sort(key = lambda x: x[1])
        chosen_players_id = [distances[0][0], distances[1][0]]
        
        return chosen_players_id
    
    def choose_and_filter_players(self, kps, player_detections):
        """
        Chooses and filters players for every frame.
        """
        filtered_player_detections = []
        for keypoint, player_detected in zip(kps, player_detections):
            chosen_player_id = self.choose_players(keypoint, player_detected)
            filtered_player_detections.append({id: bbox for id, bbox in player_detected.items() if id in chosen_player_id})
        
        return filtered_player_detections
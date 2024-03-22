import cv2
import sys
import numpy as np
sys.path.append('../')
import constants
from utils import (convert_pixel_distance_to_meters,
                   convert_meters_to_pixel_distance,
                   get_foot_position,
                   distance_between_points)

class MiniCourt():
    def __init__(self, frame):
        self.canvas_width = 250
        self.canvas_height = 600
        self.buffer_x = 50
        self.buffer_y = 100
        self.padding_court_x = 20
        self.padding_court_y = 70
        
        self.set_canvas_position(frame)
        self.set_mini_court_position()
        self.set_court_keypoints()
        self.set_court_lines()
    
    def set_canvas_position(self, frame):
        frame = frame.copy()
        self.canvas_end_x = frame.shape[1] - self.buffer_x
        self.canvas_end_y = self.buffer_y + self.canvas_height
        self.canvas_start_x = self.canvas_end_x - self.canvas_width
        self.canvas_start_y = self.canvas_end_y - self.canvas_height
        
    def set_mini_court_position(self):
        self.court_start_x = self.canvas_start_x + self.padding_court_x
        self.court_start_y = self.canvas_start_y + self.padding_court_y
        self.court_end_x = self.canvas_end_x - self.padding_court_x
        # self.court_end_y = self.canvas_end_y - self.padding_court
        
        self.court_width = self.court_end_x - self.court_start_x
        # self.court_height = self.court_end_y - self.court_start_y
        
    def convert_meters_pixel(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_width)
        
    def set_court_keypoints(self):
        drawing_key_points = [0] * 28
        
        drawing_key_points[0], drawing_key_points[1] = self.court_start_x, self.court_start_y
        
        drawing_key_points[2], drawing_key_points[3] = self.court_end_x, self.court_start_y
        
        drawing_key_points[4] = self.court_start_x
        drawing_key_points[5] = self.court_start_y + self.convert_meters_pixel(constants.HALF_COURT_LINE_HEIGHT * 2)

        drawing_key_points[6] = drawing_key_points[0] + self.court_width
        drawing_key_points[7] = drawing_key_points[5] 

        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_pixel(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 

        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_pixel(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 

        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_pixel(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_pixel(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 

        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_pixel(constants.NO_MANS_LAND_HEIGHT)

        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_pixel(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 

        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_pixel(constants.NO_MANS_LAND_HEIGHT)

        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_pixel(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 

        drawing_key_points[24] = (drawing_key_points[16] + drawing_key_points[18]) / 2
        drawing_key_points[25] = drawing_key_points[17] 

        drawing_key_points[26] = (drawing_key_points[20] + drawing_key_points[22]) / 2
        drawing_key_points[27] = drawing_key_points[21]
        
        drawing_key_points = [int(points) for points in drawing_key_points]

        self.drawing_key_points = drawing_key_points
        
    def set_court_lines(self):
        self.lines = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (12, 13)
        ]
        
    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.canvas_start_x, self.canvas_start_y), (self.canvas_end_x, self.canvas_end_y), (255, 255, 255), cv2.FILLED)
        output_frame = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        output_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        
        return output_frame
        
    def draw_court(self, frame):
        for line in self.lines:
            line_start = (self.drawing_key_points[line[0] * 2], self.drawing_key_points[(line[0] * 2) + 1])
            line_end = (self.drawing_key_points[line[1] * 2], self.drawing_key_points[(line[1] * 2) + 1])
            cv2.line(frame, line_start, line_end, (0, 0, 0), 2)
        
        net_start = (self.drawing_key_points[0] + 5, int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end = (self.drawing_key_points[2] - 5, int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start, net_end, (255, 255, 255), 2)
        
        for i in range(0, len(self.drawing_key_points), 2):
            x = self.drawing_key_points[i]
            y = self.drawing_key_points[i + 1]
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
        return frame
    
    def get_start_point_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_mini_court(self):
        return self.court_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def get_nearest_keypoint(self, player_foot_position, keypoint):
        nearest_keypoint = int(0)
        min_distance = float("inf")
        keypoint_to_check = [12, 13]
        
        for point in keypoint_to_check:
            keypoint_coordinates = (keypoint[(point) * 2], keypoint[((point) * 2) + 1])
            distance = distance_between_points(keypoint_coordinates, player_foot_position)
            if distance < min_distance:
                min_distance = distance
                nearest_keypoint = point
        
        return nearest_keypoint
    
    def get_keypoint_reference_lines(self, nearest_kp = None):
        if nearest_kp:
            if nearest_kp == 12:
                keypoint_reference_line_y = [4, 8]
            elif nearest_kp == 13:
                keypoint_reference_line_y = [10, 5]
        else:
            keypoint_reference_line_y = [12, 13]
        
        return keypoint_reference_line_y
    
    def get_mini_court_x(self, foot_position, keypoint):
        desired_y = foot_position[1]
        
        kp_line_l = [0, 2]
        kp_line_r = [1, 3]
        
        x0, y0 = keypoint[kp_line_l[0] * 2], keypoint[(kp_line_l[0] * 2) + 1]
        x2, y2 = keypoint[kp_line_l[1] * 2], keypoint[(kp_line_l[1] * 2) + 1]
        x1, y1 = keypoint[kp_line_r[0] * 2], keypoint[(kp_line_r[0] * 2) + 1]
        x3, y3 = keypoint[kp_line_r[1] * 2], keypoint[(kp_line_r[1] * 2) + 1]
        
        m_court_line_left = (y2 - y0) / (x2 - x0)
        m_court_line_right = (y3 - y1) / (x3 - x1)
        x_left = ((desired_y - y0) / m_court_line_left) + x0
        x_right = ((desired_y - y1) / m_court_line_right) + x1
        
        pixel_distance_between_keypoints = x_right - x_left
        percentage_x = (foot_position[0] - x_left) / pixel_distance_between_keypoints
        
        x0_mini, x1_mini = self.drawing_key_points[kp_line_l[0] * 2], self.drawing_key_points[kp_line_r[0] * 2]
        mini_court_x = x0_mini + ((x1_mini - x0_mini) * percentage_x)
        
        return int(mini_court_x)
    
    def get_mini_court_y(self, foot_position, keypoint, nearest_kp = None):
        if nearest_kp:
            keypoint_reference_line_y = self.get_keypoint_reference_lines(nearest_kp)
        else:
            keypoint_reference_line_y = self.get_keypoint_reference_lines()
        
        y1, y2 = keypoint[(keypoint_reference_line_y[0] * 2) + 1], keypoint[(keypoint_reference_line_y[1] * 2) + 1]
        pixel_distance_between_keypoints = y2 - y1
        percentage_y = (foot_position[1] - y1) / pixel_distance_between_keypoints
            
        y1_mini, y2_mini = self.drawing_key_points[(keypoint_reference_line_y[0] * 2) + 1], self.drawing_key_points[(keypoint_reference_line_y[1] * 2) + 1]          
        mini_court_y = y1_mini + ((y2_mini - y1_mini) * percentage_y)
        
        return int(mini_court_y)
    
    def get_mini_court_coordinates(self, ball_detections, player_detections, kps):
        self.output_mini_court_coordinates = []
        self.ball_output_mini_court_coordinates = []
        
        for ball_detection, player_detection, keypoint in zip(ball_detections, player_detections, kps):
            player_mini_court_coordinates = {}
            ball_mini_court_coordinates = {}
            
            for player_id, bbox in player_detection.items():
                player_foot_position = get_foot_position(bbox)
                nearest_kp = self.get_nearest_keypoint(player_foot_position, keypoint)
                mini_court_player_x, mini_court_player_y = self.get_mini_court_x(player_foot_position, keypoint), self.get_mini_court_y(player_foot_position, keypoint, nearest_kp)
                player_mini_court_coordinates[player_id] = mini_court_player_x, mini_court_player_y
            self.output_mini_court_coordinates.append(player_mini_court_coordinates)
            
            ball_bbox = ball_detection[1]
            ball_foot_position = get_foot_position(ball_bbox)
            mini_court_ball_x, mini_court_ball_y = self.get_mini_court_x(ball_foot_position, keypoint), self.get_mini_court_y(ball_foot_position, keypoint)
            ball_mini_court_coordinates[1] = mini_court_ball_x, mini_court_ball_y
            self.ball_output_mini_court_coordinates.append(ball_mini_court_coordinates)
    
    def draw_mini_court_players_ball(self, frame, idx):
        # self.output_mini_court_coordinates = [{1: (x1, y1, x2, y2), 2: (x1, y1, x2, y2)}, ...]
        for _, player in self.output_mini_court_coordinates[idx].items():
            cv2.circle(frame, player, 7, (255, 0, 255), -1)
        # [{1: (824.4732, 367.8773)}, {1: (824.4732, 367.8775)}]
        for _, ball in self.ball_output_mini_court_coordinates[idx].items():
            cv2.circle(frame, ball, 6, (255, 255, 0), -1)
        
        return frame
    
    def draw_on_video(self, frames):
        output_frames = []
        for idx, frame in enumerate(frames):
            drawn_frame = self.draw_background_rectangle(frame)
            drawn_frame = self.draw_court(drawn_frame)
            drawn_frame = self.draw_mini_court_players_ball(drawn_frame, idx)
            output_frames.append(drawn_frame)
            
        return output_frames

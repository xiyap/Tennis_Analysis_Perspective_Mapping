def get_bbox_midpoint(bbox):
    x1, y1, x2, y2 = bbox
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    return (x, y)

def distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    x = int((x1 + x2) / 2)
    y = int(y2)
    return (x, y)
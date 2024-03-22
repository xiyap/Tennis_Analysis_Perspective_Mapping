from ultralytics import YOLO

model = YOLO('models/player_yolov8m.pt')

result = model.track('input_video/input_video.mp4', save = True)

print(result)
print('Boxes:\n')
for box in result[0].boxes:
    print(box)
from ultralytics import YOLO

model = YOLO('models/ball_best.pt')

result = model.predict('input_video/input_video.mp4', conf = 0.2, save = True)

print(result)
print('Boxes:\n')
for box in result[0].boxes:
    print(box)
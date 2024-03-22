import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineTracker:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained = True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path, map_location = 'cuda'))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            
        kps = output.squeeze().cpu().numpy()
        ori_h, ori_w = image_rgb.shape[:2]
        
        kps[::2] *= ori_w / 224.0
        kps[1::2] *= ori_h / 224.0
        
        return kps
    
    def predict_on_video(self, frames):
        kps = []
        for frame in frames:
            kps.append(self.predict(frame))
            
        return kps
    
    def draw_kps(self, image, kps):
        for i in range(0, len(kps), 2):
            x = int(kps[i])
            y = int(kps[i + 1])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, f'{i//2}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return image
    
    def draw_kps_on_video(self, frames, kps):
        output_video_frames = []
        for frame, kps in zip(frames, kps):
            frame = self.draw_kps(frame, kps)
            output_video_frames.append(frame)
            
        return output_video_frames
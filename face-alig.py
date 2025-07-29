import cv2
import numpy as np
import os
import time
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from facenet_pytorch import MTCNN
from typing import List, Dict, Tuple, Optional

# 1. HopeNet Model Architecture (Classification Version)
class HopeNet(nn.Module):
    """HopeNet classification architecture with ResNet50 backbone"""
    def __init__(self):
        super().__init__()
        # ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # classification heads
        self.fc_yaw = nn.Linear(2048, 66)
        self.fc_pitch = nn.Linear(2048, 66)
        self.fc_roll = nn.Linear(2048, 66)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.fc_yaw(x), self.fc_pitch(x), self.fc_roll(x)

# 2. Face Alignment Pipeline
class FaceAligner:
    def __init__(self, hopenet_weights: str = None):
        # Configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize face detector
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=40,
            thresholds=[0.7, 0.8, 0.9]
        )
        
        # Initialize HopeNet 
        self.use_hopenet = False
        if hopenet_weights and os.path.exists(hopenet_weights):
            self.alignment_model = self._load_hopenet(hopenet_weights)
            self.alignment_model.eval()
            self.use_hopenet = True
            self.idx_tensor = torch.FloatTensor([i for i in range(66)]).to(self.device)
            print("HopeNet alignment model loaded")
        else:
            print("Using MTCNN landmarks for alignment")
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_hopenet(self, weights_path: str) -> nn.Module:
        """Load HopeNet model architecture"""
        model = HopeNet()
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using randomly initialized weights instead")
        return model.to(self.device)

    
    def detect_faces(self, image: Image.Image) -> Tuple:
        """Detect faces and landmarks using MTCNN"""
        try:
            boxes, probs, landmarks = self.detector.detect(image, landmarks=True)
            if boxes is None:
                return [], []
            
            # Process results
            clamped_boxes = []
            face_landmarks = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                clamped_boxes.append((
                    max(0, int(x1)),
                    max(0, int(y1)),
                    min(image.width, int(x2)),
                    min(image.height, int(y2))
                ))
                face_landmarks.append(landmarks[i])
            
            return clamped_boxes, face_landmarks
        except Exception as e:
            print(f"Detection failed: {e}")
            return [], []

    def calculate_roll_angle(self, landmarks: np.ndarray) -> float:
        """Calculate roll angle from eye landmarks"""
        # MTCNN landmarks order: left_eye, right_eye, nose, mouth_left, mouth_right
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        return np.degrees(np.arctan2(dY, dX))
    
    def refine_angle(self, face_img: Image.Image, initial_angle: float) -> float:
        """Refine angle using HopeNet if available"""
        if not self.use_hopenet:
            return initial_angle
        
        with torch.no_grad():
            try:
                img_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
                yaw_logits, pitch_logits, roll_logits = self.alignment_model(img_tensor)
                
                # Convert to probabilities
                roll_pred = nn.functional.softmax(roll_logits, dim=1)
                
                # Calculate expected value
                roll_angle = torch.sum(roll_pred * self.idx_tensor, 1) * 3 - 99
                return float(roll_angle.item())
            except Exception as e:
                print(f"Angle refinement failed: {e}")
                return initial_angle
    
    def align_face(self, face_img: Image.Image, angle: float) -> Image.Image:
        """Rotate face to upright position with proper border handling"""
        try:
            # Convert to numpy array and ensure we have 3 channels
            img_np = np.array(face_img)
            if len(img_np.shape) == 2:  # Grayscale
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:  # RGBA
                img_np = img_np[:, :, :3]
            
            h, w = img_np.shape[:2]
            center = (w // 2, h // 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            
            # Calculate new dimensions
            abs_cos = abs(M[0, 0])
            abs_sin = abs(M[0, 1])
            new_w = int(h * abs_sin + w * abs_cos)
            new_h = int(h * abs_cos + w * abs_sin)
            
            # Adjust rotation matrix
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            
            # Apply rotation with border replication
            rotated = cv2.warpAffine(
                img_np, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE  # Use edge replication instead of black
            )
            
            # Convert back to PIL image
            rotated_img = Image.fromarray(rotated)
            
            # Crop to face content
            return self.crop_to_face_content(rotated_img)
        except Exception as e:
            print(f"Alignment failed: {e}")
            return face_img

    def crop_to_face_content(self, image: Image.Image) -> Image.Image:
        """Crop image to remove black/empty borders around face"""
        try:
            # Convert to numpy array
            img_np = np.array(image)
            
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return image
                
            # Find largest contour
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Add 5% margin
            margin = int(min(w, h) * 0.05)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_np.shape[1] - x, w + 2 * margin)
            h = min(img_np.shape[0] - y, h + 2 * margin)
            
            # Crop and return
            cropped = img_np[y:y+h, x:x+w]
            return Image.fromarray(cropped)
        except Exception as e:
            print(f"Cropping failed: {e}")
            return image

    def process_cropped_face(self, img_path: str, output_dir: str) -> Dict:
        """Process a single cropped face image"""
        try:
            # Load image
            img = Image.open(img_path)
            
            # Use MTCNN to detect landmarks in cropped face
            _, _, landmarks = self.detector.detect(img, landmarks=True)
            if landmarks is None or len(landmarks) == 0:
                print(f"No landmarks detected in {os.path.basename(img_path)}")
                return None
            
            # Calculate initial roll angle
            roll_angle = self.calculate_roll_angle(landmarks[0])
            
            # Refine angle with HopeNet
            refined_angle = self.refine_angle(img, roll_angle)
            
            # Align face
            aligned_face = self.align_face(img, refined_angle)
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_aligned.jpg")
            aligned_face.save(output_path)
            
            return {
                'image': os.path.basename(img_path),
                'initial_angle': roll_angle,
                'refined_angle': refined_angle,
                'aligned_path': output_path
            }
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def process_full_image(self, img_path: str, output_dir: str) -> List[Dict]:
        """Process a full image with multiple faces"""
        try:
            # Load image
            img = Image.open(img_path)
            img_name = os.path.basename(img_path)
            
            # Create output directories
            os.makedirs(os.path.join(output_dir, 'detections'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'aligned_faces'), exist_ok=True)
            
            # Detect faces and landmarks
            boxes, all_landmarks = self.detect_faces(img)
            if not boxes:
                print(f"No faces found in {img_name}")
                return []
            
            results = []
            
            # Create detection visualization (without landmarks)
            det_img = img.copy()
            draw = ImageDraw.Draw(det_img)
            
            for i, box in enumerate(boxes):
                try:
                    # Draw bounding box only (no landmarks)
                    draw.rectangle(box, outline='red', width=3)
                    
                    # Crop face
                    face_crop = img.crop(box)
                    
                    # Calculate initial roll angle
                    roll_angle = self.calculate_roll_angle(all_landmarks[i])
                    
                    # Refine angle with HopeNet
                    refined_angle = self.refine_angle(face_crop, roll_angle)
                    
                    # Align face
                    aligned_face = self.align_face(face_crop, refined_angle)
                    
                    # Save results
                    base_name = os.path.splitext(img_name)[0]
                    aligned_path = os.path.join(output_dir, 'aligned_faces', f"{base_name}_face{i}_aligned.jpg")
                    aligned_face.save(aligned_path)
                    
                    results.append({
                        'image': img_name,
                        'face_id': i,
                        'box': tuple(box),
                        'initial_angle': roll_angle,
                        'refined_angle': refined_angle,
                        'aligned_path': aligned_path
                    })
                except Exception as e:
                    print(f"Error processing face {i}: {e}")
            
            # Save detection image (without landmarks)
            det_path = os.path.join(output_dir, 'detections', img_name)
            det_img.save(det_path)
            
            return results
        except Exception as e:
            print(f"Error processing image: {e}")
            return []
    
    def process_batch(self, input_dir: str, output_dir: str, is_cropped: bool = False) -> pd.DataFrame:
        """Process all images in a directory"""
        results = []
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(input_dir, img_file)
            print(f"Processing ({i+1}/{len(image_files)}): {img_file}")
            
            if is_cropped:
                result = self.process_cropped_face(img_path, output_dir)
                if result:
                    results.append(result)
            else:
                face_results = self.process_full_image(img_path, output_dir)
                results.extend(face_results)
        
        # Save results to CSV
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(output_dir, 'alignment_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
        else:
            print("No results to save")
        
        return results

# Command Line Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Face Detection and Alignment Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True, 
                      help='Input file or directory')
    parser.add_argument('--output', type=str, default='output', 
                      help='Output directory')
    parser.add_argument('--weights', type=str, default=None,
                      help='Path to HopeNet weights (optional)')
    parser.add_argument('--cropped', action='store_true',
                      help='Input images are pre-cropped faces')
    args = parser.parse_args()

    # Initialize pipeline
    aligner = FaceAligner(args.weights)
    
    # Process based on input type
    if os.path.isfile(args.input):
        if args.cropped:
            result = aligner.process_cropped_face(args.input, args.output)
            if result:
                print(f"Aligned face saved to {result['aligned_path']}")
        else:
            results = aligner.process_full_image(args.input, args.output)
            print(f"Processed {len(results)} faces")
    elif os.path.isdir(args.input):
        results = aligner.process_batch(args.input, args.output, args.cropped)
        print(f"Processed {len(results)} faces/crops")
    else:
        raise ValueError("Input path must be a file or directory")
    
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    
    main()
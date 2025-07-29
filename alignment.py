import cv2
import numpy as np
import os
import csv
import time
import glob
from facenet_pytorch import MTCNN
import torch
from PIL import Image
from typing import Tuple, List, Optional

class FaceAligner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=60,
            thresholds=[0.7, 0.8, 0.9]
        )

    def rotate_bound(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image without cropping, preserving entire content"""
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        return cv2.warpAffine(
            image, M, (nW, nH),
            borderMode=cv2.BORDER_REPLICATE
        )

    def align_face(self, img: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], float, str]:
        """Align face using MTCNN for more accurate detection"""
        try:
            # Convert to PIL Image
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Detect faces and landmarks
            boxes, probs, landmarks = self.detector.detect(img_pil, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                return None, 0.0, "No face detected"
            
            # Use the face with highest detection probability
            main_idx = np.argmax(probs)
            landmarks = landmarks[main_idx]
            
            # MTCNN landmarks order: left_eye, right_eye, nose, mouth_left, mouth_right
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Calculate rotation angle
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            return [tuple(map(int, left_eye)), tuple(map(int, right_eye))], angle, "Aligned"
        
        except Exception as e:
            return None, 0.0, f"Error: {str(e)}"

def process_images(input_dir: str, output_dir: str, csv_path: str):
    """Process all images in directory with improved alignment"""
    os.makedirs(output_dir, exist_ok=True)
    aligned_dir = os.path.join(output_dir, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)
    
    aligner = FaceAligner()
    
    image_paths = glob.glob(os.path.join(input_dir, "*.*"))
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = [p for p in image_paths if p.lower().endswith(valid_exts)]
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'face_angle', 'inference_time', 'alignment_status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for img_path in image_paths:
            start_time = time.time()
            img = cv2.imread(img_path)
            img_name = os.path.basename(img_path)
            output_path = os.path.join(aligned_dir, img_name)
            angle = 0.0
            status = "Skipped"
            inference_time = 0.0
            
            if img is None:
                status = 'Invalid image'
            else:
                try:
                    # Process image with MTCNN
                    eye_centers, angle, status = aligner.align_face(img)
                    inference_time = (time.time() - start_time) * 1000  # in ms
                    
                    # Rotate and save if alignment succeeded
                    if status == "Aligned":
                        aligned_img = aligner.rotate_bound(img, angle)
                        
                        # Optional: Crop to face area
                        # You can add face cropping here if desired
                        
                        cv2.imwrite(output_path, aligned_img)
                    else:
                        # Save original if alignment failed
                        cv2.imwrite(output_path, img)
                except Exception as e:
                    status = f"Error: {str(e)}"
                    cv2.imwrite(output_path, img)
            
            # Write results to CSV
            writer.writerow({
                'image_name': img_name,
                'face_angle': round(angle, 2),
                'inference_time': round(inference_time, 2),
                'alignment_status': status
            })
            print(f"Processed {img_name}: {status}, Angle: {angle:.2f}°, Time: {inference_time:.2f}ms")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Face Alignment with MTCNN')
    parser.add_argument('--input', type=str, default='input', help='Input directory with images')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    parser.add_argument('--csv', type=str, default='results.csv', help='CSV results filename')
    
    args = parser.parse_args()
    
    process_images(
        input_dir=args.input,
        output_dir=args.output,
        csv_path=args.csv
    )
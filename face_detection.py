# face_detection.py
import cv2
import numpy as np
import tempfile
import os
import time

class FaceDetector:
    def __init__(self):
        """Initialize face detector with Haar cascade classifiers"""
        try:
            # Load multiple cascade classifiers for better accuracy
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            alt_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            self.face_cascade_alt = cv2.CascadeClassifier(alt_cascade_path)
            
            # Check if cascades loaded properly
            self.is_available = (
                not self.face_cascade.empty() and 
                not self.face_cascade_alt.empty()
            )
            
            if self.is_available:
                print("Face detector initialized successfully")
            else:
                print("Warning: Could not load face cascade classifiers")
                # Try to load from alternative path if needed
                if self.face_cascade.empty():
                    print("Default cascade failed to load")
                
        except Exception as e:
            print(f"Error initializing face detector: {e}")
            self.is_available = False
    
    def detect_faces(self, image):
        """Main face detection method with multiple attempts"""
        if not self.is_available:
            return np.array([])
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better contrast
            gray = cv2.equalizeHist(gray)
            
            all_faces = []
            
            # Try different parameters to catch more faces
            param_sets = [
                # (scaleFactor, minNeighbors, minSize)
                (1.05, 5, (60, 60)),   # Standard - good balance
                (1.1, 5, (60, 60)),     # Slightly faster
                (1.05, 3, (50, 50)),    # More sensitive - smaller faces
                (1.1, 3, (50, 50)),     # More sensitive, faster
                (1.05, 5, (80, 80)),    # Larger faces only
            ]
            
            for scale, neighbors, min_size in param_sets:
                # Try with default cascade
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale,
                    minNeighbors=neighbors,
                    minSize=min_size,
                    maxSize=(500, 500)
                )
                
                if len(faces) > 0:
                    all_faces.extend(faces.tolist())
                
                # Try with alternative cascade
                alt_faces = self.face_cascade_alt.detectMultiScale(
                    gray,
                    scaleFactor=scale,
                    minNeighbors=neighbors,
                    minSize=min_size,
                    maxSize=(500, 500)
                )
                
                if len(alt_faces) > 0:
                    all_faces.extend(alt_faces.tolist())
            
            # Remove duplicate/overlapping faces
            if len(all_faces) > 0:
                all_faces = self.remove_overlapping_faces(np.array(all_faces))
            
            return all_faces
            
        except Exception as e:
            print(f"Error in detect_faces: {e}")
            return np.array([])
    
    def remove_overlapping_faces(self, faces, iou_threshold=0.4):
        """Remove overlapping face detections"""
        if len(faces) == 0:
            return faces
        
        # Convert to list and sort by area (largest first)
        faces_list = faces.tolist()
        faces_list.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        keep = []
        
        while len(faces_list) > 0:
            current = faces_list[0]
            keep.append(current)
            faces_list = faces_list[1:]
            
            # Remove overlapping faces
            filtered = []
            for face in faces_list:
                # Calculate overlap
                x1, y1, w1, h1 = current
                x2, y2, w2, h2 = face
                
                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    iou = intersection / (area1 + area2 - intersection)
                    
                    if iou < iou_threshold:
                        filtered.append(face)
                else:
                    filtered.append(face)
            
            faces_list = filtered
        
        return np.array(keep)
    
    def draw_face_boxes(self, image, faces):
        """Draw rectangles around detected faces"""
        img_copy = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Add face number label
            label = f'Face {i+1}'
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(img_copy, (x, y - text_height - 8), 
                         (x + text_width + 8, y), (0, 255, 0), -1)
            
            # Put text
            cv2.putText(img_copy, label, (x + 4, y - 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add count overlay
        if len(faces) > 0:
            count_text = f'Total Faces: {len(faces)}'
            cv2.putText(img_copy, count_text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img_copy
    
    def process_image_file(self, uploaded_file):
        """Process an uploaded image file"""
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not read image file")
            
            # Resize if too large for better performance
            height, width = image.shape[:2]
            max_dimension = 1200
            if height > max_dimension or width > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Detect faces
            faces = self.detect_faces(image)
            
            # Draw boxes
            result_image = self.draw_face_boxes(image, faces)
            
            # Convert to RGB for display
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            return result_image_rgb, len(faces)
            
        except Exception as e:
            print(f"Error in process_image_file: {e}")
            raise e
    
    def process_video_file(self, uploaded_file):
        """Process an uploaded video file"""
        video_path = None
        try:
            uploaded_file.seek(0)
            
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Process frames
            face_counts = []
            processed_frames = []
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 15th frame to save time
                if frame_count % 15 == 0:
                    faces = self.detect_faces(frame)
                    face_counts.append(len(faces))
                    
                    # Store first few frames for display
                    if len(processed_frames) < 8:
                        processed_frame = self.draw_face_boxes(frame, faces)
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        processed_frames.append(processed_frame_rgb)
                
                frame_count += 1
            
            cap.release()
            
            # Calculate statistics
            avg_faces = np.mean(face_counts) if face_counts else 0
            max_faces = max(face_counts) if face_counts else 0
            
            return processed_frames, {
                'total_frames': frame_count,
                'avg_faces': float(avg_faces),
                'max_faces': int(max_faces),
                'face_counts': face_counts
            }
            
        except Exception as e:
            print(f"Error in process_video_file: {e}")
            raise e
        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except:
                    pass
    
    def live_webcam_detection(self, duration=5):
        """Real-time face detection from webcam"""
        cap = None
        try:
            # Try to open webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                return None, {"error": "Could not open webcam. Please check if webcam is connected."}
            
            frames = []
            face_counts = []
            
            # Set resolution to smaller for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            start_time = time.time()
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                face_counts.append(len(faces))
                
                # Draw boxes
                processed_frame = self.draw_face_boxes(frame, faces)
                
                # Convert to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display (make it smaller)
                processed_frame_rgb = cv2.resize(processed_frame_rgb, (480, 360))
                frames.append(processed_frame_rgb)
                
                # Small delay
                time.sleep(0.033)  # ~30 fps
            
            cap.release()
            
            # Calculate statistics
            avg_faces = np.mean(face_counts) if face_counts else 0
            max_faces = max(face_counts) if face_counts else 0
            
            return frames, {
                'total_frames': len(face_counts),
                'avg_faces': float(avg_faces),
                'max_faces': int(max_faces),
                'face_counts': face_counts,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in live_webcam_detection: {e}")
            return None, {"error": f"Webcam error: {str(e)}"}
        finally:
            if cap is not None:
                cap.release()
    
    def live_webcam_detection_with_stream(self, duration=5):
        """Real-time face detection with streaming - yields frames one by one"""
        cap = None
        try:
            # Try to open webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                yield None, {"error": "Could not open webcam. Please check if webcam is connected."}
                return
            
            # Set resolution to smaller for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            start_time = time.time()
            face_counts = []
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                face_counts.append(len(faces))
                
                # Draw boxes
                processed_frame = self.draw_face_boxes(frame, faces)
                
                # Convert to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display (make it smaller)
                processed_frame_rgb = cv2.resize(processed_frame_rgb, (480, 360))
                
                # Yield the frame and current face count
                yield processed_frame_rgb, {
                    'current_faces': len(faces),
                    'is_capturing': True,
                    'elapsed': time.time() - start_time
                }
                
                # Small delay to control frame rate
                time.sleep(0.033)  # ~30 fps
            
            cap.release()
            
            # Calculate final statistics
            avg_faces = np.mean(face_counts) if face_counts else 0
            max_faces = max(face_counts) if face_counts else 0
            
            # Yield final result
            yield None, {
                'total_frames': len(face_counts),
                'avg_faces': float(avg_faces),
                'max_faces': int(max_faces),
                'face_counts': face_counts,
                'is_capturing': False,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in live_webcam_detection_with_stream: {e}")
            yield None, {"error": f"Webcam error: {str(e)}"}
        finally:
            if cap is not None:
                cap.release()


# SimpleFaceDetector class that inherits all methods
class SimpleFaceDetector(FaceDetector):
    """Simplified face detector that inherits all methods"""
    
    def __init__(self):
        super().__init__()
        print("SimpleFaceDetector initialized")


# Test the face detector when run directly
if __name__ == "__main__":
    print("Testing FaceDetector...")
    detector = FaceDetector()
    print(f"Face detector available: {detector.is_available}")
    
    # Test with sample image if available
    test_image = "test.jpg"
    if os.path.exists(test_image):
        with open(test_image, 'rb') as f:
            from io import BytesIO
            class MockUpload:
                def __init__(self, data):
                    self.data = data
                    self.pos = 0
                def read(self):
                    return self.data
                def seek(self, pos):
                    self.pos = pos
            
            mock_file = MockUpload(f.read())
            result, count = detector.process_image_file(mock_file)
            print(f"Detected {count} faces in test image")
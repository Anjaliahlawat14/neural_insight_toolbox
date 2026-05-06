# face_detection.py
import cv2
import numpy as np
import tempfile
import os
import time

class FaceDetector:
    def __init__(self):
        """Initialize face detector with multiple Haar cascade classifiers"""
        self.face_cascades = []
        self.cascade_names = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt2.xml',
        ]
        
        # Load all available cascades
        for cascade_name in self.cascade_names:
            try:
                cascade_path = cv2.data.haarcascades + cascade_name
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self.face_cascades.append(cascade)
            except Exception:
                continue
        
        # Load eye cascade for validation
        try:
            eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_path)
        except Exception:
            self.eye_cascade = None
        
        # Load smile cascade for expression detection
        try:
            smile_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
            self.smile_cascade = cv2.CascadeClassifier(smile_path)
        except Exception:
            self.smile_cascade = None
            
        self.is_available = len(self.face_cascades) > 0
        
        if self.is_available:
            print(f"Face detector initialized with {len(self.face_cascades)} cascades")
        else:
            print("Warning: Could not load face cascade classifiers")
    
    def detect_faces(self, image, min_face_size=(80, 80), max_face_size=(500, 500)):
        """
        Main face detection method with improved filtering
        
        Parameters:
        -----------
        image : numpy array
            Input image (BGR format)
        min_face_size : tuple
            Minimum face size (width, height) - increased to reduce false positives
        max_face_size : tuple
            Maximum face size (width, height)
            
        Returns:
        --------
        numpy array: Array of face rectangles [[x, y, w, h], ...]
        """
        if not self.is_available:
            return np.array([])
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better contrast
            gray = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            all_detections = []
            
            # Use more conservative parameters to reduce false positives
            # Only use the most reliable cascade with stricter parameters
            primary_cascade = self.face_cascades[0] if self.face_cascades else None
            
            if primary_cascade:
                # Try with stricter parameters first
                param_sets = [
                    (1.05, 8, min_face_size, max_face_size),   # Very strict - most reliable
                    (1.1, 6, min_face_size, max_face_size),     # Slightly less strict
                ]
                
                for scale, neighbors, min_size, max_size in param_sets:
                    try:
                        faces = primary_cascade.detectMultiScale(
                            gray,
                            scaleFactor=scale,
                            minNeighbors=neighbors,
                            minSize=min_size,
                            maxSize=max_size
                        )
                        if len(faces) > 0:
                            all_detections.extend(faces.tolist())
                    except Exception:
                        continue
            
            # Remove duplicate/overlapping faces
            if len(all_detections) > 0:
                all_detections = self.remove_overlapping_faces(np.array(all_detections), iou_threshold=0.3)
                
                # Validate with eye detection (most reliable filter)
                if self.eye_cascade is not None and len(all_detections) > 0:
                    all_detections = self.validate_with_eyes(gray, all_detections)
                
                # Additional filter: only keep the largest face if multiple remain
                if len(all_detections) > 1:
                    all_detections = self.keep_largest_face(all_detections)
            
            return np.array(all_detections) if len(all_detections) > 0 else np.array([])
            
        except Exception as e:
            print(f"Error in detect_faces: {e}")
            return np.array([])
    
    def validate_with_eyes(self, gray_image, faces):
        """
        Validate face detections by checking for eyes.
        Returns only faces that have at least 2 eyes detected.
        """
        validated_faces = []
        
        for (x, y, w, h) in faces:
            # Ensure coordinates are within bounds
            if x < 0 or y < 0 or x + w > gray_image.shape[1] or y + h > gray_image.shape[0]:
                continue
                
            roi_gray = gray_image[y:y+h, x:x+w]
            if roi_gray.size == 0:
                continue
            
            # Detect eyes in the face region with stricter parameters
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=8,  # Higher minNeighbors for stricter detection
                minSize=(15, 15),
                maxSize=(50, 50)
            )
            
            # Accept only if at least 2 eyes are detected (or at least 1 for profile shots)
            if len(eyes) >= 2 or (len(eyes) >= 1 and w < 150):  # Allow 1 eye for small/profile faces
                validated_faces.append([x, y, w, h])
        
        return np.array(validated_faces) if validated_faces else faces
    
    def keep_largest_face(self, faces):
        """
        Keep only the largest face when multiple faces are detected.
        This assumes there's only one person in the frame.
        """
        if len(faces) == 0:
            return faces
        
        # Calculate area for each face
        areas = [(face[2] * face[3]) for face in faces]
        
        # Find the index of the largest face
        largest_idx = np.argmax(areas)
        
        # Return only the largest face
        return np.array([faces[largest_idx]])
    
    def remove_overlapping_faces(self, faces, iou_threshold=0.3):
        """
        Remove overlapping face detections using IoU (Intersection over Union)
        Lowered threshold to be more aggressive in removing overlaps
        """
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
                    
                    # Keep only if IoU is below threshold
                    if iou < iou_threshold:
                        filtered.append(face)
                else:
                    filtered.append(face)
            
            faces_list = filtered
        
        return np.array(keep)
    
    def detect_smile(self, face_roi):
        """Detect smile in a face region"""
        if self.smile_cascade is None:
            return False, 0
        
        try:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            smiles = self.smile_cascade.detectMultiScale(
                gray_face, 
                scaleFactor=1.8, 
                minNeighbors=20, 
                minSize=(25, 25)
            )
            return len(smiles) > 0, len(smiles)
        except Exception:
            return False, 0
    
    def detect_eyes(self, face_roi):
        """Detect eyes in a face region"""
        if self.eye_cascade is None:
            return []
        
        try:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            eyes = self.eye_cascade.detectMultiScale(
                gray_face, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(15, 15)
            )
            return eyes.tolist() if len(eyes) > 0 else []
        except Exception:
            return []
    
    def draw_face_boxes(self, image, faces, show_details=True):
        """
        Draw rectangles around detected faces with additional details
        """
        img_copy = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw main rectangle around face
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            if show_details and w > 0 and h > 0:
                # Extract face region for detailed analysis
                face_roi = image[y:min(y+h, image.shape[0]), x:min(x+w, image.shape[1])]
                
                if face_roi.size > 0:
                    # Detect and draw eyes
                    eyes = self.detect_eyes(face_roi)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(img_copy, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
                    
                    # Detect smile
                    has_smile, smile_count = self.detect_smile(face_roi)
                    if has_smile:
                        cv2.putText(img_copy, "😊", (x + w - 30, y + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Add face number label
                    label = f'Face {i+1}'
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
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
            
            # Draw boxes with details
            result_image = self.draw_face_boxes(image, faces, show_details=True)
            
            # Convert to RGB for display
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Calculate additional stats
            face_analysis = []
            for i, (x, y, w, h) in enumerate(faces):
                if w > 0 and h > 0:
                    face_roi = image[y:min(y+h, image.shape[0]), x:min(x+w, image.shape[1])]
                    if face_roi.size > 0:
                        has_smile, _ = self.detect_smile(face_roi)
                        eye_count = len(self.detect_eyes(face_roi))
                        face_analysis.append({
                            'face_id': i + 1,
                            'position': (x, y, w, h),
                            'has_smile': has_smile,
                            'eye_count': eye_count
                        })
            
            return result_image_rgb, len(faces), face_analysis
            
        except Exception as e:
            print(f"Error in process_image_file: {e}")
            raise e
    
    def process_video_file(self, uploaded_file, sample_interval=15):
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
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process frames
            face_counts = []
            processed_frames = []
            frame_analyses = []
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process at specified interval
                if frame_count % sample_interval == 0:
                    faces = self.detect_faces(frame)
                    face_counts.append(len(faces))
                    
                    # Store frame analysis
                    frame_analyses.append({
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'face_count': len(faces)
                    })
                    
                    # Store first few frames for display (up to 8)
                    if len(processed_frames) < 8:
                        processed_frame = self.draw_face_boxes(frame, faces, show_details=False)
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        processed_frames.append(processed_frame_rgb)
                
                frame_count += 1
            
            cap.release()
            
            # Calculate statistics
            avg_faces = np.mean(face_counts) if face_counts else 0
            max_faces = max(face_counts) if face_counts else 0
            min_faces = min(face_counts) if face_counts else 0
            
            return processed_frames, {
                'total_frames': frame_count,
                'processed_frames': len(face_counts),
                'avg_faces': float(avg_faces),
                'max_faces': int(max_faces),
                'min_faces': int(min_faces),
                'face_counts': face_counts,
                'frame_analyses': frame_analyses,
                'fps': fps
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
    
    def live_webcam_detection_with_stream(self, duration=5):
        """Real-time face detection with streaming"""
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                yield None, {"error": "Could not open webcam. Please check if webcam is connected."}
                return
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            start_time = time.time()
            face_counts = []
            face_analyses = []
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces with stricter parameters
                faces = self.detect_faces(frame, min_face_size=(80, 80))
                face_counts.append(len(faces))
                
                # Analyze each face
                current_analysis = []
                for i, (x, y, w, h) in enumerate(faces):
                    if w > 0 and h > 0:
                        face_roi = frame[y:min(y+h, frame.shape[0]), x:min(x+w, frame.shape[1])]
                        if face_roi.size > 0:
                            has_smile, _ = self.detect_smile(face_roi)
                            eye_count = len(self.detect_eyes(face_roi))
                            current_analysis.append({
                                'face_id': i + 1,
                                'has_smile': has_smile,
                                'eye_count': eye_count
                            })
                
                face_analyses.append({
                    'timestamp': time.time() - start_time,
                    'face_count': len(faces),
                    'faces': current_analysis
                })
                
                # Draw boxes
                processed_frame = self.draw_face_boxes(frame, faces, show_details=True)
                
                # Convert to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                processed_frame_rgb = cv2.resize(processed_frame_rgb, (480, 360))
                
                yield processed_frame_rgb, {
                    'current_faces': len(faces),
                    'is_capturing': True,
                    'elapsed': time.time() - start_time,
                    'face_analysis': current_analysis
                }
                
                time.sleep(0.033)
            
            cap.release()
            
            avg_faces = np.mean(face_counts) if face_counts else 0
            max_faces = max(face_counts) if face_counts else 0
            
            total_smiles = sum(
                1 for analysis in face_analyses 
                for face in analysis.get('faces', []) 
                if face.get('has_smile', False)
            )
            
            yield None, {
                'total_frames': len(face_counts),
                'avg_faces': float(avg_faces),
                'max_faces': int(max_faces),
                'face_counts': face_counts,
                'face_analyses': face_analyses,
                'total_smiles_detected': total_smiles,
                'is_capturing': False,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in live_webcam_detection_with_stream: {e}")
            yield None, {"error": f"Webcam error: {str(e)}"}
        finally:
            if cap is not None:
                cap.release()
    
    def live_webcam_detection(self, duration=5):
        """Real-time face detection from webcam (batch mode)"""
        frames = []
        final_stats = None
        
        generator = self.live_webcam_detection_with_stream(duration)
        
        for frame, stats in generator:
            if frame is not None:
                frames.append(frame)
            else:
                final_stats = stats
        
        return frames, final_stats


class SimpleFaceDetector(FaceDetector):
    """Simplified face detector that inherits all methods"""
    
    def __init__(self):
        super().__init__()
        print("SimpleFaceDetector initialized")


# Test the face detector
if __name__ == "__main__":
    print("Testing Advanced FaceDetector...")
    detector = FaceDetector()
    print(f"Face detector available: {detector.is_available}")
    print(f"Number of cascades loaded: {len(detector.face_cascades)}")
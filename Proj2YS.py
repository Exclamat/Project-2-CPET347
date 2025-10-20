import cv2
import numpy as np
import os
import time
import joblib
from sklearn.metrics import mean_squared_error

PCA_MODEL_FILE = 'pca_model.joblib'
MODEL_DIMS_FILE = 'model_dims.npy'
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
LBP_CASCADE_FILE = 'lbpcascade_frontalface.xml' # HOG alternative

MSE_THRESHOLD = 4000      # Face-likeness

RE_DETECTION_THRESHOLD = 0.3 # Redetect if only 30% or less points of seen
BBOX_PADDING = 8

TRACK_COLOR = (0, 0, 255)         # Red
POINT_COLOR = (0, 255, 0)         # Green
FACE_COLOR = (0, 255, 0)          # Green
NOT_FACE_COLOR = (0, 0, 255)      # Red
TEXT_COLOR = (255, 255, 255)    # White
OUTLINE_COLOR = (0, 0, 0)       # Black

HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 7
HAAR_MIN_SIZE = (25, 25)

feature_params = dict( 
    maxCorners = 100,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7
)
lk_params = dict( 
    winSize = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def draw_text_with_outline(img, text, org, font, scale, color, thickness=2):
    # Draw black outline, then main text
    cv2.putText(img, text, org, font, scale, OUTLINE_COLOR, thickness * 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def main():
    # Check for calibration
    print("Loading all models...")
    required_files = [PCA_MODEL_FILE, MODEL_DIMS_FILE, 
                      HAAR_CASCADE_FILE, LBP_CASCADE_FILE]
    
    if not all(os.path.exists(f) for f in required_files):
        print("--- ERROR ---")
        print("One or more required model/cascade files are missing.")
        print(f"Missing: {[f for f in required_files if not os.path.exists(f)]}")
        return

    try:
        pca = joblib.load(PCA_MODEL_FILE)
        model_h, model_w = np.load(MODEL_DIMS_FILE)
        haar_detector = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
        lbp_detector = cv2.CascadeClassifier(LBP_CASCADE_FILE)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
        
    print("All models loaded successfully.")

    print("Starting webcam feed...")
    print("  - Press 'b' to toggle Background/Foreground view.")
    print("  - Press 'd' to cycle Detector Mode (Haar, HOG/LBP).")
    print("  - Press 't' to toggle All Annotations (boxes/points).")
    print("  - Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Tracking
    tracking = False
    bbox = None
    p0 = None
    old_gray = None
    initial_point_count = 0 
    
    # Background Subtraction
    gmm_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    # Toggles
    show_background = False # 'b'
    show_all_annotations = True # 't'
    detector_mode = 0 # 0 = Haar+Recog, 1 = HOG
    mode_labels = ["Haar + Recog", "HOG (LBP) Only"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_frame = frame.copy() # Frame that is drawn on

        # GMM Background Model
        fg_mask = gmm_subtractor.apply(frame)
        _, binary_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        if tracking:
            mode_text = "MODE: Tracking"
            p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            p1_good = p1[status == 1] if p1 is not None else np.array([])
            p0_good = p0[status == 1] if p1 is not None else np.array([])

            # Outlier Rejection
            p_really_good = [] 
            if len(p1_good) > 5:
                move_vectors = p1_good - p0_good
                median_dx = np.median(move_vectors[:, 0])
                median_dy = np.median(move_vectors[:, 1])
                median_vector = np.array([median_dx, median_dy])
                distances = np.linalg.norm(move_vectors.reshape(-1, 2) - median_vector, axis=1)
                distance_threshold = np.percentile(distances, 75) + 10.0 
                for i, dist in enumerate(distances):
                    if dist < distance_threshold:
                        p_really_good.append(p1_good[i])
                p_really_good = np.array(p_really_good)

            # Re-detection Logic
            if len(p_really_good) < (initial_point_count * RE_DETECTION_THRESHOLD) or len(p_really_good) < 5:
                tracking = False # Lost track, redetect
                p0 = None
                bbox = None
            else:
                # Update Bounding Box
                x_new, y_new, w_new, h_new = cv2.boundingRect(p_really_good)
                h_frame, w_frame = output_frame.shape[:2]
                x_new = max(0, x_new - BBOX_PADDING)
                y_new = max(0, y_new - BBOX_PADDING)
                w_new = min(w_frame - x_new, w_new + 2 * BBOX_PADDING)
                h_new = min(h_frame - y_new, h_new + 2 * BBOX_PADDING)
                bbox = (x_new, y_new, w_new, h_new)
                
                # Toggleable Drawing
                if show_all_annotations:
                    cv2.rectangle(output_frame, (x_new, y_new), (x_new + w_new, y_new + h_new), TRACK_COLOR, 2)
                    draw_text_with_outline(output_frame, "Face", (x_new, y_new - 10), 
                                           font, 0.6, FACE_COLOR)
                    for point in p_really_good:
                        cv2.circle(output_frame, (int(point[0]), int(point[1])), 3, POINT_COLOR, -1)
                
                p0 = p_really_good.reshape(-1, 1, 2)
        
        if not tracking:
            mode_text = f"MODE: {mode_labels[detector_mode]}"

            # Haar
            if detector_mode == 0:
                faces_haar = haar_detector.detectMultiScale(
                    frame_gray, HAAR_SCALE_FACTOR, HAAR_MIN_NEIGHBORS, minSize=HAAR_MIN_SIZE
                )
                for (x, y, w, h) in faces_haar:
                    face_crop = frame_gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_crop, (model_w, model_h)) 
                    face_flat = face_resized.flatten().reshape(1, -1)
            
                    transformed_face = pca.transform(face_flat)
                    reconstructed_face = pca.inverse_transform(transformed_face)
                    mse = mean_squared_error(face_flat, reconstructed_face)
                    
                    label = "Not-Face"
                    color = NOT_FACE_COLOR
                    can_track = False

                    if mse <= MSE_THRESHOLD:
                        label = "Face"
                        color = FACE_COLOR
                        can_track = True
                    
                    if show_all_annotations:
                        cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
                        draw_text_with_outline(output_frame, label, (x, y - 10), 
                                               font, 0.6, color)

                    # Switch to tracker
                    if can_track and not tracking:
                        tracking = True
                        bbox = (x, y, w, h)
                        mask = np.zeros_like(frame_gray)
                        mask[y:y+h, x:x+w] = 255
                        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                        if p0 is not None:
                            initial_point_count = len(p0)
                        else:
                            tracking = False # No features

            # HOG (LBP) Mode
            elif detector_mode == 1:
                faces_lbp = lbp_detector.detectMultiScale(frame_gray, 1.1, 5, minSize=(30, 30))
                if show_all_annotations:
                    for (x, y, w, h) in faces_lbp:
                        cv2.rectangle(output_frame, (x, y), (x+w, y+h), FACE_COLOR, 2)
            
        # Update for next loop
        old_gray = frame_gray.copy()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('b'):
            show_background = not show_background # Flip toggle
        if key == ord('d'):
            detector_mode = (detector_mode + 1) % 2 # 0, 1
            tracking = False # Changing resets tracker
            p0 = None
            bbox = None
        if key == ord('t'):
            show_all_annotations = not show_all_annotations
            
        # Always draw mode text
        if show_background:
            if tracking: mode_text = "MODE: Tracking"
            draw_text_with_outline(mask_bgr, mode_text, (10, 30), 
                                   font, 0.7, TEXT_COLOR)
            cv2.imshow('Live System - Press "q" to quit', mask_bgr)
        else:
            if tracking: mode_text = "MODE: Tracking"
            draw_text_with_outline(output_frame, mode_text, (10, 30), 
                                   font, 0.7, TEXT_COLOR)
            cv2.imshow('Live System - Press "q" to quit', output_frame)

    print("--- Webcam feed stopped ---")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

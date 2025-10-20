import cv2
import numpy as np
import os
import time
import joblib
import sys
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Dependencies
PCA_MODEL_FILE = 'pca_model.joblib' # Can be made in software
MODEL_DIMS_FILE = 'model_dims.npy' # Can be made in software
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
LBP_CASCADE_FILE = 'lbpcascade_frontalface.xml' # HOG alternative
INPUT_VIDEO_PATH = 'Tam Nakano vs Saya Kamitami - Faces.mp4' # Used for training

# Files made by training
MODEL_FILES = [PCA_MODEL_FILE, MODEL_DIMS_FILE]

MODEL_H = 64  # All faces will be resized to 64x64
MODEL_W = 64
N_COMPONENTS = 100 # Number of eigenfaces
RANDOM_STATE = 42

MSE_THRESHOLD = 4000      # Face-likeness
RE_DETECTION_THRESHOLD = 0.3 # Redetect if tracker loses more than 70% of points
BBOX_PADDING = 8

TRACK_COLOR = (0, 0, 255)         # Red Box
POINT_COLOR = (0, 255, 0)         # Green Points)
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
    # Create outline, draw text
    cv2.putText(img, text, org, font, scale, OUTLINE_COLOR, thickness * 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def train_pca_model_from_video():
    print("\n--- Starting Model Training from Video ---")

    print(f"Loading video: {INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return False

    haar_detector = cv2.CascadeClassifier(HAAR_CASCADE_FILE)

    face_data = []
    frame_count = 0
    print("Scanning video to extract face samples...")
    print("This may take a few minutes...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        print(f"Processing frame {frame_count}...", end='\r')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_detector.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE
        )
        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (MODEL_W, MODEL_H))
            face_data.append(face_resized.flatten())

    cap.release()
    print(f"\nVideo scan complete. Found {len(face_data)} face samples.")

    if len(face_data) < 20: # Min sample #
        print("Error: Not enough faces were found in the video to train.")
        return False

    # Run Eigenfaces Algorithm
    X = np.array(face_data)
    n_components_actual = min(N_COMPONENTS, X.shape[0])
    print(f"Running PCA to find {n_components_actual} eigenfaces...")
    pca = PCA(
        n_components=n_components_actual,
        svd_solver='randomized',
        random_state=RANDOM_STATE
    ).fit(X)
    print("PCA training complete.")
    
    # Save training
    print("Saving model files to disk...")
    try:
        joblib.dump(pca, PCA_MODEL_FILE)
        print(f"  - Saved: {PCA_MODEL_FILE}")
        np.save(MODEL_DIMS_FILE, np.array([MODEL_H, MODEL_W]))
        print(f"  - Saved: {MODEL_DIMS_FILE}")
        print("\n--- Training Complete ---")
        return True # Success
    except Exception as e:
        print(f"\n--- ERROR: Failed to save models ---")
        print(f"Details: {e}")
        return False # Fail

def run_live_system():
    """Runs the main application loop."""
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

    tracking = False
    bbox = None
    p0 = None
    old_gray = None
    initial_point_count = 0
    gmm_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    show_background = False
    show_all_annotations = True
    detector_mode = 0
    mode_labels = ["Haar + Recog", "HOG (LBP) Only"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_frame = frame.copy()

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

            # Re-detection
            if len(p_really_good) < (initial_point_count * RE_DETECTION_THRESHOLD) or len(p_really_good) < 5:
                tracking = False
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

                # Draw tracking annotations
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

                    # Eigenface Novelty Detection
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

                    # Draw detection annotations
                    if show_all_annotations:
                        cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
                        draw_text_with_outline(output_frame, label, (x, y - 10),
                                               font, 0.6, color)

                    # Switch to tracker if face confirmed
                    if can_track and not tracking:
                        tracking = True
                        bbox = (x, y, w, h)
                        mask = np.zeros_like(frame_gray)
                        mask[y:y+h, x:x+w] = 255
                        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                        if p0 is not None:
                            initial_point_count = len(p0)
                        else:
                            tracking = False # Abort if no features found

            # HOG (LBP) Mode
            elif detector_mode == 1:
                faces_lbp = lbp_detector.detectMultiScale(frame_gray, 1.1, 5, minSize=(30, 30))
                # Draw LBP annotations if enabled
                if show_all_annotations:
                    for (x, y, w, h) in faces_lbp:
                        cv2.rectangle(output_frame, (x, y), (x+w, y+h), FACE_COLOR, 2)

        # Update frame for next loop
        old_gray = frame_gray.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('b'): show_background = not show_background
        if key == ord('d'):
            detector_mode = (detector_mode + 1) % 2 # Cycle 0, 1
            tracking = False # Reset tracker
            p0 = None
            bbox = None
        if key == ord('t'): show_all_annotations = not show_all_annotations

        display_frame = mask_bgr if show_background else output_frame
        # Always draw the mode text
        if tracking: mode_text = "MODE: Tracking"
        draw_text_with_outline(display_frame, mode_text, (10, 30),
                               font, 0.7, TEXT_COLOR)
        cv2.imshow('Live System - Press "q" to quit', display_frame)
        
    print("--- Webcam feed stopped ---")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    critical_files = [HAAR_CASCADE_FILE, LBP_CASCADE_FILE, INPUT_VIDEO_PATH]
    if not all(os.path.exists(f) for f in critical_files):
        print("--- CRITICAL ERROR ---")
        print("One or more required files are missing and cannot be trained:")
        if not os.path.exists(HAAR_CASCADE_FILE): print(f"- {HAAR_CASCADE_FILE}")
        if not os.path.exists(LBP_CASCADE_FILE): print(f"- {LBP_CASCADE_FILE}")
        if not os.path.exists(INPUT_VIDEO_PATH): print(f"- {INPUT_VIDEO_PATH} (needed for training)")
        time.sleep(5)
        sys.exit()

    models_exist = all(os.path.exists(f) for f in MODEL_FILES)

    while True:
        # No models, forced training
        if not models_exist:
            print("--- Welcome ---")
            print("Trained model files not found.")
            print("The local training process must run once.")
            if train_pca_model_from_video():
                models_exist = True # Training succeeded
            else:
                print("Training failed. Exiting.")
                time.sleep(3)
                sys.exit() # Exit if training fails

        print("\n--- Main Menu ---")
        print("Trained models are ready.")
        print("[1] Run Webcam")
        print("[2] Re-Train Model (Deletes old models)")
        print("[3] Exit")

        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            run_live_system()

        elif choice == '2':
            print("Re-training model...")
            # Delete old models
            for f in MODEL_FILES:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not delete {f}. {e}")
            models_exist = False # Trigger training next loop

        elif choice == '3':
            print("Exiting.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

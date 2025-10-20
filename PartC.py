import cv2
import numpy as np
import os
import time

INPUT_VIDEO_PATH = 'Tam Nakano vs Saya Kamitami - Faces.mp4'
OUTPUT_VIDEO_PATH = 'face_tracker_output.mp4'
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml' # Manual import

DETECT_COLOR = (255, 0, 0) # Blue
TRACK_COLOR = (0, 0, 255)   # Red
POINT_COLOR = (0, 255, 0)   # Green

RE_DETECTION_THRESHOLD = 0.3  # Redetect if only 30% or less points of seen
BBOX_PADDING = 8

# Parameters for ShiTomasi
feature_params = dict( 
    maxCorners = 100,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7
)

# Parameters for Lucas-Kanade
lk_params = dict( 
    winSize = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def main():
    # Make sure XML is there
    if not os.path.exists(HAAR_CASCADE_FILE):
        print(f"Error: Could not find Haar file: {HAAR_CASCADE_FILE}")
        return
    haar_detector = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    print("Haar Cascade detector loaded successfully.")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {INPUT_VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    print(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")

    tracking = False
    bbox = None
    p0 = None
    old_gray = None
    last_good_point_count = 0 

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}...", end='\r')

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_frame = frame.copy()

        if tracking:
            p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                p1_good = p1[status == 1]
                p0_good = p0[status == 1]
            else:
                p1_good = np.array([]) 
                p0_good = np.array([])

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
            if len(p_really_good) < (last_good_point_count * RE_DETECTION_THRESHOLD) or len(p_really_good) < 5:
                tracking = False
                bbox = None
                p0 = None
                cv2.putText(output_frame, "TRACK LOST - RE-DETECTING", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Bounding Box Update
                x_new, y_new, w_new, h_new = cv2.boundingRect(p_really_good)
                
                x_new = max(0, x_new - BBOX_PADDING)
                y_new = max(0, y_new - BBOX_PADDING)
                w_new = min(width - x_new, w_new + 2 * BBOX_PADDING)
                h_new = min(height - y_new, h_new + 2 * BBOX_PADDING)
                
                bbox = (x_new, y_new, w_new, h_new)
                cv2.rectangle(output_frame, (x_new, y_new), (x_new + w_new, y_new + h_new), TRACK_COLOR, 2)
                
                for point in p_really_good:
                    cv2.circle(output_frame, (int(point[0]), int(point[1])), 3, POINT_COLOR, -1)
                
                p0 = p_really_good.reshape(-1, 1, 2)
                last_good_point_count = len(p0)
        
        if not tracking:
            faces = haar_detector.detectMultiScale(
                frame_gray, 
                scaleFactor=1.1, 
                minNeighbors=7,   # Change to make detector stricter (larger #)
                minSize=(25, 25) # Detect smnaller faces
            )
            
            if len(faces) > 0:
                tracking = True
                # Select the largest face found
                bbox = max(faces, key=lambda b: b[2] * b[3])
                x, y, w, h = bbox
                
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), DETECT_COLOR, 2)
                
                # Extract features
                mask = np.zeros_like(frame_gray)
                mask[y:y+h, x:x+w] = 255
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                
                if p0 is not None:
                    last_good_point_count = len(p0)
                    for point in p0:
                        cv2.circle(output_frame, (int(point[0][0]), int(point[0][1])), 3, POINT_COLOR, -1)
                else:
                    tracking = False
                    bbox = None

        old_gray = frame_gray.copy()
        out.write(output_frame)
        
    print(f"\nProcessing complete. Tracked video saved to: {OUTPUT_VIDEO_PATH}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
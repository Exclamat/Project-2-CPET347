import cv2
import numpy as np
import os
import time

# False = Video file instead
USE_WEBCAM = False 
INPUT_VIDEO_PATH = 'Tam Nakano vs Saya Kamitami - Faces.mp4' 
OUTPUT_VIDEO_PATH = 'face_detection_haar_vs_HOG.mp4'

# Local XML files for broken python install
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
LBP_CASCADE_FILE = 'lbpcascade_frontalface.xml'

HAAR_COLOR = (255, 0, 0) # Blue
LBP_COLOR = (0, 255, 0)  # Green

def main():
    
    # Make sure files load in correctly
    if not os.path.exists(HAAR_CASCADE_FILE):
        print(f"Error: Could not find Haar file: {HAAR_CASCADE_FILE}")
        print("Please download it and place it in the same folder as the script.")
        return
    haar_detector = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    print("Haar Cascade detector loaded successfully.")
    if not os.path.exists(LBP_CASCADE_FILE):
        print(f"Error: Could not find LBP file: {LBP_CASCADE_FILE}")
        print("Please download it and place it in the same folder as the script.")
        return
    lbp_detector = cv2.CascadeClassifier(LBP_CASCADE_FILE)
    print("LBP Cascade detector loaded successfully.")

    # Webcam
    if USE_WEBCAM:
        print("--- Starting real-time (webcam) detection ---")
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.exists(INPUT_VIDEO_PATH):
            print(f"Error: Input video not found at '{INPUT_VIDEO_PATH}'.")
            return
        print(f"--- Starting file processing for comparison video ---")
        cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Output file setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # .mp4
    
    out = None
    if not USE_WEBCAM:
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
        print(f"Output comparison video will be saved to: {OUTPUT_VIDEO_PATH}")

    frame_count = 0
    while True: # Frame by Frame processing
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}...", end='\r')

        output_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Haar
        t_start = time.perf_counter()
        faces_haar = haar_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        t_haar = (time.perf_counter() - t_start) * 1000

        for (x, y, w, h) in faces_haar:
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), HAAR_COLOR, 2)

        # LBP/HOG
        t_start = time.perf_counter()
        faces_lbp = lbp_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        t_lbp = (time.perf_counter() - t_start) * 1000

        for (x, y, w, h) in faces_lbp:
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), LBP_COLOR, 2)

        haar_text = f"Haar (Blue): {len(faces_haar)} faces ({t_haar:.2f} ms)"
        lbp_text = f"HOG (Green): {len(faces_lbp)} faces ({t_lbp:.2f} ms)"
        
        cv2.putText(output_frame, haar_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HAAR_COLOR, 2)
        cv2.putText(output_frame, lbp_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LBP_COLOR, 2)

        if USE_WEBCAM:
            cv2.imshow("Face Detection (Haar vs LBP) - Press 'q' to quit", output_frame)
        else:
            out.write(output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\nProcessing complete. Processed {frame_count} frames.")
    cap.release()
    if out is not None:
        out.release()
        print(f"\nComparison video saved to: {OUTPUT_VIDEO_PATH}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os

INPUT_VIDEO_PATH = 'cpet347_background.mp4'
OUTPUT_BACKGROUND_VIDEO = 'background_plate_gmm.mp4'
OUTPUT_MASK_VIDEO = 'foreground_mask_video.mp4'

GMM_HISTORY = 500
GMM_VAR_THRESHOLD = 16
DETECT_SHADOWS = True

def segment_video_gmm(video_path: str, bg_video_path: str, mask_video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # .mp4

    # GMM Subtractor
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=GMM_HISTORY, 
        varThreshold=GMM_VAR_THRESHOLD, 
        detectShadows=DETECT_SHADOWS
    )

    # Video Writer for mask
    mask_out = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True: # Frame by Frame processing
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}...", end='\r')

        # Get the mask, create binary mask, convert to 3 channel, write it
        fg_mask = subtractor.apply(frame)
        _, binary_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        mask_out.write(mask_bgr)

    print(f"\nPass 1 complete. {frame_count} frames.")
    print(f" Foreground video saved to: {mask_video_path}")
    
    cap.release()
    mask_out.release()

    # Main backgroud
    background_plate = subtractor.getBackgroundImage()
    
    cap = cv2.VideoCapture(video_path)
    
    # Writer for background
    bg_out = cv2.VideoWriter(bg_video_path, fourcc, fps, (width, height))

    frame_count_pass2 = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
            
        frame_count_pass2 += 1
        print(f"Writing background frame {frame_count_pass2}...", end='\r')
        
        # Background plate to frame
        bg_out.write(background_plate)

    print(f"\nPass 2 complete. {frame_count_pass2} frames.")
    print(f" Background video saved to: {bg_video_path}")

    cap.release()
    bg_out.release()
    print("--- Done ---")


if __name__ == "__main__":
        segment_video_gmm(
            video_path=INPUT_VIDEO_PATH,
            bg_video_path=OUTPUT_BACKGROUND_VIDEO,
            mask_video_path=OUTPUT_MASK_VIDEO
        )
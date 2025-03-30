import cv2
import numpy as np
import torch
import json
import argparse
from ultralytics import YOLO
from room_localization import RoomLocalizer

def process_frame(frame, model, localizer, conf_threshold=0.5):
    """
    Process a single frame with YOLOv11 and room localization
    
    Args:
        frame: Input image frame
        model: YOLOv11 model
        localizer: RoomLocalizer instance
        conf_threshold: Confidence threshold for detections
        
    Returns:
        processed_frame: Annotated frame
        results: List of person detections with room info
    """
    # Run YOLOv11 inference
    yolo_results = model(frame)
    
    # Extract detections
    detections = []
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls_id = box.cls[0].cpu().numpy()
            
            # Only keep person detections (class 0) above threshold
            if int(cls_id) == 0 and conf > conf_threshold:
                detections.append([x1, y1, x2, y2, conf, cls_id])
    
    # Process room localization
    results = []
    for det in detections:
        x1, y1, x2, y2, conf, _ = det
        room = localizer.locate_person([x1, y1, x2, y2])
        results.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "room": room
        })
    
    # Visualize results
    processed_frame = localizer.visualize_localization(frame, detections)
    
    return processed_frame, results

def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Room Detection")
    parser.add_argument("--model", default="yolov11n.pt", help="Path to YOLOv11 model")
    parser.add_argument("--boundaries", required=True, help="Path to room boundaries JSON file")
    parser.add_argument("--source", required=True, help="Source (0 for webcam, or video file path)")
    parser.add_argument("--output", help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Load YOLOv11 model
    model = YOLO(args.model)
    
    # Load room localizer
    localizer = RoomLocalizer(args.boundaries)
    
    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer if output is specified
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, results = process_frame(frame, model, localizer, args.conf)
        
        # Display room detection status
        y_offset = 30
        for i, res in enumerate(results):
            text = f"Person {i+1}: {res['room']}"
            cv2.putText(processed_frame, text, (10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30
        
        # Display frame
        cv2.imshow("YOLOv11 Room Detection", processed_frame)
        
        # Write frame to output video
        if args.output:
            out.write(processed_frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if args.output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
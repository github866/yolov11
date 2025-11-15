#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import sys
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
import time

def main(
    server_host='10.55.164.170', 
    server_port=9988, 
    video_src=0, 
    resize_wh=(640, 480), 
    mask_path=None, 
    axis_api_ip=None, 
    auth=None,
    use_webcam=False
):
    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f'Connected to server at {server_host}:{server_port}')

    if use_webcam:
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            print(f'Failed to open video source: {video_src}')
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video source FPS: {fps}")
    else:
        cap = None

    if mask_path is not None:
        room_map = cv2.imread(mask_path)
        room_map = cv2.resize(room_map, resize_wh)
        room_map = room_map.astype(np.uint8)
    else:
        # Create a fake room segmentation map for testing
        map_height, map_width = resize_wh[1], resize_wh[0]
        room_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        # Left part green (BGR)
        room_map[:, :map_width // 2] = (0, 255, 0)
        # Right part blue (BGR)
        room_map[:, map_width // 2:] = (255, 0, 0)

    try:
        frame_id = 0
        if use_webcam:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("No more frames or failed to read frame.")
                    break

                frame = cv2.resize(frame, resize_wh)

                # Prepare the data packet with display options
                data_packet = {
                    'frame': frame
                }

                # On the first frame, send the segmentation map
                if frame_id == 0:
                    data_packet['segmentation_map'] = room_map
                    print("Sending segmentation map with the first frame...")

                data = pickle.dumps(data_packet)
                size = len(data)
                # Send size then data
                sock.sendall(struct.pack("!L", size))
                sock.sendall(data)
                frame_id += 1

                # Receive result
                payload_size = struct.calcsize("!L")
                data = b""
                while len(data) < payload_size:
                    packet = sock.recv(4096)
                    if not packet:
                        print("Failed to receive size.")
                        return
                    data += packet
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("!L", packed_msg_size)[0]
                while len(data) < msg_size:
                    packet = sock.recv(4096)
                    if not packet:
                        print("Failed to receive data.")
                        return
                    data += packet
                response_packet = pickle.loads(data[:msg_size])
                processed_frame = response_packet['frame']
                metadata = response_packet['metadata']
                print(f"Frame: {frame_id} | Metadata: {metadata}")

                # Display processed frame
                cv2.imshow("YOLO-Pose Processed", processed_frame)
                            
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break
        else:
            start_time = time.time()
            frames_for_fps = 0
            while True:
                
                url = f"http://{axis_api_ip}/axis-cgi/jpg/image.cgi"
                print(f"auth: {auth}")
                auth = HTTPDigestAuth("admin", "Noldus123")
                response = requests.get(url=url, auth=auth, timeout=1)
                image_bytes = response.content
                np_arr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                frame = cv2.resize(frame, resize_wh)

                frames_for_fps += 1
                if frames_for_fps >= 30: # calculate every 30 frames
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fps = frames_for_fps / elapsed_time
                    print(f"Stream FPS: {fps:.2f}")
                    start_time = time.time()
                    frames_for_fps = 0

                # Prepare the data packet with display options
                data_packet = {
                    'frame': frame
                }

                # On the first frame, send the segmentation map
                if frame_id == 0:
                    data_packet['segmentation_map'] = room_map
                    print("Sending segmentation map with the first frame...")

                data = pickle.dumps(data_packet)
                size = len(data)
                # Send size then data
                sock.sendall(struct.pack("!L", size))
                sock.sendall(data)
                frame_id += 1

                # Receive result
                payload_size = struct.calcsize("!L")
                data = b""
                while len(data) < payload_size:
                    packet = sock.recv(4096)
                    if not packet:
                        print("Failed to receive size.")
                        return
                    data += packet
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("!L", packed_msg_size)[0]
                while len(data) < msg_size:
                    packet = sock.recv(4096)
                    if not packet:
                        print("Failed to receive data.")
                        return
                    data += packet
                response_packet = pickle.loads(data[:msg_size])
                processed_frame = response_packet['frame']
                metadata = response_packet['metadata']
                print(f"Frame: {frame_id} | Metadata: {metadata}")

                # Display processed frame
                cv2.imshow("YOLO-Pose Processed", processed_frame)
                            
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break
    finally:
        # cap.release()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Realtime YOLO-Pose Client')
    parser.add_argument('--host', type=str, default='10.55.164.170', help='Server IP')
    parser.add_argument('--port', type=int, default=9988, help='Server Port')
    parser.add_argument('--use_webcam', type=bool, default=False, help='Use video source (0 for webcam)')
    parser.add_argument('--video', type=str, default="0", help='Video source (0 for webcam or path)')
    parser.add_argument('--resize_wh', type=tuple, default=(960, 540), help='Resize width and height')
    parser.add_argument('--mask_path', type=str, default="./mask/456.png", help='Path to room segmentation mask')
    parser.add_argument('--auth', type=list, default=['admin', 'Noldus123'])
    parser.add_argument('--axis_api_ip', type=str, default="169.254.8.255")
    args = parser.parse_args()

    # Check if the video source is an integer or path
    try:
        video_src = int(args.video)
    except ValueError:
        video_src = args.video

    main(
        server_host=args.host, 
        server_port=args.port, 
        video_src=video_src, 
        resize_wh=args.resize_wh,
        mask_path=args.mask_path, 
        axis_api_ip=args.axis_api_ip,
        auth=args.auth, 
        use_webcam=args.use_webcam
    )
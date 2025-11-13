#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import sys
import numpy as np

def main(server_host='10.55.164.170', server_port=9988, video_src=0, resize_wh=(640, 360)):
    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f'Connected to server at {server_host}:{server_port}')

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f'Failed to open video source: {video_src}')
        sys.exit(1)

    # Create a fake room segmentation map for testing
    map_height, map_width = resize_wh[1], resize_wh[0]
    fake_room_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)
    # Left part green (BGR)
    fake_room_map[:, :map_width // 2] = (0, 255, 0)
    # Right part blue (BGR)
    fake_room_map[:, map_width // 2:] = (255, 0, 0)

    try:
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, resize_wh)
            if not ret:
                print("No more frames or failed to read frame.")
                break

            # Prepare the data packet with display options
            data_packet = {
                'frame': frame,
                'show_pose': True,
                'show_bbox': True,
                'show_hand': False,
                'show_face': False,
            }

            # On the first frame, send the segmentation map
            if frame_id == 0:
                data_packet['segmentation_map'] = fake_room_map
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
        cap.release()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Realtime YOLO-Pose Client')
    parser.add_argument('--host', type=str, default='10.55.164.170', help='Server IP')
    parser.add_argument('--port', type=int, default=9988, help='Server Port')
    parser.add_argument('--video', type=str, default='0', help='Video source (0 for webcam or path)')
    parser.add_argument('--resize_wh', type=tuple, default=(640, 360), help='Resize width and height')
    args = parser.parse_args()

    # Check if the video source is an integer or path
    try:
        video_src = int(args.video)
    except ValueError:
        video_src = args.video

    main(server_host=args.host, server_port=args.port, video_src=video_src, resize_wh=args.resize_wh)

import os
import cv2
import glob
import xml.etree.ElementTree as ET
import argparse

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def create_video_from_images(input_folder, frame_rate=30):
    # Get all PNG files in the input folder, sorted by filename
    images = sorted(glob.glob(os.path.join(input_folder, '*.png')))

    if not images:
        print("No PNG images found in the specified folder.")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape

    # Define video writer with MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
    video = cv2.VideoWriter(os.path.join(input_folder, "output_video.mp4"), fourcc, frame_rate, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print("Video saved successfully")

def parse_annotations(xml_file):
    """
    Parse the CVAT XML file and return a dictionary mapping frame numbers
    to a list of annotations. Each annotation is a tuple (label, (x, y), color).

    Only keypoints marked as inside the frame (outside="0") are included.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = {}
    
    # Iterate over each track in the XML.
    for track in root.findall('track'):
        label = track.attrib.get('label')
        # Define colors based on CVAT annotations.
        # Note: OpenCV uses BGR, so we convert from the provided hex colors.
        if label == 'pupil_phantom':
            # Hex "#fa3253" -> RGB (250, 50, 83) -> BGR (83, 50, 250)
            color = (83, 50, 250)
        elif label == 'pupil_human':
            # Hex "#24b353" -> RGB (36, 179, 83) -> BGR (83, 179, 36)
            color = (83, 179, 36)
        else:
            color = (0, 255, 255)  # Fallback color if needed.

        # Process each annotated point.
        for point in track.findall('points'):
            frame = int(point.attrib.get('frame'))
            outside = point.attrib.get('outside')
            # Skip points marked as outside the frame.
            if outside == '1':
                continue
            pts = point.attrib.get('points')
            x_str, y_str = pts.split(',')
            x = float(x_str)
            y = float(y_str)
            # Save the annotation for this frame.
            if frame not in annotations:
                annotations[frame] = []
            annotations[frame].append((label, (int(x), int(y)), color))
    return annotations

def annotate_video(video_path, annotations, output_path):
    """
    Read the video file, draw the annotations on frames that have keypoints,
    and write the annotated frames to a new output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # If there are any annotations for this frame, draw them.
        if frame_num in annotations:
            for label, (x, y), color in annotations[frame_num]:
                # Draw a filled circle at the keypoint.
                cv2.circle(frame, (x, y), 5, color, -1)
                # Optionally, annotate with text (offset a bit from the keypoint).
                cv2.putText(frame, label, (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    print("Annotated video saved as:", output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Annotate video with keypoints from CVAT XML annotations.'
    )
    # Use nargs='?' to make these positional arguments optional with defaults.
    parser.add_argument('video', nargs='?', default='C:/Users/michaelfeng/Documents/IML/data/EyeVideos/e1.mp4', help='Path to the input video file')
    parser.add_argument('xml', nargs='?', default='C:/Users/michaelfeng/Documents/IML/data/EyeVideos/annotations/e1_annotations.xml', help='Path to the CVAT XML annotations file')
    parser.add_argument('output', nargs='?', default='C:/Users/michaelfeng/Documents/IML/data/EyeVideos/annotations/e1_annotated.mp4', help='Path to the output annotated video file')
    args = parser.parse_args()

    # Parse the annotations from the XML file.
    annotations = parse_annotations(args.xml)
    # Process the video and produce the annotated output.
    annotate_video(args.video, annotations, args.output)

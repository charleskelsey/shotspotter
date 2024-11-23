
from ultralytics import YOLO
import torch
import cv2
import os

combined_boxes= []
combined_boxes= []

model1= YOLO('training/runs/detect/train9/weights/best.pt')
model2= YOLO('training/runs/detect/train3/weights/best.pt')

print(model1.names)
print(model2.names)
desired_classes_model1 = [0, 1, 2]
desired_classes_model2 = [0, 1]

results1 = model1.predict('input_vid/wet_book.mp4', classes=desired_classes_model1)
results2 = model2.predict('input_vid/wet_book.mp4', classes=desired_classes_model2)

offset = max(desired_classes_model1) + 1  # In this case, offset = 2

combined_results = []
num_frames = min(len(results1), len(results2))

for i in range(num_frames):
    boxes_list = []

    # Extract boxes from model1
    boxes1 = results1[i].boxes
    if boxes1 is not None:
        for j in range(len(boxes1)):
            boxes_list.append(boxes1[j])

    # Extract boxes from model2 and adjust class indices
    boxes2 = results2[i].boxes
    if boxes2 is not None:
        boxes2.cls += offset  # Adjust class indices
        for j in range(len(boxes2)):
            boxes_list.append(boxes2[j])

    # Create a new Boxes object if there are any boxes
    if boxes_list:
        # Convert list of boxes to a data tensor
        combined_data = torch.stack([box.data.squeeze() for box in boxes_list])
        #combined_boxes = Boxes(combined_data)
    else:
        combined_boxes = None

    # Create a new result object with combined boxes
    combined_result = results1[i]
    combined_result.boxes = combined_boxes
    combined_results.append(combined_result)


for frame_number, result in enumerate(combined_results):
    print(f"Frame {frame_number}")
    print (result)
    print('===========================================')
    for box in result.boxes:
        print(box)


def perform_nms(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return []

    # Prepare tensors
    boxes_tensor = torch.cat([torch.tensor([[*box.xyxy[0], box.conf[0], box.cls[0]]]) for box in boxes])

    # Apply NMS
    keep_indices = torch.ops.torchvision.nms(
        boxes_tensor[:, :4],  # Box coordinates
        boxes_tensor[:, 4],   # Confidence scores
        iou_threshold
    )

    # Return kept boxes
    return [boxes[i] for i in keep_indices]



import numpy as np
import cv2
import torch
import dataset
from pathlib import Path
from tqdm import tqdm
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from model import build_pointing_network
from draw_arrow import WIDTH, HEIGHT

""" My added modules"""
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class_labels = {
    44: 'bottle',
    47: 'cup',
    73: 'laptop',
    74: 'mouse',
    53: 'apple',
    52: 'banana'
}
# box_position = (225, 550)
# box_size= (50 , 150)
@hydra.main(version_base=None, config_path="../conf", config_name="base")
def main(cfg: DictConfig) -> None:
    import logging

    logging.info(
        "Successfully loaded settings:\n"
        + "==================================================\n"
        + f"{OmegaConf.to_yaml(cfg)}"
        + "==================================================\n"
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        logging.warning("Running DeePoint with CPU takes a long time.")

    assert (
        cfg.movie is not None
    ), "Please specify movie path as `movie=/path/to/movie.mp4`"

    assert (
        cfg.lr is not None
    ), "Please specify whether the pointing hand is left or right with `lr=l` or `lr=r`."

    assert cfg.ckpt is not None, "checkpoint should be specified for evaluation"

    cfg.hardware.bs = 2
    cfg.hardware.nworkers = 0
    ds = dataset.MovieDataset(cfg.movie, cfg.lr, cfg.model.tlength, DEVICE)
    dl = DataLoader(
        ds,
        batch_size=cfg.hardware.bs,
        num_workers=cfg.hardware.nworkers,
    )

    network = build_pointing_network(cfg, DEVICE)

    # Since the model trained using pytorch lightning contains `model.` as an prefix to the keys of state_dict, we should remove them before loading
    model_dict = torch.load(cfg.ckpt, map_location = DEVICE)["state_dict"]
    new_model_dict = dict()
    for k, v in model_dict.items():
        new_model_dict[k[6:]] = model_dict[k]
    model_dict = new_model_dict
    network.load_state_dict(model_dict)
    network.to(DEVICE)

    Path("demo").mkdir(exist_ok=True)
    fps = 15
    out_green = cv2.VideoWriter(
        f"demo/{Path(cfg.movie).name}-processed-green-{cfg.lr}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (WIDTH, HEIGHT),
    )
    # out_greenred = cv2.VideoWriter(
    #     f"demo/{Path(cfg.movie).name}-processed-greenred-{cfg.lr}.mp4",
    #     cv2.VideoWriter_fourcc("m", "p", "4", "v"),
    #     fps,
    #     (WIDTH, HEIGHT),
    # )

    prev_arrow_base = np.array((0, 0))

    for batch in tqdm(dl):
        result = network(batch)
        # bs may be smaller than cfg.hardware.bs for the last iteration
        bs = batch["abs_joint_position"].shape[0]
        for i_bs in range(bs):
            joints = batch["abs_joint_position"][i_bs][-1].to("cpu").numpy()
            image = batch["orig_image"][i_bs].to("cpu").numpy() / 255

            direction = result["direction"][i_bs]
            prob_pointing = float(
                (result["action"][i_bs, 1].exp() / result["action"][i_bs].exp().sum())
            )
            print(f"{prob_pointing=}")

            ORIG_HEIGHT, ORIG_WIDTH = image.shape[:2]

            scale_x = 960 / ORIG_WIDTH
            scale_y = 720 / ORIG_HEIGHT

            hand_idx = 9 if batch["lr"][i_bs] == "l" else 10
            if (joints[hand_idx] < 0).any():
                arrow_base = prev_arrow_base
            else:
                arrow_base = (
                    joints[hand_idx] / np.array((ORIG_WIDTH, ORIG_HEIGHT)) * 2 - 1
                )
                prev_arrow_base = arrow_base

            """ my added code starts here"""
            box_size, box_position, object_class = find_pointed_object_details(joints[hand_idx], direction, image, scale_x, scale_y)

            image_green = draw_arrow_on_image(
                image,
                (
                    arrow_base[0],
                    -arrow_base[1],
                    direction[0].cpu(),
                    direction[2].cpu(),
                    -direction[1].cpu(),
                    box_position,
                    box_size,
                    object_class,
                ),
                dict(
                    acolor=(
                        0,
                        1,
                        0,
                    ),  # Green. OpenCV uses BGR
                    asize=0.05 * prob_pointing,
                    offset=0.02,
                ),
            )
            # image_greenred = draw_arrow_on_image(
            #     image,
            #     (
            #         arrow_base[0],
            #         -arrow_base[1],
            #         direction[0].cpu(),
            #         direction[2].cpu(),
            #         -direction[1].cpu(),
            #     ),
            #     dict(
            #         acolor=(
            #             0,
            #             prob_pointing,
            #             1 - prob_pointing,
            #         ),  # Green to red. OpenCV uses BGR
            #         asize=0.05 * prob_pointing,
            #         offset=0.02,
            #     ),
            # )

            cv2.imshow("", image_green)
            cv2.waitKey(10)

            out_green.write((image_green * 255).astype(np.uint8))
            # out_greenred.write((image_greenred * 255).astype(np.uint8))

    return


def draw_arrow_on_image(image, arrow_spec, kwargs):
    """
    Params:
    image: np.ndarray(height, width, 3), with dtype=float, value in the range of [0,1]
    arrow_spec, kwargs: options for render_frame
    Returns:
    image: np.ndarray(HEIGHT, WIDTH, 3), with dtype=float, value in the range of [0,1]
    """
    from draw_arrow import render_frame, WIDTH, HEIGHT

    ret_image = cv2.resize(image, (WIDTH, HEIGHT)).astype(float)
    img_arrow = render_frame(*arrow_spec, **kwargs).astype(float) / 255
    arrow_mask = (img_arrow.sum(axis=2) == 0.0).astype(float)[:, :, None]
    ret_image = arrow_mask * ret_image + (1 - arrow_mask) * img_arrow
    return ret_image


""" My added code to this program starts here"""

# Load a pre-trained object detection model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def detect_objects(image,scale_x, scale_y, device = DEVICE):
    """
    Uses a pre-trained model to detect objects in the image and returns bounding boxes and scores
    for specific object types (bottle, cup, laptop, mouse) with a confidence score greater than 0.98,
    printing the detected objects.
    """
    # COCO class labels for the interested classes


    # Define the classes you are interested in (COCO dataset class IDs)
    interested_classes = [44, 47, 73, 74, 53, 52]  # Bottle, Cup, Laptop, Mouse

    # Ensure the image is in float32 format if it's not already
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Normalize the image to be between 0 and 1 if it's not already
    if image.max() > 1.0:
        image /= 255.0
    
    # Convert the image to a tensor
    transform = T.Compose([T.ToTensor()])

    # transform = T.Compose([
    # T.ToPILImage(),
    # T.Resize((800, 800)),
    # T.ToTensor()
    # ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move tensor to the same device as the model
    image_tensor = image_tensor.to(device)

    # Perform detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Filter predictions based on the interested classes and confidence threshold
    pred_boxes = predictions[0]['boxes']
    pred_scores = predictions[0]['scores']
    pred_labels = predictions[0]['labels']

    # Filter out boxes for unwanted classes and low confidence
    class_mask = torch.isin(pred_labels, torch.tensor(interested_classes).to(device))
    high_confidence_mask = pred_scores > 0.90  # Confidence threshold
    final_mask = class_mask & high_confidence_mask

    filtered_boxes = pred_boxes[final_mask]
    filtered_boxes_resized = adjust_box_coordinates(filtered_boxes, scale_x, scale_y)

    filtered_scores = pred_scores[final_mask]
    filtered_labels = pred_labels[final_mask]

    # Print detected objects
    for label, score in zip(filtered_labels.tolist(), filtered_scores.tolist()):
        if score > 0.9:  # Double-checking the score for printing
            print(f"Detected {class_labels[label]} with confidence {score:.2f}")
            # detected_class_name= class_labels[label]

    return {'boxes': filtered_boxes_resized, 'scores': filtered_scores, 'labels': filtered_labels}



def find_pointed_object_details(joints, direction, image, scale_x, scale_y, min_distance_threshold=0.1):
    """
    Detects objects and finds the details of the object pointed at by the direction vector,
    ensuring that only objects in the positive direction of the pointing vector are considered.
    """

    box_size = (1, 1)
    box_position = (1, 1)
    best_class_name = "None"
    threshold_angle = np.radians(60)
    ORIG_WIDTH, ORIG_HEIGHT = image.shape[1], image.shape[0]
    predictions = detect_objects(image, scale_x, scale_y)
    boxes = predictions['boxes'][predictions['scores'] > 0.5]  # Apply confidence threshold
    labels = predictions["labels"][predictions["scores"]> 0.5]


    # Calculate normalized base and direction vectors
    norm_base = joints / np.array([ORIG_WIDTH, ORIG_HEIGHT])
    direction = direction.detach().numpy()  # Detach and convert to numpy
    norm_direction = np.array([direction[0], -direction[2]])  # Ensure this matches your coordinate system correctly
    norm_direction /= np.linalg.norm(norm_direction)  # Normalize

    best_box = None
    best_box_distance = float('inf')
    best_label = None

    # Print debug information
    print("Norm Direction:", norm_direction)

    for i, box in enumerate(boxes) :
        box = box.detach().numpy()  # Convert tensor to numpy array
        label = labels[i].item()
        box_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) / np.array([ORIG_WIDTH, ORIG_HEIGHT])
        box_vector = box_center - norm_base
        box_distance = np.linalg.norm(box_vector)
        box_vector_normalized = box_vector / box_distance  # Normalize

        # Calculate the dot product to ensure the box is in the positive direction of the pointing vector
        dot_product = np.dot(norm_direction, box_vector_normalized)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip for safety due to floating point precision

        # Print debug information
        print("Box Center:", box_center, "Dot Product:", dot_product, "Box Distance:", box_distance)

        if dot_product > 0 and box_distance > min_distance_threshold and box_distance < best_box_distance and angle <= threshold_angle:
            best_box_distance = box_distance
            best_box = box
            best_label = label
            print(best_label)

    if best_box is not None:
        box_size = (best_box[2] - best_box[0], best_box[3] - best_box[1])
        box_position = ((best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2)
        best_class_name = class_labels.get(best_label, "Unknown")
        print(best_class_name)
        return box_size, box_position , best_class_name
    else:
        return box_size, box_position, best_class_name
    

def adjust_box_coordinates(boxes,scale_x, scale_y):
    """
    Adjusts bounding box coordinates from original dimensions to target dimensions.
    
    Args:
    boxes (np.array): Array of bounding boxes [x1, y1, x2, y2].
    orig_dim (tuple): Original dimensions (width, height) of the image.
    target_dim (tuple): Target dimensions (width, height) for rendering.

    Returns:
    np.array: Adjusted bounding boxes.
    """

    # Adjust boxes
    boxes[:, 0] *= scale_x  # x1
    boxes[:, 2] *= scale_x  # x2
    boxes[:, 1] *= scale_y  # y1
    boxes[:, 3] *= scale_y  # y2

    return boxes

"""
# Example usage:
orig_width, orig_height = ORIG_WIDTH, ORIG_HEIGHT  # Dimensions used during detection
render_width, render_height = 960, 720  # Dimensions of the rendered frame

# Assuming 'detected_boxes' is an array of detected bounding boxes [x1, y1, x2, y2]
adjusted_boxes = adjust_box_coordinates(detected_boxes, (orig_width, orig_height), (render_width, render_height))

"""

# Example usage in your processing loop
# Assume 'image' is a numpy array, 'joints' and 'direction' are obtained from your system
"""box_size, box_position = find_pointed_object_details(joints[hand_idx], direction, image)
if box_size and box_position:
    print("Box Size (width, height):", box_size)
    print("Box Position (center x, center y):", box_position)
else:
    print("No object found in the pointing direction")"""


if __name__ == "__main__":
    main()

"""OmniParser-based visual actions for processing GUI screenshots."""

import logging
import time
from typing import Tuple, List, Union, Any
import torch
import io
import os
from PIL import Image, ImageDraw, ImageFont
from crab.utils.common import base64_to_image, image_to_base64
from crab.core.decorators import action
from crab.core.models import Action
from crab.core.models.action import Action
import pathlib
import base64
import numpy as np
import traceback
import tempfile
from io import BytesIO

# Import OmniParser utilities
from OmniParser.utils import get_yolo_model, check_ocr_box, remove_overlap, get_som_labeled_img, predict_yolo

# Constants
BOX_TRESHOLD = 0.05  # Lower threshold for box detection to catch more potential GUI elements
YOLO_MODEL_PATH = os.path.join(pathlib.Path(__file__).parent.parent.parent, "OmniParser/weights/icon_detect/model.pt")

logger = logging.getLogger(__name__)

# Global model instances
yolo_model = None

# Type alias for box and description tuple
AnnotatedBoxType = Tuple[List[float], str]

def check_omniparser_import():
    """Check if OmniParser is imported correctly."""
    global yolo_model

    start_time = time.time()
    logger.info("Loading YOLO model...")
    if yolo_model is None:
        logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
        yolo_model = get_yolo_model(YOLO_MODEL_PATH)
    yolo_time = time.time()
    logger.info(f"YOLO model loaded in {yolo_time - start_time:.2f} seconds")

def detect_gui_elements(image: Union[str, Image.Image], box_threshold: float = BOX_TRESHOLD, text_threshold: float = 0.3) -> List[Any]:
    """Detect GUI elements in an image using OmniParser.
    
    Args:
        image: Either a base64 encoded image string or a PIL Image
        box_threshold: Confidence threshold for box detection
        text_threshold: Confidence threshold for text detection
        
    Returns:
        List of detection results for each image, where each result is a list of tuples (box, description)
    """
    try:
        global yolo_model
        
        logger.info("Starting detect_gui_elements")
        start_time = time.time()
        
        # Convert input to PIL Image if needed
        if isinstance(image, str):
            logger.info("Converting base64 to PIL Image")
            image = base64_to_image(image)
            
        # Log image details
        logger.info(f"Processing image: {image.size}x{image.mode}")
        
        # Create temporary file for input image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as temp_file:
            logger.info(f"Saving temporary file: {temp_file.name}")
            image.save(temp_file.name)
            
            # Load or use global YOLO model
            if yolo_model is None:
                logger.info("Loading YOLO model (first time)")
                check_omniparser_import()
                load_start = time.time()
                logger.info(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
                yolo_model = get_yolo_model(YOLO_MODEL_PATH)
                logger.info(f"YOLO model loaded in {time.time() - load_start:.2f} seconds")
            else:
                logger.info("Using existing YOLO model")
            
            # Get predictions
            logger.info("Running YOLO predictions")
            predict_start = time.time()
            boxes, logits, _ = predict_yolo(
                model=yolo_model,
                image_path=temp_file.name,
                box_threshold=box_threshold,
                imgsz=640  # Standard size
            )
            logger.info(f"YOLO predictions completed in {time.time() - predict_start:.2f} seconds")
            
            # Convert to pixel coordinates
            w, h = image.size
            boxes = boxes.cpu()  # Move to CPU first
            detections = []
            for box, logit in zip(boxes, logits):
                x1, y1, x2, y2 = box.tolist()
                detections.append(([x1, y1, x2, y2], f"GUI element ({logit:.2f})"))
                logger.debug(f"Added box: {[x1, y1, x2, y2]} with confidence {logit:.2f}")
            
            logger.info(f"Found {len(detections)} boxes")
            total_time = time.time() - start_time
            logger.info(f"Total detection time: {total_time:.2f} seconds")
            return [detections]
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return [[]]  # Return empty boxes for failed image

def get_ocr_boxes(
    image: Image.Image,
    use_paddle: bool = True,
) -> list[AnnotatedBoxType]:
    """Get the bounding boxes of text in the image using OCR.

    Args:
        image: The target image.
        use_paddle: If True, use PaddleOCR, otherwise use EasyOCR.

    Returns:
        The list of tuples (bounding boxes, text).
    """
    # Save image to temporary file since OCR requires file path
    with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
        image.save(temp_file.name)
        (text, boxes), _ = check_ocr_box(
            temp_file.name,
            display_img=False,
            output_bb_format='xyxy',
            use_paddleocr=use_paddle
        )
    
    return list(zip(boxes, text))

@action(local=True)
def detect_and_annotate_gui_elements(
    image: str,
    env,
    font_size: int = 12,
    use_paddle: bool = True,
) -> dict[str, Union[str, list[AnnotatedBoxType]]]:
    """Detect and annotate GUI elements in an image.
    
    Args:
        image: Base64 encoded image string
        env: Environment object for storing detection results
        font_size: Font size for annotations
        use_paddle: Whether to use PaddleOCR for text detection
        
    Returns:
        Dictionary containing:
        - "image": Base64 encoded annotated image
        - "boxes": List of annotated boxes with their descriptions
    """
    try:
        logger.info("Starting detect_and_annotate_gui_elements")
        logger.info(f"Input image length: {len(image) if image else 'None'}")
        logger.info(f"Font size: {font_size}, Use PaddleOCR: {use_paddle}")

        # Create temporary file for image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            logger.info(f"Created temp file: {temp_file.name}")
            image_data = base64.b64decode(image)
            temp_file.write(image_data)
            temp_file.flush()

        # Load YOLO model
        yolo_model = get_yolo_model(YOLO_MODEL_PATH)
        logger.info("YOLO model loaded successfully")

        # Get OCR boxes first
        img = Image.open(BytesIO(image_data))
        logger.info(f"Loaded image size: {img.size}")
        ocr_boxes = get_ocr_boxes(img, use_paddle=use_paddle)
        logger.info(f"OCR detected {len(ocr_boxes)} text boxes")

        # Get YOLO boxes
        boxes, logits, phrases = predict_yolo(yolo_model, temp_file.name, BOX_TRESHOLD, imgsz=640)
        logger.info(f"YOLO detected {len(boxes)} GUI elements")
        logger.info(f"Box coordinates: {boxes}")
        logger.info(f"Box logits: {logits}")
        
        # Load original image for annotation
        img = Image.open(temp_file.name)
        img = img.convert('RGB')  # Ensure RGB mode
        logger.info(f"Loaded original image for annotation: {img.size}")
        
        # Convert YOLO boxes to our format and combine with OCR boxes
        yolo_boxes = []
        boxes = boxes.cpu()  # Move to CPU first
        for box, logit in zip(boxes, logits):
            x1, y1, x2, y2 = box.tolist()
            yolo_boxes.append(([x1, y1, x2, y2], f"element_{logit:.2f}"))
        logger.info(f"Converted YOLO boxes: {yolo_boxes}")

        all_boxes = ocr_boxes + yolo_boxes
        logger.info(f"Combined boxes count: {len(all_boxes)}")
        
        # Filter boxes using overlap removal
        filtered_boxes = []
        for i, (box1, label1) in enumerate(all_boxes):
            should_add = True
            for j, (box2, label2) in enumerate(filtered_boxes):
                if i != j:
                    iou = calculate_iou(box1, box2)
                    if iou > 0.5:  # Increased overlap threshold from 0.3 to 0.5
                        # If boxes overlap significantly, keep the one with text label
                        if label2.startswith('element_') and not label1.startswith('element_'):
                            filtered_boxes[j] = (box1, label1)
                        should_add = False
                        break
            if should_add:
                filtered_boxes.append((box1, label1))
        
        logger.info(f"Filtered boxes count: {len(filtered_boxes)}")
        
        # Map back to original labels
        final_boxes = []
        for i, (box, desc) in enumerate(filtered_boxes):
            # Convert box to list if it's a tensor
            if isinstance(box, torch.Tensor):
                box = box.tolist()
            if i < len(ocr_boxes):
                final_boxes.append((box, all_boxes[i][1]))  # OCR text
            else:
                final_boxes.append((box, all_boxes[i][1]))  # YOLO label
        logger.info(f"Final boxes with labels: {final_boxes}")

        # Store results in environment
        if hasattr(env, 'set') and callable(env.set):
            env.set("element_label_map", [box[1] for box in final_boxes])
            env.set("element_position_map", [box[0] for box in final_boxes])
        else:
            env.element_label_map = [box[1] for box in final_boxes]
            env.element_position_map = [box[0] for box in final_boxes]
        logger.info("Stored results in environment")

        # Draw boxes on image with distinct colors and clear labels
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        
        for i, (box, label) in enumerate(final_boxes):
            logger.info(f"Drawing box {box} with label {label}")
            # Alternate between different colors for better visibility
            color = "red" if i % 2 == 0 else "blue"
            
            # Draw box with thicker outline
            draw.rectangle(box, outline=color, width=3)
            
            # Draw text with background for better readability
            text_bbox = draw.textbbox((box[0], box[1] - font_size - 4), label, font=font)
            draw.rectangle(text_bbox, fill="white")
            draw.text((box[0], box[1] - font_size - 4), label, fill=color, font=font)

        # Convert back to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG", quality=95)  # Higher quality PNG
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logger.info(f"Successfully converted annotated image back to base64 (length: {len(img_str)})")

        return {
            "image": img_str,
            "boxes": final_boxes
        }

    except Exception as e:
        logger.error(f"Error in detect_and_annotate_gui_elements: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

@action(local=True)
def detect_and_annotate_gui_elements_legacy(image: str, env) -> tuple[str, list[AnnotatedBoxType]]:
    """Detect and annotate GUI elements in an image.
    
    Args:
        image: Base64 encoded image
        env: Environment object for storing detection results
        
    Returns:
        Base64 encoded annotated image
    """
    try:
        # Ensure models are loaded
        global yolo_model
        if yolo_model is None:
            logger.info("Initializing models")
            check_omniparser_import()
            
        logger.info("Converting base64 to PIL Image")
        img = base64_to_image(image)
        
        logger.info("Detecting GUI elements")
        detections = detect_gui_elements(img)[0]
        logger.info(f"Found {len(detections)} elements")
        
        # Store detections in environment
        if hasattr(env, 'store') and callable(env.store):
            logger.info("Storing detections in environment")
            env.store("detected_boxes", detections)
        else:
            logger.warning("Environment object does not support storing detections")
        
        # Draw boxes on image
        logger.info("Drawing boxes on image")
        draw = ImageDraw.Draw(img)
        for i, (box, desc) in enumerate(detections):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1-15), desc, fill="red")
            
        # Convert back to base64
        logger.info("Converting back to base64")
        result = image_to_base64(img)
        logger.info("Processing completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error annotating image: {str(e)}")
        return image  # Return original image on error

@action(local=True)
def get_elements_prompt(image: str, env) -> tuple[str, str]:
    """Generate a descriptive prompt for the detected GUI elements.

    Uses the previously detected GUI elements stored in the environment to generate a
    natural language description of the elements and their locations.

    Args:
        image: Base64 encoded annotated image
        env: Environment object containing detected boxes

    Returns:
        A tuple containing:
        - Base64 encoded annotated image
        - Natural language description of detected elements
    """
    try:
        logger.info("Starting get_elements_prompt")
        
        # Get detected boxes from environment
        boxes = []
        if hasattr(env, 'get') and callable(env.get):
            logger.info("Environment has get method, retrieving boxes")
            boxes = env.get("element_position_map", [])
            labels = env.get("element_label_map", [])
            boxes = list(zip(boxes, labels))
            logger.info(f"Retrieved {len(boxes)} boxes from environment")
        elif hasattr(env, 'element_position_map') and hasattr(env, 'element_label_map'):
            logger.info("Environment has direct attributes, retrieving boxes")
            boxes = list(zip(env.element_position_map, env.element_label_map))
            logger.info(f"Retrieved {len(boxes)} boxes from environment")
        else:
            logger.warning("Environment does not have element maps")
        
        logger.info(f"Retrieved {len(boxes)} boxes from environment")
        
        # Generate natural language description
        if len(boxes) == 0:
            logger.info("No boxes found, generating empty description")
            description = "I can see the following elements:\nNo GUI elements were detected in the image."
        else:
            logger.info(f"Generating description for {len(boxes)} elements")
            description = "I can see the following elements:\n"
            
            # Add details about each element
            for i, (box, desc) in enumerate(boxes):
                logger.debug(f"Processing element {i+1}: {desc}")
                x1, y1, x2, y2 = box
                description += f"- {desc} located at coordinates ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})\n"
                logger.debug(f"Added element {i+1} to description")
        
        logger.info("Description generated successfully")
        return image, description
        
    except Exception as e:
        logger.error(f"Error generating prompt: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return image, "I can see the following elements:\nError generating description of GUI elements."

@action(local=True)
def get_elements_prompt_action(image: str, env) -> tuple[str, str]:
    """Wrapped version of get_elements_prompt that uses the action decorator."""
    return get_elements_prompt(image, env)

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

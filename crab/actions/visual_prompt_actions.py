# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
import logging
from functools import cache
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from crab import action
from crab.utils.common import base64_to_image, image_to_base64

logger = logging.getLogger(__name__)

try:
    import easyocr
    import numpy as np
    import torch
    from transformers import (
        AutoProcessor,
        GroundingDinoForObjectDetection,
        GroundingDinoProcessor,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    TRANSFORMERS_ENABLE = True
except ImportError:
    TRANSFORMERS_ENABLE = False

BoxType = tuple[int, int, int, int]
AnnotatedBoxType = tuple[BoxType, str | None]


def check_transformers_import() -> None:
    if not TRANSFORMERS_ENABLE:
        raise ImportError(
            "Please install the required dependencies to use this function by running"
            " `pip install crab-framework[client]`"
        )


def _calculate_iou(box1: BoxType, box2: BoxType) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    iou = interArea / unionArea

    return iou


def _calculate_center(box: BoxType) -> tuple[int, int]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _remove_invalid_boxes(
    boxes_with_label: AnnotatedBoxType, width: int, height: int
) -> AnnotatedBoxType:
    boxes = [box[0] for box in boxes_with_label]
    boxes_to_remove = set()
    for idx, box in enumerate(boxes):
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            boxes_to_remove.add(idx)
            continue
        if box[0] >= box[2] or box[1] >= box[3]:
            boxes_to_remove.add(idx)
            continue

    boxes_filt = [
        box for idx, box in enumerate(boxes_with_label) if idx not in boxes_to_remove
    ]
    return boxes_filt


def _filter_boxes_by_center(
    boxes_with_label: list[AnnotatedBoxType], center_dis_thresh: float
) -> list[AnnotatedBoxType]:
    boxes = [box[0] for box in boxes_with_label]
    boxes_to_remove = set()
    for i in range(len(boxes)):
        if i in boxes_to_remove:
            continue
        center_i = _calculate_center(boxes[i])
        for j in range(i + 1, len(boxes)):
            center_j = _calculate_center(boxes[j])
            # fmt: off
            center_close = ((center_i[0] - center_j[0]) ** 2 + 
                            (center_i[1] - center_j[1]) ** 2 < 
                            center_dis_thresh**2)
            # fmt: on
            if center_close:
                boxes_to_remove.add(j)

    boxes_filt = [
        box for idx, box in enumerate(boxes_with_label) if idx not in boxes_to_remove
    ]
    return boxes_filt


def _box_a_in_b(a: BoxType, b: BoxType) -> bool:
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]


def _filter_boxes_by_overlap(
    boxes_with_label: list[AnnotatedBoxType],
) -> list[AnnotatedBoxType]:
    boxes = [box[0] for box in boxes_with_label]
    boxes_to_remove = set()
    for i in range(len(boxes)):
        if i in boxes_to_remove:
            continue
        for j in range(len(boxes)):
            if i != j and _box_a_in_b(boxes[i], boxes[j]):
                boxes_to_remove.add(j)

    boxes_filt = [
        box for idx, box in enumerate(boxes_with_label) if idx not in boxes_to_remove
    ]
    return boxes_filt


def _filter_boxes_by_iou(
    boxes_with_label: list[AnnotatedBoxType], iou_threshold=0.5
) -> list[AnnotatedBoxType]:
    boxes = [box[0] for box in boxes_with_label]
    boxes_to_remove = set()
    for i in range(len(boxes)):
        if i in boxes_to_remove:
            continue
        for j in range(i + 1, len(boxes)):
            iou = _calculate_iou(boxes[i], boxes[j])
            if iou >= iou_threshold:
                boxes_to_remove.add(j)

    boxes_filt = [
        box for idx, box in enumerate(boxes_with_label) if idx not in boxes_to_remove
    ]
    return boxes_filt


def _draw_boxes(
    image: Image.Image,
    boxes: list[BoxType],
    font_size: int = 30,
) -> None:
    draw = ImageDraw.Draw(image)
    for idx, box in enumerate(boxes):
        color = tuple(np.random.randint(64, 191, size=3).tolist())
        font = ImageFont.load_default(font_size)
        center = _calculate_center(box)

        draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)

        if hasattr(font, "getbbox"):
            _, _, w, h = draw.textbbox((0, 0), str(idx), font)
        else:
            w, h = draw.textsize(str(idx), font)
        if box[0] >= w:
            bbox = (
                round(box[0] - w),
                round(center[1] - h / 2),
                round(box[0]),
                round(center[1] + h / 2),
            )
        else:
            bbox = (
                round(box[2]),
                round(center[1] - h / 2),
                round(box[2] + w),
                round(center[1] + h / 2),
            )

        draw.rectangle(bbox, fill=color)
        draw.text((bbox[0], bbox[1]), str(idx), fill="white", font=font)


@cache
def _get_grounding_dino_model(
    type: Literal["tiny", "base"] = "tiny",
) -> tuple[GroundingDinoProcessor, GroundingDinoForObjectDetection]:
    """Get the grounding dino model.

    Args:
        type: The version of the Gounding Dino Model.

    Returns:
        A tuple (processor, model).
    """
    model_name = f"IDEA-Research/grounding-dino-{type}"
    processor = AutoProcessor.from_pretrained(model_name)
    model = GroundingDinoForObjectDetection.from_pretrained(model_name).to(device)
    return processor, model


@cache
def _get_easyocr_model() -> easyocr.Reader:
    return easyocr.Reader(["en"])


def get_groundingdino_boxes(
    images: Image.Image | list[Image.Image],
    text_prompt: str,
    box_threshold: float = 0.05,
    text_threshold: float = 0.5,
) -> list[list[AnnotatedBoxType]]:
    """Get the bounding boxes of the objects in the image using GroundingDino.

    Args:
        images: The image or list of images.
        text_prompt: The text prompt to use for all the images.
        box_threshold: The box threshold.
        text_threshold: The text threshold.

    Returns:
        The first level list is for each image, and the second level list contains
        tuples (detected boxes, its sementical representation) as the result of the
        image.
    """
    processor, model = _get_grounding_dino_model()
    if isinstance(images, Image.Image):
        images = [images]
    image_number = len(images)
    images = [image.convert("RGB") for image in images]
    inputs = processor(
        images=images,
        text=[text_prompt] * image_number,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [image.size[::-1] for image in images]
    detection_results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )
    final_output = []
    for result in detection_results:
        boxes = result["boxes"].cpu().int().tolist()
        labels = result["labels"]
        final_output.append(list(zip(boxes, labels)))
    return final_output


def get_easyocr_boxes(
    image: Image.Image,
) -> list[AnnotatedBoxType]:
    """Get the bounding boxes of the text in the image using EasyOCR.

    Args:
        image: The taget image.

    Returns:
        The list of tuple of bounding boxes and their corresponding text.
    """
    reader = _get_easyocr_model()
    result = reader.readtext(np.array(image), text_threshold=0.9)
    boxes = []
    for detect in result:
        boxes.append(
            (
                (
                    detect[0][0][0],
                    detect[0][0][1],
                    detect[0][2][0],
                    detect[0][2][1],
                ),
                detect[1],
            )
        )
    return boxes


@action(local=True)
def groundingdino_easyocr(
    input_base64_image: str,
    font_size: int,
    env,
) -> tuple[str, list[AnnotatedBoxType]]:
    """Get the interative elements in the image.

    Using GroundingDino and EasyOCR to detect the interactive elements in the image.
    Mark the detected elements with bounding boxes and labels. Store the labels and
    boxes in the environment to be used in other actions.

    Args:
        input_base64_image: The base64 encoded image.
        font_size: The font size of the label.

    Returns:
        A tuple (base64_image, boxes), where base64_image is the base64 encoded image
        drawn with bounding boxes and labels, and box is the list of detected boxes and
        labels.
    """
    check_transformers_import()
    image = base64_to_image(input_base64_image)
    od_boxes = get_groundingdino_boxes(image, "icon . logo .", box_threshold=0.02)[0]
    od_boxes = _filter_boxes_by_iou(od_boxes, iou_threshold=0.5)
    ocr_boxes = get_easyocr_boxes(image)
    boxes_with_label = ocr_boxes + od_boxes
    filtered_boxes = _remove_invalid_boxes(boxes_with_label, image.width, image.height)
    filtered_boxes = _filter_boxes_by_overlap(filtered_boxes)
    center_dis = round(max(image.height, image.width) / 80.0)
    filtered_boxes = _filter_boxes_by_center(filtered_boxes, center_dis)
    env.element_label_map = [box[1] for box in filtered_boxes]
    result_boxes = [box[0] for box in filtered_boxes]
    _draw_boxes(image, result_boxes, font_size)
    env.element_position_map = result_boxes
    env.ocr_results = "".join([box[1] for box in ocr_boxes])
    return image_to_base64(image), filtered_boxes


@action(local=True)
def get_elements_prompt(
    input: tuple[str, list[AnnotatedBoxType]], env
) -> tuple[str, str]:
    """Get the text prompt passing to the agent for the image.

    Args:
        input: The base64 encoded image and the list of detected boxes and labels.

    Returns:
        A tuple (image, prompt) contains the base64 encoded image and the prompt.
    """
    image, boxes = input
    labels = ""
    for id, box in enumerate(boxes):
        if box[1] is not None:
            labels += f"[{id}|{box[1]}]\n"
    prompt = (
        "Some elements in the current screenshot have labels. I will give you "
        "these labels by [id|label].\n" + labels
    )
    return image, prompt

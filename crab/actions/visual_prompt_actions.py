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
from functools import cache

try:
    import easyocr
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except ImportError:
    pass

from crab import action
from crab.utils.common import base64_to_image, image_to_base64

device = "cuda" if torch.cuda.is_available() else "cpu"

BoxType = tuple[int, int, int, int]


def calculate_iou(box1, box2):
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


def calculate_size(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_center(box) -> tuple[int, int]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def get_grounding_output(
    model, processor, image, caption, box_threshold, text_threshold, with_logits=True
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold,
        text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    logits = results["scores"].cpu()  # (nq, 256)  # TODO: 要不要sigmoid
    boxes = results["boxes"].cpu()  # (nq, 4)
    labels = results["labels"]

    pred_phrases = []
    scores = []
    for logit, box, label in zip(logits, boxes, labels):
        if with_logits:
            pred_phrases.append(label + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(label)
        scores.append(logit.item())

    return boxes, torch.Tensor(scores), pred_phrases


def remove_boxes(boxes_filt, size, iou_threshold=0.5):
    boxes_to_remove = set()

    for i in range(len(boxes_filt)):
        if calculate_size(boxes_filt[i]) > 0.05 * size[0] * size[1]:
            boxes_to_remove.add(i)
        for j in range(len(boxes_filt)):
            if calculate_size(boxes_filt[j]) > 0.05 * size[0] * size[1]:
                boxes_to_remove.add(j)
            if i == j:
                continue
            if i in boxes_to_remove or j in boxes_to_remove:
                continue
            iou = calculate_iou(boxes_filt[i], boxes_filt[j])
            if iou >= iou_threshold:
                boxes_to_remove.add(j)

    boxes_filt = [
        box for idx, box in enumerate(boxes_filt) if idx not in boxes_to_remove
    ]

    return boxes_filt


def remove_invalid_boxes(boxes_with_label, width, height):
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


def box_a_in_b(a: BoxType, b: BoxType):
    return a[0] > b[0] and a[1] > b[1] and a[2] < b[2] and a[3] < b[3]


def filter_boxes_by_overlap(boxes_with_label):
    boxes = [box[0] for box in boxes_with_label]
    boxes_to_remove = set()
    for i in range(len(boxes)):
        if i in boxes_to_remove:
            continue
        for j in range(len(boxes)):
            if box_a_in_b(boxes[i], boxes[j]):
                boxes_to_remove.add(j)

    boxes_filt = [
        box for idx, box in enumerate(boxes_with_label) if idx not in boxes_to_remove
    ]
    return boxes_filt


@cache
def get_model():
    # Define the model identifier
    model_id = "IDEA-Research/grounding-dino-tiny"

    # Load the processor and model from Hugging Face Transformers
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    model.eval()

    return model, processor


def draw_boxes(
    image: Image.Image,
    boxes: list[BoxType],
    font_size: int = 30,
) -> None:
    draw = ImageDraw.Draw(image)
    for idx, box in enumerate(boxes):
        color = tuple(np.random.randint(64, 191, size=3).tolist())
        font = ImageFont.load_default(font_size)
        center = calculate_center(box)

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


def get_groundingdino_boxes(
    input_image,
    text_prompt,
    groundingdino_model,
    groundingdino_processor,
    box_threshold=0.05,
    text_threshold=0.5,
) -> list[BoxType]:
    image = input_image
    size = image.size

    image_pil = image.convert("RGB")

    boxes_filt, _, _ = get_grounding_output(
        groundingdino_model,
        groundingdino_processor,
        image_pil,
        text_prompt,
        box_threshold,
        text_threshold,
    )

    boxes_filt = boxes_filt.cpu().int().tolist()
    filtered_boxes = remove_boxes(boxes_filt, size)

    return [(box, None) for box in filtered_boxes]


@cache
def get_easyocr_model():
    return easyocr.Reader(["en"])


def get_easyocr_boxes(
    image: Image.Image,
) -> tuple[list[BoxType], list[str]]:
    reader = get_easyocr_model()
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
) -> tuple[str, list[tuple[BoxType, str]]]:
    image = base64_to_image(input_base64_image)
    groundingdino_model, groundingdino_processor = get_model()
    od_boxes = get_groundingdino_boxes(
        image,
        "icon . logo .",
        groundingdino_model,
        groundingdino_processor,
        box_threshold=0.02,
    )
    ocr_boxes = get_easyocr_boxes(image)
    boxes_with_label = ocr_boxes + od_boxes
    filtered_boxes = remove_invalid_boxes(boxes_with_label, image.width, image.height)
    filtered_boxes = filter_boxes_by_overlap(filtered_boxes)
    result_boxes = [box[0] for box in filtered_boxes]
    draw_boxes(image, result_boxes, font_size)
    env.element_position_map = [
        (
            box[0] / image.width,
            box[1] / image.height,
            box[2] / image.width,
            box[3] / image.height,
        )
        for box in result_boxes
    ]
    env.ocr_results = "".join([box[1] for box in ocr_boxes])
    return image_to_base64(image), filtered_boxes


@action(local=True)
def get_elements_prompt(input: tuple[str, list[tuple[BoxType, str]]], env):
    image, boxes = input
    labels = ""
    for id, box in enumerate(boxes):
        if box[1] is not None:
            labels += f"[{id}|{box[1]}]\n"
    prompt = (
        "Some elements in the current screenshot have labels. I will give you these "
        "labels by [id|label].\n" + labels
    )
    return image, prompt


def get_element_position(element_id, env):
    """Get element position provided by function `zs_object_detection`"""
    box = env.element_position_map[element_id]
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return round(x * env.width), round(y * env.height)

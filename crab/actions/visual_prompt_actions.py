import os
from functools import cache
from urllib.request import urlretrieve

from PIL import Image, ImageDraw, ImageFont

from crab import action
from crab.utils.common import base64_to_image, image_to_base64

try:
    import cv2
    import easyocr
    import numpy as np
    import PIL
    import pytesseract
    import torch

    from thirdparty.groundingdino.datasets import transforms as T
    from thirdparty.groundingdino.models import build_model
    from thirdparty.groundingdino.util.slconfig import SLConfig
    from thirdparty.groundingdino.util.utils import (
        clean_state_dict,
        get_phrases_from_posmap,
    )
except ImportError:
    pass

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
    model, image, caption, box_threshold, text_threshold, with_logits=True
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


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


def filter_boxes_by_center(boxes_with_label, center_dis_thresh):
    boxes = [box[0] for box in boxes_with_label]
    boxes_to_remove = set()
    for i in range(len(boxes)):
        if i in boxes_to_remove:
            continue
        center_i = calculate_center(boxes[i])
        for j in range(i + 1, len(boxes)):
            center_j = calculate_center(boxes[j])
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


def transform_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def download_weights(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    def progress_hook(count, block_size, total_size):
        downloaded = count * block_size
        percent = int(downloaded * 100 / total_size)
        print(
            f"\rDownloading: {percent}% [{downloaded}/{total_size} bytes]",
            end="",
            flush=True,
        )

    print("Downloading...")
    urlretrieve(url, destination, reporthook=progress_hook)
    print("\nDownload completed.")


def load_model(model_config_path, ckpt_filenmae, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(ckpt_filenmae, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


@cache
def get_model():
    """
    TODO: Use Hugging Face Transformers
    from transformers import AutoProcessor, AutoModelForObjectDetection

    def get_model():
        # Define the model identifier
        model_id = "IDEA-Research/grounding-dino-tiny"

        # Load the processor and model from Hugging Face Transformers
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForObjectDetection.from_pretrained(model_id).to(device)

        return model
    """

    WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = "thirdparty/groundingdino/weights/groundingdino_swint_ogc.pth"
    MODEL_CONFIG_PATH = "thirdparty/groundingdino/config/GroundingDINO_SwinT_OGC.py"

    if not os.path.isfile(WEIGHTS_PATH):
        download_weights(WEIGHTS_URL, WEIGHTS_PATH)

    # Load model configuration
    args = SLConfig.fromfile(MODEL_CONFIG_PATH)
    args.device = "cuda"

    # Build model
    model = build_model(args)

    # Load weights
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

    # Set model to evaluation mode
    _ = model.eval()

    return model


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
        # bbox = (
        #     round(center[0] - w / 2),
        #     round(center[1] - h / 2),
        #     round(center[0] + w / 2),
        #     round(center[1] + h / 2),
        # )
        # if box[1] >= h:
        #     bbox = (
        #         round(box[0]),
        #         round(box[1] - h),
        #         round(box[0] + w),
        #         round(box[1]),
        #     )
        # else:
        #     bbox = (
        #         round(box[0]),
        #         round(box[3]),
        #         round(box[0] + w),
        #         round(box[3] + h),
        #     )
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


def detect(
    input_image,
    text_prompt,
    groundingdino_model,
    box_threshold=0.05,
    text_threshold=0.5,
    font_size=30,
    center_dis_threshold=20,
) -> tuple[Image.Image, list[BoxType]]:
    """
    Detects interactive components in the given image based on the specified text
    prompt.

    Args:
        input_image (Image.Image): The input image to analyze.
        text_prompt (str): The text prompt used for detection.
        groundingdino_model: The groundingdino model used for detection.
        box_threshold (float, optional): The box detection threshold. Defaults to 0.05.
        text_threshold (float, optional): The text detection threshold. Defaults to 0.5.

    Returns:
        tuple[Image.Image, list[tuple[int]]]: A tuple containing the marked image and a
            list of filtered bounding boxes.
    """
    image = input_image
    size = image.size

    image_pil = image.convert("RGB")
    image = np.array(image_pil)

    transformed_image = transform_image(image_pil)
    draw = ImageDraw.Draw(image_pil)

    boxes_filt, _, _ = get_grounding_output(
        groundingdino_model,
        transformed_image,
        text_prompt,
        box_threshold,
        text_threshold,
    )

    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu().int().tolist()
    filtered_boxes = remove_boxes(boxes_filt, size)
    filtered_boxes = filter_boxes_by_center(filtered_boxes, center_dis_threshold)
    for idx, box in enumerate(filtered_boxes):
        color = tuple(np.random.randint(64, 191, size=3).tolist())
        font = ImageFont.load_default(font_size)
        center = calculate_center(box)

        if hasattr(font, "getbbox"):
            _, _, w, h = draw.textbbox((0, 0), str(idx), font)
        else:
            w, h = draw.textsize(str(idx), font)
        bbox = (
            round(center[0] - w / 2),
            round(center[1] - h / 2),
            round(center[0] + w / 2),
            round(center[1] + h / 2),
        )
        draw.rectangle(bbox, fill=color)
        draw.text((bbox[0], bbox[1]), str(idx), fill="white", font=font)

    return image_pil, filtered_boxes


@action(local=True)
def zs_object_detect(
    input_base64_image: str,
    target: str,
    env,
) -> tuple[str, list[BoxType]]:
    """
    Detects interactive components in the given image based on the specified text
    prompt.

    Args:
        input_base64_image: The input image in base64 format to analyze.
        target: The text prompt used for detection.

    Returns:
        A tuple containing the marked image and a list of filtered bounding boxes.
    """
    image = base64_to_image(input_base64_image)
    font_size = round(max(image.height, image.width) / 60.0)
    center_dis = round(max(image.height, image.width) / 80.0)
    groundingdino_model = get_model().eval()
    marked_image, boxes = detect(
        image,
        target,
        groundingdino_model,
        font_size=font_size,
        center_dis_threshold=center_dis,
    )
    env.element_position_map = boxes
    return image_to_base64(marked_image), boxes


def get_groundingdino_boxes(
    input_image,
    text_prompt,
    groundingdino_model,
    box_threshold=0.05,
    text_threshold=0.5,
) -> list[BoxType]:
    image = input_image
    size = image.size

    image_pil = image.convert("RGB")
    image = np.array(image_pil)

    transformed_image = transform_image(image_pil)

    boxes_filt, _, _ = get_grounding_output(
        groundingdino_model,
        transformed_image,
        text_prompt,
        box_threshold,
        text_threshold,
    )

    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

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
    center_dis = round(max(image.height, image.width) / 80.0)
    groundingdino_model = get_model().eval()
    od_boxes = get_groundingdino_boxes(
        image, "icon . logo .", groundingdino_model, box_threshold=0.02
    )
    ocr_boxes = get_easyocr_boxes(image)
    boxes_with_label = ocr_boxes + od_boxes
    filtered_boxes = remove_invalid_boxes(boxes_with_label, image.width, image.height)
    filtered_boxes = filter_boxes_by_overlap(filtered_boxes)
    filtered_boxes = filter_boxes_by_center(filtered_boxes, center_dis)
    env.element_label_map = [box[1] for box in filtered_boxes]
    result_boxes = [box[0] for box in filtered_boxes]
    draw_boxes(image, result_boxes, font_size)
    env.element_position_map = result_boxes
    env.ocr_results = "".join([box[1] for box in ocr_boxes])
    return image_to_base64(image), filtered_boxes


def groundingdino2x2_output(
    input_base64_image: str,
    box_threshold: float = 0.02,
    text_prompt: str = "icon . logo .",
) -> tuple[str, list[tuple[BoxType, str]]]:
    image = base64_to_image(input_base64_image)

    font_size = 18
    center_dis = round(max(image.height, image.width) / 80.0)
    groundingdino_model = get_model().eval()

    # split the image into 2x2
    new_image_list = []
    width, height = image.size
    new_width = width // 2
    new_height = height // 2

    new_image_list.append([image.crop((0, 0, new_width, new_height)), 0, 0])
    new_image_list.append([image.crop((new_width, 0, width, new_height)), new_width, 0])
    new_image_list.append(
        [image.crop((0, new_height, new_width, height)), 0, new_height]
    )
    new_image_list.append(
        [image.crop((new_width, new_height, width, height)), new_width, new_height]
    )
    new_image_list.append([image, 0, 0])

    od_boxes = []
    for image_now, base_width, base_heiht in new_image_list:
        od_boxes_now = get_groundingdino_boxes(
            image_now,
            text_prompt,
            groundingdino_model,
            box_threshold=box_threshold,
        )
        for item in od_boxes_now:
            item_new = (
                (
                    item[0][0] + base_width,
                    item[0][1] + base_heiht,
                    item[0][2] + base_width,
                    item[0][3] + base_heiht,
                ),
                item[1],
            )
            od_boxes.append(item_new)

    boxes_with_label = od_boxes
    filtered_boxes = filter_boxes_by_center(boxes_with_label, center_dis)
    result_boxes = [box[0] for box in filtered_boxes]
    draw_boxes(image, result_boxes, font_size)
    return image_to_base64(image), filtered_boxes


@action(local=True)
def get_elements_prompt(input: tuple[str, list[tuple[BoxType, str]]], env):
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


def get_element_position(element_id, env):
    """Get element position provided by function `zs_object_detection`"""
    box = env.element_position_map[element_id]
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return round(x), round(y)


def find_text_in_image(
    image: Image.Image, target_text: str, debug=False
) -> tuple[Image.Image, list[tuple[float, float]]]:
    """
    Detects target text in a given image, draw rectangle on it and calculate center
    points

    Args:
        image: The input image.
        target_text: The text we wish to search for.

    Returns:
        A tuple containing the visual prompt and a list of corresponding box center
        points.
    """
    # convert PIL to numpy array
    image_array = np.array(image)
    image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_array_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # get tesseract ocr output
    ocr_output = pytesseract.image_to_data(
        image_array_gray, output_type=pytesseract.Output.DICT
    )
    ocr_output = list(
        zip(
            ocr_output["left"],
            ocr_output["top"],
            ocr_output["height"],
            ocr_output["width"],
            ocr_output["text"],
        )
    )

    # iterate over each box
    # draw rectangle on target text and calculate its center point
    center_points = []
    text_id = 0
    epsilon = 5
    for left, top, height, width, text in ocr_output:
        # draw every box
        if debug:
            cv2.rectangle(
                image_array_rgb,
                (left - epsilon, top - epsilon),
                (left + width + epsilon, top + height + epsilon),
                (255, 255, 0),
                thickness=3,
            )
            cv2.putText(
                image_array_rgb,
                text,
                (left, top - epsilon * 2),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255, 255, 0),
                thickness=3,
            )
        # check whether including text
        index_start = text.lower().find(target_text.lower())
        if index_start == -1:
            pass
        else:
            percent_start = index_start / len(str(text))
            percent_end = (index_start + len(target_text)) / len(str(text))
            adjusted_left = int(left + width * percent_start)
            adjusted_right = int(left + width * percent_end)
            cv2.rectangle(
                image_array_rgb,
                (adjusted_left - epsilon, top - epsilon),
                (adjusted_right + epsilon, top + height + epsilon),
                (255, 255, 0),
                thickness=3,
            )
            t = str(text_id)
            cv2.putText(
                image_array_rgb,
                t,
                (adjusted_left, top - epsilon * 2),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255, 255, 0),
                thickness=3,
            )
            text_id += 1

            center_points.append(
                ((adjusted_left + adjusted_right) / 2, (top + top + height) / 2)
            )

    output_image = PIL.Image.fromarray(image_array_rgb)

    return output_image, center_points

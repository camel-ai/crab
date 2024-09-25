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
# ruff: noqa: E501
import base64
import hashlib
import io
import os
import re
import subprocess
import time
from collections import Counter
from functools import cache
from typing import Callable, List, Optional, Tuple

import cv2
import easyocr
import imageio as imio
import networkx as nx
import numpy as np
import psutil
import pyperclip
import requests
import torch
from networkx import DiGraph, path_graph
from numpy.linalg import norm
from PIL import Image

from crab import SubTask, TaskGenerator, action, evaluator
from crab.actions.crab_actions import check_submit, submit


class ImageMatcher:
    """
    A class to handle image matching, resizing, and cropping operations using accelerated feature matching.
    See https://github.com/verlab/accelerated_features.
    """

    def __init__(self, top_k: int = 4096):
        """
        Initializes the ImageMatcher with a pretrained XFeat model.

        Parameters:
        top_k (int): The number of top features to use for matching.
        """
        self.xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k
        )
        self.top_k = top_k

    def warp_corners_and_draw_matches(
        self,
        ref_points: np.ndarray,
        dst_points: np.ndarray,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the homography matrix and warps the corners of the first image to the second image space.

        Parameters:
        ref_points (np.ndarray): Reference points from the first image.
        dst_points (np.ndarray): Destination points from the second image.
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Image with warped corners and the warped corners coordinates.
        """
        H, mask = cv2.findHomography(
            ref_points,
            dst_points,
            cv2.USAC_MAGSAC,
            3.5,
            maxIters=1000,
            confidence=0.999,
        )
        mask = mask.flatten()

        h, w = img1.shape[:2]
        corners_img1 = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners_img1, H)

        img2_with_corners = img2.copy()
        for i in range(len(warped_corners)):
            start_point = tuple(warped_corners[i - 1][0].astype(int))
            end_point = tuple(warped_corners[i][0].astype(int))
            cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)

        keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
        keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

        img_matches = cv2.drawMatches(
            img1,
            keypoints1,
            img2_with_corners,
            keypoints2,
            matches,
            None,
            matchColor=(0, 255, 0),
            flags=2,
        )

        return img_matches, warped_corners

    def _get_bounding_box(
        self, warped_corners: np.ndarray, img_shape: Tuple[int, int]
    ) -> List[int]:
        """
        Computes the bounding box around the warped corners.

        Parameters:
        warped_corners (np.ndarray): The warped corners coordinates.
        img_shape (Tuple[int, int]): The shape of the image as (height, width).

        Returns:
        List[int]: Bounding box coordinates [x_min, x_max, y_min, y_max].
        """
        h, w = img_shape

        x_min = np.min(warped_corners[:, 0, 0])
        x_max = np.max(warped_corners[:, 0, 0])
        y_min = np.min(warped_corners[:, 0, 1])
        y_max = np.max(warped_corners[:, 0, 1])

        x_min = max(0, x_min)
        x_max = min(w - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(h - 1, y_max)

        return [int(x_min), int(x_max), int(y_min), int(y_max)]

    def _resize_image(
        self, img1: np.ndarray, img2: np.ndarray, scale: float, match_dimension: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resizes img1 to match a scaled dimension of img2.

        Parameters:
        img1 (np.ndarray): The first image to be resized.
        img2 (np.ndarray): The reference image.
        scale (float): The scale factor (0.5 for half size).
        match_dimension (str): The dimension to match ('height' or 'width').

        Returns:
        Tuple[np.ndarray, np.ndarray]: Resized img1 and original img2.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if match_dimension == "height":
            new_height = int(h2 * scale)
            new_width = int(w1 * (new_height / h1))
        elif match_dimension == "width":
            new_width = int(w2 * scale)
            new_height = int(h1 * (new_width / w1))
        else:
            raise ValueError("match_dimension must be either 'height' or 'width'.")

        resized_img1 = cv2.resize(img1, (new_width, new_height))
        return resized_img1, img2

    def get_resizing_functions(
        self,
    ) -> List[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        Provides a list of resizing functions.

        Returns:
        List[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]: List of resizing functions.
        """
        return [
            lambda x, y: (x, y),
            lambda x, y: self._resize_image(x, y, 1.0, "height"),
            lambda x, y: self._resize_image(x, y, 1.0, "width"),
            lambda x, y: self._resize_image(x, y, 0.5, "height"),
            lambda x, y: self._resize_image(x, y, 0.5, "width"),
        ]

    def match_images(
        self,
        im1_path: str,
        im2_path: str,
        top_k: int = 4096,
        match_num_threshold: int = 80,
    ) -> Tuple[Optional[List[int]], Optional[np.ndarray], int]:
        """
        Matches two images and finds the bounding box around the matched area if sufficient matches are found.

        Parameters:
        im1_path (str): Path to the first image.
        im2_path (str): Path to the second image.
        top_k (int): The number of top features to use for matching.
        match_num_threshold (int): The minimum number of matches required to consider the match valid.

        Returns:
        Tuple[Optional[List[int]], Optional[np.ndarray], int]: Bounding box, image with matched keypoints drawn, and the number of matches found.
        """
        im1 = self.load_and_convert_image(im1_path)
        im2 = self.load_and_convert_image(im2_path)

        best_matches = {
            "count": 0,
            "im1_resized": None,
            "im2_resized": None,
            "mkpts_0": None,
            "mkpts_1": None,
        }

        for resize_func in self.get_resizing_functions():
            try:
                im1_resized, im2_resized = resize_func(im1, im2)
                mkpts_0, mkpts_1 = self.xfeat.match_xfeat_star(
                    im1_resized, im2_resized, top_k=top_k
                )

                if len(mkpts_0) > best_matches["count"]:
                    best_matches.update(
                        {
                            "count": len(mkpts_0),
                            "im1_resized": im1_resized,
                            "im2_resized": im2_resized,
                            "mkpts_0": mkpts_0,
                            "mkpts_1": mkpts_1,
                        }
                    )
            except Exception:
                continue

        if best_matches["count"] >= match_num_threshold:
            canvas, warped_corners = self.warp_corners_and_draw_matches(
                best_matches["mkpts_0"],
                best_matches["mkpts_1"],
                best_matches["im1_resized"],
                best_matches["im2_resized"],
            )
            bbox = self._get_bounding_box(warped_corners, im2_resized.shape[:2])
        else:
            bbox, canvas = None, None

        return bbox, canvas, best_matches["count"]

    def load_and_convert_image(self, filepath: str) -> np.ndarray:
        """
        Loads an image from a file and converts it to JPG format if necessary.

        Parameters:
        filepath (str): The path to the image file.

        Returns:
        np.ndarray: The loaded and converted image.
        """
        image = Image.open(filepath)
        if image.mode != "RGB":
            image = image.convert("RGB")
        with io.BytesIO() as output:
            image.save(output, format="JPEG")
            converted_image = np.copy(imio.v2.imread(output)[..., ::-1])
        return converted_image


image_matcher = ImageMatcher()


def from_env_load_and_save_file(env, file_path, output_dir="/tmp/local_save"):
    """
    Load a file, convert it to bytes, and save it to a local directory with the same basename.

    Args:
        env: The environment object with the _action_endpoint method.
        file_path (str): The path to the file to be loaded.
        output_dir (str): The directory where the file should be saved (default is "/tmp/local_save").

    Returns:
        str: The path to the saved file.
    """

    @action(env_name="ubuntu")
    def get_encoded_file(file_path: str) -> bytes | None:
        try:
            with open(file_path, "rb") as file:
                file_bytes = file.read()
                encoded_string = base64.b64encode(file_bytes).decode("utf-8")
        except Exception:
            return None

        return encoded_string

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the file and convert to bytes
    encoded_string = env._action_endpoint(get_encoded_file, {"file_path": file_path})

    # Decode the Base64 string back to bytes
    decoded_bytes = base64.b64decode(encoded_string.encode("utf-8"))

    # Create the output file path
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, file_name)

    # Save the decoded bytes to the output path
    with open(output_file_path, "wb") as file:
        file.write(decoded_bytes)

    return output_file_path


def crop_image(img: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Crops the image based on the bounding box coordinates.

    Parameters:
    img (np.ndarray): The input image.
    bbox (List[int]): Bounding box coordinates [x_min, x_max, y_min, y_max].

    Returns:
    np.ndarray: The cropped image.
    """
    x_min, x_max, y_min, y_max = bbox
    return img[y_min:y_max, x_min:x_max]


def calculate_bbox_center(bbox: List[int]) -> Tuple[int, int]:
    """
    Calculates the center of a bounding box.

    Parameters:
    bbox (List[int]): The bounding box coordinates [x_min, x_max, y_min, y_max].

    Returns:
    Tuple[int, int]: The center coordinates (x, y).
    """
    x_min, x_max, y_min, y_max = bbox
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    return x_center, y_center


def is_bbox_in_direction(bbox_1: List[int], bbox_2: List[int], direction: str) -> bool:
    """
    Check if the center of bbox_1 is in the specified direction relative to the center of bbox_2.

    Args:
        bbox_1 (List[int]): The bounding box coordinates [x_min, x_max, y_min, y_max] of the first bounding box.
        bbox_2 (List[int]): The bounding box coordinates [x_min, x_max, y_min, y_max] of the second bounding box.
        direction (str): The direction to check ("left", "right", "above", "below").

    Returns:
        bool: True if the center of bbox_1 is in the specified direction relative to bbox_2, False otherwise.
    """

    center_1 = calculate_bbox_center(bbox_1)
    center_2 = calculate_bbox_center(bbox_2)

    if direction == "left":
        return center_1[0] < center_2[0]
    elif direction == "right":
        return center_1[0] > center_2[0]
    elif direction == "above":
        return center_1[1] < center_2[1]
    elif direction == "below":
        return center_1[1] > center_2[1]
    else:
        raise ValueError("Invalid direction. Use 'left', 'right', 'above', or 'below'.")


def ocr_text_matching(
    image_path: str, text: str
) -> Optional[Tuple[List[int], str, float]]:
    """
    Performs OCR on an image to find a specific text string and returns the bounding box, matched text, and confidence level.

    Parameters:
    image_path (str): The path to the image file.
    text (str): The text string to search for in the image.

    Returns:
    Optional[Tuple[List[int], str, float]]: The bounding box coordinates [x_min, y_min, x_max, y_max], the matched text, and the confidence level if found, otherwise None.
    """
    reader = easyocr.Reader(["en"])
    result = reader.readtext(image_path)

    for entry in result:
        bbox, detected_text, confidence = entry
        if text in detected_text:
            # Extract the bounding box coordinates
            x_min = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
            x_max = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
            y_min = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            y_max = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            return (
                [int(x_min), int(x_max), int(y_min), int(y_max)],
                detected_text,
                confidence,
            )

    return None


def convert_file_to_images(file_path: str) -> List[str]:
    """
    Convert a file to JPG images using LibreOffice and return the list of image file paths.

    Args:
        file_path (str): The path to the file.

    Returns:
        List[str]: List of paths to the generated image files.
    """
    output_format = "jpg"
    output_dir = "/tmp/converted_images"
    os.makedirs(output_dir, exist_ok=True)

    # Run LibreOffice conversion command
    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            output_format,
            "--outdir",
            output_dir,
            file_path,
        ],
        capture_output=True,
        text=True,
    )

    # Check if the conversion was successful
    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr}")

    # Collect the generated image file paths
    image_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(f".{output_format}")
    ]

    # Verify if the files were successfully saved
    if not image_files:
        raise FileNotFoundError(
            f"No {output_format} files found in the output directory"
        )

    # Get the basename of the original file (without extension)
    file_basename = os.path.splitext(os.path.basename(file_path))[0]

    # Check if any of the images match the basename of the original file
    matching_images = [f for f in image_files if file_basename in os.path.basename(f)]
    if not matching_images:
        raise FileNotFoundError(
            f"No images found with basename matching the original file: {file_basename}"
        )

    return matching_images


def cleanup_files(files: List[str]):
    """
    Delete the list of files.

    Args:
        files (List[str]): List of paths to the files to be deleted.
    """
    for file in files:
        os.remove(file)


def is_valid_url(url):
    # Regular expression to check if the string is a valid HTTP/HTTPS URL
    url_pattern = re.compile(
        r"^(https?://)"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return bool(re.match(url_pattern, url))


def is_valid_image_data_uri(uri):
    # Regular expression to check if the string is a valid Data URI for image formats
    data_uri_pattern = re.compile(
        r"^data:image/(png|jpeg|gif|svg\+xml|bmp|webp);base64,[A-Za-z0-9+/]+={0,2}$",
        re.IGNORECASE,
    )
    return bool(re.match(data_uri_pattern, uri))


def is_github_repo_url(url):
    # Regular expression to check if the URL is a GitHub repository URL
    github_repo_pattern = re.compile(
        r"^https?://"  # Protocol
        r"github\.com/"  # Domain
        r"[^/]+/"  # Username
        r"[^/]+/?$",  # Repository name, optional trailing slash
        re.IGNORECASE,
    )
    return bool(re.match(github_repo_pattern, url))


def get_rgb_values_outside_bbox(
    img: np.ndarray, bbox: List[int], margin: int = 10
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Reads the pixel color RGB values outside of the bounding box with an additional margin and finds the most frequent RGB value.

    Parameters:
    img (np.ndarray): The input image.
    bbox (List[int]): Bounding box coordinates [x_min, x_max, y_min, y_max].
    margin (int): The margin to add outside the bounding box. Default is 10.

    Returns:
    Tuple[np.ndarray, Tuple[int, int, int]]: The RGB values outside the bounding box with the margin and the most frequent RGB value.
    """
    x_min, x_max, y_min, y_max = bbox

    # Ensure the coordinates with margin are within image dimensions
    x_min_with_margin = max(0, x_min - margin)
    x_max_with_margin = min(img.shape[1], x_max + margin)
    y_min_with_margin = max(0, y_min - margin)
    y_max_with_margin = min(img.shape[0], y_max + margin)

    # Create a mask for the bounding box area with margin
    mask = np.ones(img.shape[:2], dtype=bool)
    mask[y_min_with_margin:y_max_with_margin, x_min_with_margin:x_max_with_margin] = (
        False
    )

    # Extract the RGB values outside the bounding box with margin
    rgb_values = img[mask]

    # Find the most frequent RGB value
    rgb_values_tuple = [tuple(rgb) for rgb in rgb_values]
    most_common_rgb = Counter(rgb_values_tuple).most_common(1)[0][0]

    return list(most_common_rgb)[::-1]


def contains_required_strings(clipboard_content: str, required_strings: list) -> bool:
    """
    Check if all required strings are present in the clipboard content.

    Args:
        clipboard_content (str): The content from the clipboard.
        required_strings (list): A list of required strings to check.

    Returns:
        bool: True if all required strings are found in the clipboard content, False otherwise.
    """
    for string in required_strings:
        if string not in clipboard_content:
            return False
    return True


@evaluator(env_name="ubuntu")
def verify_file_content_with_clipboard(file_path: str) -> bool:
    """
    Verify that the content of the file matches the clipboard content line by line.

    Args:
        file_path (str): The path to the file to verify.

    Returns:
        bool: True if the file content matches the clipboard content, False otherwise.
    """

    def verify_content_with_clipboard(file_content: str) -> bool:
        """
        Verify that the provided file content matches the clipboard content line by line.

        Args:
            file_content (str): The content of the file to verify.

        Returns:
            bool: True if the file content matches the clipboard content, False otherwise.
        """
        clipboard_content = pyperclip.paste()
        clipboard_lines = clipboard_content.split("\n")
        file_lines = file_content.split("\n")

        # Check if each line from the clipboard content is in the corresponding line in the file content
        for clipboard_line, file_line in zip(clipboard_lines, file_lines):
            if clipboard_line not in file_line:
                return False

        return True

    with open(file_path, "r") as file:
        file_content = file.read()

    return verify_content_with_clipboard(file_content)


@evaluator(env_name="ubuntu")
def verify_odt_file_content_with_clipboard(file_path: str) -> bool:
    """
    Verify that the content of the ODT file matches the clipboard content.

    Args:
        file_path (str): The path to the ODT file to verify.

    Returns:
        bool: True if the ODT file content matches the clipboard content, False otherwise.
    """
    from odf import teletype, text
    from odf.opendocument import load

    def verify_content_with_clipboard(file_content: str) -> bool:
        """
        Verify that the provided file content matches the clipboard content line by line.

        Args:
            file_content (str): The content of the file to verify.

        Returns:
            bool: True if the file content matches the clipboard content, False otherwise.
        """
        clipboard_content = pyperclip.paste()
        clipboard_lines = clipboard_content.split("\n")
        file_lines = file_content.split("\n")

        # Check if each line from the clipboard content is in the corresponding line in the file content
        for clipboard_line, file_line in zip(clipboard_lines, file_lines):
            if clipboard_line not in file_line:
                return False

        return True

    textdoc = load(file_path)
    allparas = textdoc.getElementsByType(text.P)
    odt_content = "\n".join([teletype.extractText(p) for p in allparas])

    return verify_content_with_clipboard(odt_content)


@evaluator(env_name="ubuntu", local=True)
def verify_combined_image(
    image_path_1: str, image_path_2: str, file_path: str, direction: str, env
) -> bool:
    """
    Check if the combined file contains both input images without overlay and in the specified direction.

    Args:
        image_path_1 (str): Path to the first image.
        image_path_2 (str): Path to the second image.
        file_path (str): Path to the combined file.
        direction (str): The direction to check ("left", "right", "above", "below").

    Returns:
        bool: True if the combined file contains both input images in the specified direction without overlay, False otherwise.
    """

    saved_image_path_1 = from_env_load_and_save_file(env, image_path_1)
    saved_image_path_2 = from_env_load_and_save_file(env, image_path_2)
    saved_file_path = from_env_load_and_save_file(env, file_path)

    # Determine if file_path is already an image

    if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        combined_image_path = saved_file_path
    else:
        # Convert the file to images
        combined_image_path = convert_file_to_images(saved_file_path)[0]

    try:
        # Match the first image within the combined image
        bbox_1, _, _ = image_matcher.match_images(
            saved_image_path_1, combined_image_path
        )

        # Match the second image within the combined image
        bbox_2, _, _ = image_matcher.match_images(
            saved_image_path_2, combined_image_path
        )

        # Check if both bounding boxes are found
        if bbox_1 is None or bbox_2 is None:
            return False

        # Check if bbox_1 is in the specified direction relative to bbox_2
        correct_direction = is_bbox_in_direction(bbox_1, bbox_2, direction)

        return correct_direction
    finally:
        # Cleanup intermediate image files if they were created
        cleanup_files(
            [
                combined_image_path,
                saved_image_path_1,
                saved_image_path_2,
                saved_file_path,
            ]
        )


@evaluator(env_name="ubuntu")
def is_image_2_brighter(image_path_1: str, image_path_2: str) -> bool:
    """
    Check if the second image is brighter than the first image.

    Args:
        image_path_1(str): The path to the first image.
        image_path_2(str): The path to the second image.
    """

    def brightness(image_path: str) -> float:
        # Load the image
        img = cv2.imread(image_path)
        if len(img.shape) == 3:
            # Colored RGB or BGR (*Do Not* use HSV images with this function)
            # create brightness with euclidean norm
            return float(np.average(norm(img, axis=2)) / np.sqrt(3))
        else:
            # Grayscale
            return float(np.average(img))

    brightness_1 = brightness(image_path_1)
    brightness_2 = brightness(image_path_2)

    return brightness_2 > brightness_1


@evaluator(env_name="ubuntu")
def is_img_url_in_clipboard() -> bool:
    """
    Check if the clipboard contains a valid URL or a Data URI that is specific to images.

    Args:
        env (Environment): The current testing environment, used to simulate clipboard functionality.

    Returns:
        bool: True if a valid URL or Data URI specific to images is found in the clipboard, False otherwise.
    """
    clipboard_content = pyperclip.paste()  # Simulate clipboard paste action
    data_uri_pattern = re.compile(
        r"^data:image/(png|jpeg|gif|svg\+xml|bmp|webp);base64,[A-Za-z0-9+/]+={0,2}$",
        re.IGNORECASE,
    )
    is_valid_image_data = bool(re.match(data_uri_pattern, clipboard_content))
    url_pattern = re.compile(
        r"^(https?://)"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    is_valid_url = bool(re.match(url_pattern, clipboard_content))
    if is_valid_url or is_valid_image_data:
        return True
    return False


@evaluator(env_name="ubuntu")
def is_github_repo_url_in_clipboard(keyword: str) -> bool:
    """
    Check if the clipboard contains a valid GitHub repository URL.

    Returns:
        bool: True if the clipboard content is a valid GitHub repository URL, False otherwise.
    """
    clipboard_content = pyperclip.paste()  # Access the clipboard content
    if keyword.lower() not in clipboard_content:
        return False
    github_repo_pattern = re.compile(
        r"^https?://"  # Protocol
        r"github\.com/"  # Domain
        r"[^/]+/"  # Username
        r"[^/]+/?$",  # Repository name, optional trailing slash
        re.IGNORECASE,
    )
    return bool(re.match(github_repo_pattern, clipboard_content))
    # return is_github_repo_url(clipboard_content)


@evaluator(env_name="ubuntu")
def is_software_installed(package_name: str) -> bool:
    try:
        subprocess.check_call(
            ["dpkg", "-s", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


@cache
def get_file_url_hash(url):
    response = requests.get(url)
    response.raise_for_status()
    return hashlib.sha256(response.content).hexdigest()


@evaluator(env_name="ubuntu")
def download_and_verify_file(url: str, file_path: str) -> bool:
    # Check if the file was downloaded
    if not os.path.isfile(file_path):
        return False

    # Calculate the hash of the downloaded file
    with open(file_path, "rb") as f:
        file_data = f.read()
        downloaded_file_hash = hashlib.sha256(file_data).hexdigest()

    # Get the file content directly from the URL
    try:
        original_file_hash = get_file_url_hash(url)
    except requests.RequestException:
        return False

    # Compare the hashes
    return downloaded_file_hash == original_file_hash


@evaluator(env_name="ubuntu")
def download_from_clipboard_and_verify_file(file_path: str) -> bool:
    # Check if the file was downloaded
    if not os.path.isfile(file_path):
        return False

    # Calculate the hash of the downloaded file
    with open(file_path, "rb") as f:
        file_data = f.read()
        downloaded_file_hash = hashlib.sha256(file_data).hexdigest()

    # Get the url from clipboard
    content = pyperclip.paste()
    """
    Problem: 
        1. There exist infinite possibilities of the downloable format in the clipboard. Not sure if we need to verify the format.
    """
    # Get the file content directly from the URL
    try:
        original_file_hash = get_file_url_hash(content)
    except requests.RequestException:
        return False

    # Compare the hashes
    return downloaded_file_hash == original_file_hash


@evaluator(env_name="ubuntu")
def check_color_scheme(assmue: str) -> bool:
    out = subprocess.check_output(
        ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
        text=True,
    )
    return assmue in out


@evaluator(env_name="ubuntu")
def check_text_in_current_window_name(text: str) -> bool:
    try:
        out = subprocess.check_output(
            ["xdotool", "getwindowfocus", "getwindowname"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return False
    return text in out


@evaluator(env_name="ubuntu")
def check_current_window_process(assmue: str) -> bool:
    try:
        out = subprocess.check_output(
            ["xdotool", "getwindowfocus", "getwindowpid"], text=True
        ).strip()
        if not out.isdigit():
            return False
        process = psutil.Process(int(out))
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        psutil.ZombieProcess,
        subprocess.CalledProcessError,
    ):
        return False
    return assmue.strip() == process.name()


@evaluator(env_name="ubuntu")
def check_file_exist(file_path: str) -> bool:
    return os.path.isfile(file_path)


@evaluator(env_name="ubuntu")
def check_file_content(file_path: str, content: str) -> bool:
    if not os.path.isfile(file_path):
        return False
    with open(file_path, "r") as f:
        file_content = f.read()
    return content in file_content


@evaluator(env_name="ubuntu")
def empty_evaluator() -> bool:
    return False


@evaluator(env_name="ubuntu")
def is_process_open(process_name: str) -> bool:
    """
    Check if the given process is currently running.

    Args:
        process_name(str): The process name to check.
    """
    for process in psutil.process_iter(["name"]):
        try:
            if process_name.lower() in process.info["name"].lower():  # type: ignore
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


@evaluator(env_name="ubuntu")
def check_app_usage_history(app_name: str) -> bool:
    """
    Check if the given application has been in the usage history.
    Args:
        app_name(str): The name of the application to check.
    Returns:
        bool: True if the app was recently used, False otherwise.
    """
    for process in psutil.process_iter(["name", "create_time"]):
        try:
            if app_name.lower() in process.info["name"].lower():
                # Assuming 'recently used' implies a running process was started within the last hour
                if time.time() - process.info["create_time"] < 3600:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False


@evaluator(env_name="ubuntu")
def check_process_closed(app_name: str) -> bool:
    """
    Verify that the specified process is not running.
    Args:
        app_name(str): The application name to check for its absence.
    Returns:
        bool: True if the app is not running, False otherwise.
    """
    return not any(
        app_name.lower() in proc.info["name"].lower()
        for proc in psutil.process_iter(["name"])
        if proc.is_running()
    )


@evaluator(env_name="ubuntu")
def verify_background(photo_path: str) -> bool:
    """
    Verify that the specified photo is currently set as the desktop background.

    Args:
        photo_path (str): The path to the photo file.

    Returns:
        bool: True if the photo is the current background, False otherwise.
    """
    out = subprocess.check_output(
        ["gsettings", "get", "org.gnome.desktop.background", "picture-uri"],
        universal_newlines=True,
    )
    current_background = (
        out.strip().split("'")[1].split("file:/")[1]
    )  # Extract the path

    # Compute hashes to compare files
    if os.path.exists(photo_path) and os.path.exists(current_background):
        with open(photo_path, "rb") as f:
            original_hash = hashlib.sha256(f.read()).hexdigest()
        with open(current_background, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()

        return original_hash == current_hash

    return False


@evaluator(env_name="ubuntu")
def is_torch_matmul_example_copied_correctly() -> bool:
    """
    Verify if the clipboard contains the correct torch.matmul example snippets from PyTorch 1.13 documentation.
    """

    def contains_required_strings(
        clipboard_content: str, required_strings: list
    ) -> bool:
        """
        Check if all required strings are present in the clipboard content.

        Args:
            clipboard_content (str): The content from the clipboard.
            required_strings (list): A list of required strings to check.

        Returns:
            bool: True if all required strings are found in the clipboard content, False otherwise.
        """
        for string in required_strings:
            if string not in clipboard_content:
                return False
        return True

    required_strings = [
        "tensor1 = torch.randn",
        "tensor2 = torch.randn",
        "torch.matmul(tensor1, tensor2).size()",
    ]
    clipboard_content = pyperclip.paste().strip()
    if not clipboard_content:
        return False

    return contains_required_strings(clipboard_content, required_strings)


@evaluator(env_name="ubuntu")
def check_directory_exists(dir_path: str) -> bool:
    """Check if the specified directory exists."""
    return os.path.isdir(dir_path)


@evaluator(env_name="ubuntu")
def verify_files_copied(source_dir: str, target_dir: str, file_extension: str) -> bool:
    """Verify that files were copied correctly."""
    source_files = {
        file for file in os.listdir(source_dir) if file.endswith(f".{file_extension}")
    }
    target_files = {
        file for file in os.listdir(target_dir) if file.endswith(f".{file_extension}")
    }
    return source_files == target_files


@evaluator(env_name="ubuntu", local=True)
def check_contain_input_text_list(texts: list[str], env) -> bool:
    """
    Check if all provided search terms were entered in the browser.

    Args:
        search_terms: A list of strings, each representing a search term that needs to be verified.
        env: The current testing environment, used to simulate browser interactions.

    Returns:
        bool: True if all search terms are found in the written text, False otherwise.
    """
    if env.trajectory:
        inputs = [
            params["text"].lower()
            for action_name, params, _ in env.trajectory
            if action_name == "write_text"
        ]
        return all(
            any(term.lower() in input_text for input_text in inputs) for term in texts
        )
    return False


@evaluator(env_name="ubuntu")
def is_google_maps_url_in_clipboard() -> bool:
    """
    Check if the clipboard contains a valid shortened Google Maps URL.
    """
    clipboard_content = pyperclip.paste()
    maps_url_pattern = re.compile(
        r"^https://maps\.app\.goo\.gl/[A-Za-z0-9]+$",
        re.IGNORECASE,
    )
    return bool(re.match(maps_url_pattern, clipboard_content))


@evaluator(env_name="ubuntu", local=True)
def check_contain_input_text(text: str, env) -> bool:
    """
    Check if the input text is contained in the written text action in a case-insensitive manner.

    Args:
        text (str): The text to check for.
        env: The current testing environment, used to access the trajectory.

    Returns:
        bool: True if the input text is found in the written text action, False otherwise.
    """
    if env.trajectory:
        inputs = [
            params["text"].lower()
            for action_name, params, _ in env.trajectory
            if action_name == "write_text"
        ]
        return any(text.lower() in input_text for input_text in inputs)
    return False


@evaluator(env_name="ubuntu")
def verify_country_data_in_ods(country: str, file_path: str) -> bool:
    from bs4 import BeautifulSoup
    from pyexcel_ods import get_data

    def extract_population(text):
        # Use regex to extract the first sequence of numbers which possibly contains commas
        if text:
            match = re.search(r"\d{1,3}(?:,\d{3})*(?=\[|$)", text)
            if match:
                return match.group(0).replace(",", "")  # Remove commas
        return "0"

    def normalize_population(text):
        # Ensure the input is treated as a string, whether it's originally an int or str
        text = str(text)
        # Normalize the population string by removing non-digit characters
        return "".join(filter(str.isdigit, text))

    def fetch_country_data(country):
        country_norm = country.replace(" ", "_")  # Replace spaces with underscores
        url = f"https://en.wikipedia.org/wiki/{country_norm}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        infobox = soup.find("table", {"class": "infobox"})
        capital_city = None
        population = None

        if infobox:
            for row in infobox.find_all("tr"):
                header = row.find("th")
                if header:
                    header_text = header.text.strip()
                    if "Capital" in header_text:
                        capital_city = row.find("td").text.strip()
                        capital_city = " ".join(
                            capital_city.split()
                        )  # Normalize and clean up text
                    if "Population" in header_text:
                        if row.find("td"):
                            population_text = row.find("td").text.strip()
                        else:
                            next_row = row.find_next_sibling("tr")
                            if next_row and next_row.find("td"):
                                population_text = next_row.find("td").text.strip()
                        population = extract_population(population_text)

        return capital_city, population

    capital_city, population = fetch_country_data(country)

    if not capital_city or not population:
        return False

    # Load data from ODS file
    data = get_data(file_path)
    sheet = data[list(data.keys())[0]]  # Assume data is in the first sheet

    # Search for country and verify data
    for row in sheet:
        if row[0].lower() == country.lower():
            recorded_capital_city = row[1]
            recorded_population = normalize_population(row[2])
            # Check if the capital city and population in the sheet match Wikipedia
            if (
                recorded_capital_city in capital_city
                and recorded_population == population
            ):
                return True
            else:
                return False

    return True


ubuntu_subtasks = [
    SubTask(
        id="0f589bf9-9b26-4581-8b78-2961b115ab49",
        description='Open "{file_path}" using vim in a terminal, write "{content}", then save and exit vim.',
        attribute_dict={"file_path": "file_path", "content": "message"},
        output_type="file_path",
        output_generator=lambda file_path, content: file_path,
        evaluator_generator=lambda file_path, content: nx.path_graph(
            [
                check_current_window_process("gnome-terminal-server"),
                is_process_open("vim"),
                ~is_process_open("vim"),
                check_file_content(file_path, content),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="5b527839-0e58-426d-bab6-7160200b0d24",
        description='Get the content of "{file_path}" by printing it to the command line interface through a terminal',
        attribute_dict={"file_path": "file_path"},
        output_type="message",
        output_generator="manual",
        evaluator_generator=lambda file_path: nx.path_graph(
            [
                check_current_window_process("gnome-terminal-server"),
                check_contain_input_text("cat " + file_path),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="1c3bedc3-ea5a-453c-a15b-223d72ab756d",
        description='Submit content "{content}"',
        attribute_dict={"content": "message"},
        output_type="None",
        output_generator="manual",
        evaluator_generator=lambda content: nx.path_graph(
            [
                check_submit(content),
            ],
            create_using=nx.DiGraph,
        ),
        extra_action=[submit],
    ),
    SubTask(
        id="a313ea4d-e501-4971-b4fe-db2aad19eac1",
        description='Download a file from "{url}" to "{file_path}".',
        attribute_dict={"url": "url", "file_path": "file_path"},
        output_type="file_path",
        output_generator=lambda file_path, content: file_path,
        evaluator_generator=lambda url, file_path: nx.path_graph(
            [
                download_and_verify_file(url, file_path),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="a313ea4d-e501-4971-b4fe-db2aad19acsd",
        description='Download a file from the URL stored in the clipboard to "{file_path}".',
        attribute_dict={"file_path": "file_path"},
        output_type="file_path",
        output_generator=lambda file_path, content: file_path,
        evaluator_generator=lambda file_path: nx.path_graph(
            [
                download_from_clipboard_and_verify_file(file_path),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="017102b6-d2c3-466b-96f7-37c8bcddc41a",
        description='Use Firefox to search for an image using the keyword "{keyword}" and copy the URL of the image to the clipboard.',
        attribute_dict={"keyword": "keyword"},
        output_type="None",
        evaluator_generator=lambda keyword: path_graph(
            [
                check_text_in_current_window_name("Mozilla Firefox"),
                check_contain_input_text(keyword),
                is_img_url_in_clipboard(),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="bcd03c9f-62c9-4001-8d86-78358c59ce22",
        description='Use Firefox to find a code repository about "{keyword}" in GitHub and copy the URL of the repository to the clipboard.',
        attribute_dict={"keyword": "keyword"},
        output_type="None",
        evaluator_generator=lambda keyword: path_graph(
            [
                check_text_in_current_window_name("GitHub — Mozilla Firefox"),
                check_contain_input_text(keyword),
                is_github_repo_url_in_clipboard(keyword),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a207ef38-b3b2-4c6c-a1e3-75c38162f5ba",
        description='Set "{photo_path}" as the screen background of the system',
        attribute_dict={"photo_path": "photo_path"},
        output_type="None",
        evaluator_generator=lambda photo_path: path_graph(
            [verify_background(photo_path)],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="217ababc-ccc7-4b9f-af07-c239d92848fe",
        description='Create a new directory "{target_dir}" and copy all files with the specified "{file_extension}" extension from "{source_dir}" to the directory "{target_dir}".',
        attribute_dict={
            "file_extension": "file_extension",
            "source_dir": "dir_path",
            "target_dir": "dir_path",
        },
        output_type="message",
        evaluator_generator=lambda file_extension,
        source_dir,
        target_dir: nx.path_graph(
            [
                check_directory_exists(target_dir),
                verify_files_copied(source_dir, target_dir, file_extension),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="2b189dc2-c77f-4fa3-8432-ba4355cc294c",
        description='Use Firefox to find out a "{place_type}" around "{place_name}" on Google Maps and copy the Google Maps sharing URL of that "{place_type}" to the clipboard',
        attribute_dict={"place_type": "place_type", "place_name": "place_name"},
        output_type="None",
        evaluator_generator=lambda place_type, place_name: path_graph(
            [
                # check_current_window_process("firefox"),
                check_text_in_current_window_name("Google Maps — Mozilla Firefox"),
                check_contain_input_text_list([place_name, place_type]),
                is_google_maps_url_in_clipboard(),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="cc1adae7-bef9-4c8a-865d-00d44486dd69",
        description='Use GIMP (GNU Image Manipulation Program) to adjust the brightness of the image from "{image_path_before_edit}" to a higher value (brighter) and save it to "{image_path_after_edit}".',
        attribute_dict={
            "image_path_before_edit": "photo_path",
            "image_path_after_edit": "photo_path",
        },
        output_type="photo_path",
        evaluator_generator=lambda image_path_before_edit,
        image_path_after_edit: nx.path_graph(
            [
                check_text_in_current_window_name("GNU Image Manipulation Program"),
                check_file_exist(image_path_after_edit),
                is_image_2_brighter(image_path_before_edit, image_path_after_edit),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="434402f3-647a-4a9a-9d8f-10f5bb6c7cf0",
        description='Use LibreOffice Impress to adjust the brightness of the image from "{image_path_before_edit}" to a lower value (darker) and save it to "{image_path_after_edit}".',
        attribute_dict={
            "image_path_before_edit": "photo_path",
            "image_path_after_edit": "photo_path",
        },
        output_type="photo_path",
        evaluator_generator=lambda image_path_before_edit,
        image_path_after_edit: nx.path_graph(
            [
                check_text_in_current_window_name("LibreOffice Impress"),
                check_file_exist(image_path_after_edit),
                ~is_image_2_brighter(image_path_before_edit, image_path_after_edit),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="4cf246ea-0a7f-43da-84b6-61d74a2699af",
        description='Combine two images from Image 1 "{image_path_1}" and Image 2 "{image_path_2} using GIMP (GNU Image Manipulation Program) and save the resulting image to "{output_path}". Image 1 should be placed on the left side of Image 2.',
        attribute_dict={
            "image_path_1": "photo_path_1",
            "image_path_2": "photo_path_2",
            "output_path": "photo_path_ouput",
        },
        output_type="photo_path",
        evaluator_generator=lambda image_path_1,
        image_path_2,
        output_path: nx.path_graph(
            [
                check_text_in_current_window_name("GNU Image Manipulation Program"),
                check_file_exist(output_path),
                verify_combined_image(image_path_1, image_path_2, output_path, "left"),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="0111384f-38ca-41a2-9504-cb1c55002b3c",
        description='Combine two images from Image 1 "{image_path_1}" and Image 2 "{image_path_2}" using LibreOffice Writer and save the resulting ODT file to "{output_path}". Image 1 should be placed above Image 2.',
        attribute_dict={
            "image_path_1": "photo_path_1",
            "image_path_2": "photo_path_2",
            "output_path": "file_path",
        },
        output_type="file_path",
        evaluator_generator=lambda image_path_1,
        image_path_2,
        output_path: nx.path_graph(
            [
                check_text_in_current_window_name("LibreOffice Writer"),
                check_file_exist(output_path),
                verify_combined_image(image_path_1, image_path_2, output_path, "above"),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="467f17a6-c42f-4eda-996f-a53385eb3efd",
        description='Combine two images from Image 1 "{image_path_1}" and Image 2 "{image_path_2}" using LibreOffice Impress and save the resulting file in PDF format to "{output_path}". Image 1 should be placed on the right side of Image 2.',
        attribute_dict={
            "image_path_1": "photo_path_1",
            "image_path_2": "photo_path_2",
            "output_path": "file_path",
        },
        output_type="file_path",
        evaluator_generator=lambda image_path_1,
        image_path_2,
        output_path: nx.path_graph(
            [
                check_text_in_current_window_name("LibreOffice Impress"),
                check_file_exist(output_path),
                verify_combined_image(image_path_1, image_path_2, output_path, "right"),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    SubTask(
        id="49b614c5-c4bb-4c20-aab8-ab9dcc7de1b5",
        description="Find the example provided of torch.matmul by official PyTorch version 1.13 documentation using Firefox and copy all the lines of code in the example to the clipboard.",
        attribute_dict={},
        output_type="None",
        evaluator_generator=lambda: nx.path_graph(
            [
                check_text_in_current_window_name(
                    "torch.matmul — PyTorch 1.13 documentation — Mozilla Firefox"
                ),
                is_torch_matmul_example_copied_correctly(),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="76de4bdb-c980-4b3a-9bd3-c87db467dffe",
        description='Paste clipboard content into LibreOffice Writer and save it as an ODT file at "{file_path}".',
        attribute_dict={"file_path": "file_path"},
        output_type="file_path",
        evaluator_generator=lambda file_path: path_graph(
            [
                check_text_in_current_window_name("LibreOffice Writer"),
                check_file_exist(file_path),
                verify_odt_file_content_with_clipboard(file_path),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="8491e674-596b-452b-9e0e-58a44d90f947",
        description='Paste clipboard content into Visual Studio Code (VS Code) and save it as a file at "{file_path}".',
        attribute_dict={"file_path": "file_path"},
        output_type="file_path",
        evaluator_generator=lambda file_path: path_graph(
            [
                check_text_in_current_window_name("Visual Studio Code"),
                check_file_exist(file_path),
                verify_file_content_with_clipboard(file_path),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="1cd6519a-9ee0-442b-ba5a-9238aeb00ff6",
        description='Use Firefox to search for the country "{country}" on Wikipedia, extract the capital city and population, and save this information in an ODS file at "{file_path}" with LibreOffice Calc. The first column will save the country name, the second will save the capital city name, and the third will save the population. No header is needed in the ODS file.',
        attribute_dict={"country": "country", "file_path": "file_path"},
        output_type="file_path",
        evaluator_generator=lambda country, file_path: nx.path_graph(
            [
                check_text_in_current_window_name("Wikipedia — Mozilla Firefox"),
                check_text_in_current_window_name("LibreOffice Calc"),
                check_file_exist(file_path),
                verify_country_data_in_ods(country, file_path),
            ],
            create_using=nx.DiGraph,
        ),
    ),
]


if __name__ == "__main__":
    generator = TaskGenerator(attribute_pool={})

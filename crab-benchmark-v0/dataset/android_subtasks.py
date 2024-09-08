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
import io
import os
import re
from typing import Callable, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
import torch
from lxml import etree
from lxml.etree import _Element
from networkx import DiGraph, path_graph
from PIL import Image

from crab import SubTask, evaluator
from crab.actions.android_actions import execute_adb, screenshot


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
            converted_image = np.copy(np.array(Image.open(output))[..., ::-1])
        return converted_image


image_matcher = ImageMatcher()


def cleanup_files(files: List[str]):
    """
    Delete the list of files.

    Args:
        files (List[str]): List of paths to the files to be deleted.
    """
    for file in files:
        os.remove(file)


def from_env_load_and_save_screenshot(env, output_dir="/tmp/local_save"):
    """
    Load a file, convert it to bytes, and save it to a local directory with the same basename.

    Args:
        env: The environment object with the _action_endpoint method.
        output_dir (str): The directory where the file should be saved (default is "/tmp/local_save").

    Returns:
        str: The path to the saved file.
    """

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the file and convert to bytes
    encoded_string = env._action_endpoint(screenshot, {"env": env})

    # Decode the Base64 string back to bytes
    decoded_bytes = base64.b64decode(encoded_string.encode("utf-8"))

    # Create the output file path
    file_name = "temp_file.png"  # Replace with an appropriate file name
    output_file_path = os.path.join(output_dir, file_name)

    # Save the decoded bytes to the output path
    with open(output_file_path, "wb") as file:
        file.write(decoded_bytes)

    return output_file_path


def get_xml_etree(env) -> _Element | None:
    xml_str = execute_adb("exec-out uiautomator dump /dev/tty", env)
    if "UI hierchary dumped to: /dev/tty" not in xml_str:
        return None
    xml_str = xml_str.removesuffix("UI hierchary dumped to: /dev/tty")
    return etree.fromstring(xml_str.encode("utf-8"))


@evaluator(env_name="ubuntu2204", local=True)
def verify_screenshot_containing_image(image_path: str, env) -> bool:
    """
    Verifies whether a given image is contained within a screenshot taken from the current environment.

    This function captures a screenshot of the current environment using the `screenshot` action and checks
    if the image at `image_path` is contained within the screenshot. If the image is found within the
    screenshot, it returns True; otherwise, it returns False.

    Parameters:
    - image_path (str): The path to the image file that needs to be checked against the screenshot.
    - env: The environment object that contains methods to interact with the environment (e.g., capturing screenshots).

    Returns:
    - bool: True if the image is found within the screenshot, False otherwise.
    """

    # Capture and save the screenshot from the environment to a local path
    screenshot_path = from_env_load_and_save_screenshot(env)

    try:
        # Match the given image against the captured screenshot
        bbox, _, _ = image_matcher.match_images(image_path, screenshot_path)

        # If a bounding box is found, the image is contained within the screenshot
        return bbox is not None

    finally:
        # Clean up the saved screenshot file to free up resources
        cleanup_files([screenshot_path])


@evaluator(env_name="android", local=True)
def check_contain_input_text(text: str, env) -> bool:
    if env.trajectory:
        action_name, params, _ = env.trajectory[-1]
        if action_name == "write_text" and text.lower() in params["text"].lower():
            return True
    return False


@evaluator(env_name="android", local=True)
def check_contain_input_text_multiple(text: str, env) -> bool:
    if env.trajectory:
        for action_name, params, _ in env.trajectory:
            if action_name == "write_text" and text in params["text"].lower():
                return True
    return False


@evaluator(env_name="android")
def check_contain_contact(name: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    title_node = root.xpath(
        '//node[@resource-id="com.android.contacts:id/photo_touch_intercept_overlay"]'
    )
    if not title_node:
        return False
    if title_node[0].get("content-desc") != name:
        return False
    info_node = root.xpath('//*[@class="android.widget.RelativeLayout"]')
    if not info_node:
        return False
    print("info node checked")
    mail_node = None
    for node in info_node:
        desc = node.get("content-desc")
        if "Email" in desc:
            mail_node = node
    if mail_node == None:
        return False
    real_mail_node = mail_node.xpath(
        '//*[@resource-id="com.android.contacts:id/header"]'
    )
    if not real_mail_node:
        return False
    context = real_mail_node[0].get("text")
    print("context get")
    pattern = re.compile(r"^\w+@\w+.com")
    if pattern.match(context):
        return True
    return False


@evaluator(env_name="android")
def check_current_package_name(name: str, env) -> bool:
    result = execute_adb(
        r'shell "dumpsys activity activities | grep mResumedActivity"', env
    )
    return name in result


@evaluator(env_name="android", local=True)
def check_ocr_results(text: str, env) -> bool:
    return text in env.ocr_results


@evaluator(env_name="android")
def check_current_message_page(title: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    title_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/conversation_title"]'
    )
    if title_node:
        return title == title_node[0].get("text")
    else:
        return False


@evaluator(env_name="android")
def check_message_text_box_contain(text: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    text_box_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/compose_message_text"]'
    )
    if text_box_node:
        return text.lower() in text_box_node[0].get("text").lower()
    else:
        return False


@evaluator(env_name="android")
def check_message_text_box_empty(env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    text_box_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/compose_message_text"]'
    )
    if not text_box_node:
        return False
    if text_box_node[0].get("text").strip() == "Text message":
        return True
    else:
        return False


@evaluator(env_name="android")
def check_send_message(title: str, message: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    title_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/conversation_title"]'
    )
    if not title_node or title != title_node[0].get("text"):
        return False
    messages_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/message_text"]'
    )
    for node in messages_node:
        if message in node.get("text"):
            return True
    return False


@evaluator(env_name="android")
def check_note_content(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    title_node = root.xpath(
        '//node[@resource-id="com.google.android.keep:id/editable_title"]'
    )
    if not title_node:
        return False
    if title_node[0].get("text") != "Title":
        return False
    node = root.xpath(
        '//node[@resource-id="com.google.android.keep:id/edit_note_text"]'
    )
    if not node:
        return False
    if content in node[0].get("text"):
        return True
    return False


@evaluator(env_name="android")
def check_bluetooth_name(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    bluetooth_node = root.xpath('//node[@resource-id="android:id/summary"]')
    if not bluetooth_node:
        return False
    if content in bluetooth_node[0].get("text"):
        return True
    return False


@evaluator(env_name="android")
def check_map_direction_page(from_des: str, to_des: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    from_node = root.xpath(f'//node[@content-desc="Start location, {from_des}"]')
    if not from_node:
        return False
    to_node = root.xpath(f'//node[@content-desc="Destination, {to_des}"]')
    if not to_node:
        return False
    return True


@evaluator(env_name="android")
def check_dial_number(phone_number: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    dialer_node = root.xpath('//node[@resource-id="com.android.dialer:id/digits"]')
    if not dialer_node:
        return False
    number = dialer_node[0].get("text")
    number = re.sub("[^0-9]", "", number)
    target = re.sub("[^0-9]", "", phone_number)
    return number == target


@evaluator(env_name="android")
def check_calendar_registered(date: str, content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    calendar_node = root.xpath(
        '//node[@resource-id="com.google.android.calendar:id/alternate_timeline_fragment_container"]'
    )
    if not calendar_node:
        return False
    itr_calendar_node = calendar_node[0].xpath(
        '//node[@class="android.support.v7.widget.RecyclerView"]'
    )
    if not itr_calendar_node:
        return False
    target_nodes = itr_calendar_node[0].xpath('//node[@content-desc="{content}"]')
    if not target_nodes:
        return False
    return True


@evaluator(env_name="android")
def check_drive_registered(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    entry_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.docs:id/entry_label"]'
    )
    if not entry_node:
        return False
    for node in entry_node:
        if content == node.get("text") and f"{content} Folder" == node.get(
            "content-desc"
        ):
            return True
    return False


@evaluator(env_name="android")
def check_contact_registered(mail: str, name: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    name_node = root.xpath('//node[@resource-id="com.android.contacts:id/large_title"]')
    if not name_node:
        return False
    text = name_node[0].get("text")
    if text not in name:
        return False

    mail_node = root.xpath('//node[@resource-id="com.android.contacts:id/header"]')
    text = mail_node[0].get("text")
    if text not in mail:
        return False
    return True


@evaluator(env_name="android")
def check_calling_number(phone_number: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    dialer_node = root.xpath(
        '//node[@resource-id="com.android.dialer:id/contactgrid_contact_name"]'
    )
    if not dialer_node:
        return False
    number = dialer_node[0].get("text")
    number = re.sub("[^0-9]", "", number)
    target = re.sub("[^0-9]", "", phone_number)
    return number == target


@evaluator(env_name="android")
def check_google_tasks_name(target: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    task_nodes = root.xpath(
        '//node[@resource-id="com.google.android.apps.tasks:id/task_name"]'
    )
    if not task_nodes:
        return False
    for node in task_nodes:
        task_name = node.get("text")
        if target in task_name:
            return True
    return False


@evaluator(env_name="android")
def check_date(target: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    date_nodes = root.xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/datetime_item_layout"]'
    )
    if not date_nodes:
        return False
    prev_node = date_nodes.xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/label"]'
    )
    time = prev_node.get("text")
    pattern = re.compile(r"^\w{3},\s\w{3}\s\d{2},\s\d{4}\s•\s\d{1,2}:\d{2}\s[AP]M$")
    if pattern.match(time):
        return True
    return False


@evaluator(env_name="android")
def check_city_clock(place_name: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    city_nodes = root.xpath(
        '//node[@resource-id="com.google.android.deskclock:id/city_name"]'
    )
    if city_nodes is None:
        return False
    for city_node in city_nodes:
        text = city_node.get("text")
        if place_name == text:
            return True
    return False


@evaluator(env_name="android")
def check_event(date: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    event_nodes = root.xpath('//node[@class="android.support.v7.widget.RecyclerView"]')
    if event_nodes is None:
        return False
    for node in event_nodes[0]:
        text = node.get("content-desc")
        if date in text:
            return True
    return False


@evaluator(env_name="android")
def check_event_registered(date: str, content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    event_nodes = root.xpath('//node[@class="android.support.v7.widget.RecyclerView"]')
    if not event_nodes:
        return False
    time_reg = False
    content_reg = False
    for node in event_nodes[0]:
        text = node.get("content-desc")
        if date.lower() in text.lower():
            time_reg = True
        if content.lower() in text.lower():
            content_reg = True
    if time_reg and content_reg:
        return True
    return False


@evaluator(env_name="android")
def check_location(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    checked_node = root.xpath(f'//node[@content-desc="{content}"]')
    if not checked_node:
        return False
    return True


@evaluator(env_name="android")
def check_contain_city(number: str, city: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    business_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.maps:id/search_omnibox_text_box"]'
    )
    if not business_node:
        return False
    text = None
    for node in business_node[0]:
        text = node.get("text")
    if text is None:
        return False
    if city in text and str(number) in text:
        return True
    return False


@evaluator(env_name="android")
def check_file(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    name_source_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/exif_item_layout"]'
    )
    if not name_source_node:
        return False
    name_nodes = name_source_node[0].xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/label"]'
    )
    if not name_nodes:
        return False
    target_node = None
    for node in name_nodes:
        text = node.get("text")
        if content in text:
            target_node = node
    if target_node is None:
        return False
    time_source_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/datetime_item_layout"]'
    )
    if not time_source_node:
        return False
    time_nodes = time_source_node[0].xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/label"]'
    )
    if not time_nodes:
        return False
    target_node = None
    for node in time_nodes:
        text = node.get("text")
        pattern = re.compile(
            r"(Tue|Mon|Wed|Thu|Fri|Sat|Sun),\s(May|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2},\s\d{4} • \d{2}:\d{2}\s(AM|PM)"
        )
        if pattern.match(text):
            return True
        return False


@evaluator(env_name="android")
def check_mail_sent(mail: str, content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    to_node = root.xpath(
        '//node[@resource-id="com.google.android.gm:id/peoplekit_chip"]'
    )
    if not to_node:
        return False
    checked = False
    for node in to_node:
        text = node.get("content-desc")
        if mail in text:
            checked = True
    if not checked:
        return False
    # check the mail information-> Done

    # check the content information
    body_node = root.xpath(
        '//node[@resource-id="com.google.android.gm:id/body_wrapper"]'
    )
    if not body_node:
        return False
    text_node = body_node[0].xpath('//node[@class="android.widget.EditText"]')
    if not text_node:
        return False
    for node in text_node:
        text = node.get("text")
        if content in text:
            return True
    return False


def distance_evaluator_generator(place_name_1: str, place_name_2: str):
    result = nx.DiGraph()
    a = check_current_package_name("com.google.android.apps.maps")
    b = check_contain_input_text(place_name_1)
    c = check_contain_input_text(place_name_2)
    d = check_map_direction_page(place_name_1, place_name_2)
    result.add_edges_from([(a, b), (a, c), (b, d), (c, d)])
    return result


def mail_evaluator_generator(mail: str, content: str):
    result = nx.DiGraph()
    a = check_current_package_name("com.google.android.gm")
    b = check_contain_input_text(mail)
    c = check_contain_input_text(content)
    d = check_mail_sent(mail, content)
    result.add_edges_from([(a, b), (a, c), (b, d), (c, d)])
    return result


def contact_evaluator_generator(mail: str, name: str):
    result = nx.DiGraph()
    a = check_current_package_name("com.android.contacts")
    b = check_contain_input_text(mail)
    c = check_contain_input_text(name)
    d = check_contact_registered(mail, name)
    result.add_edges_from([(a, b), (a, c), (b, d), (c, d)])
    return result


android_subtasks = [
    SubTask(
        id="e5b05095-7167-4c6b-ba8d-ad8df2e2d12f",
        description='In Android, using "XXXX" App, to put the latest image in "Photos" App "XXXX doing whatX XXX"',  # TODO: replace with actual app name and operation
        attribute_dict={},
        output_type="None",
        evaluator_generator=lambda content: path_graph(
            [
                verify_screenshot_containing_image(
                    "image_path_on_host"
                )  # TODO: replace with actual image path, this image path is on the host (not in the Android environment), and the image should be the same as the latest image put in the "Photos" app
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="1a1b72d7-78c9-4027-8278-86083ae01045",
        description='In Android, using "Google Map" App, find the distance of the shortest route from "{place_name_1}" to "{place_name_2}"',
        attribute_dict={"place_name_1": "place_name_1", "place_name_2": "place_name_2"},
        output_type="number",
        evaluator_generator=distance_evaluator_generator,
    ),
    SubTask(
        id="eb92a1e6-4c86-4d56-baac-95fc8397732e",
        description='In Android, using "Keep Notes" App, record "{content}" in a new note without title.',
        attribute_dict={"content": "content"},
        output_type="None",
        evaluator_generator=lambda content: path_graph(
            [
                check_current_package_name("com.google.android.keep"),
                check_contain_input_text(content),
                check_note_content(content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="caa29623-1811-402d-963a-19f7eecc63d8",
        description='In Android, using "Messages", send "{content}" to "{number}".',
        attribute_dict={"content": "content", "number": "number"},
        output_type="None",
        evaluator_generator=lambda content, number: path_graph(
            [
                check_current_package_name("com.google.android.apps.messaging"),
                check_current_message_page(number),
                check_contain_input_text(content),
                check_send_message(number, content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="955d8773-dd7a-4072-b87c-7e546be7de4e",
        description='In Android, call "{number}".',
        attribute_dict={"number": "number"},
        output_type="None",
        evaluator_generator=lambda number: path_graph(
            [
                check_current_package_name("com.android.dialer"),
                check_dial_number(number),
                check_calling_number(number),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548af",
        description='Using "Tasks" app, add a new task with text "{content}".',
        attribute_dict={"content": "content"},
        output_type="None",
        evaluator_generator=lambda content: path_graph(
            [
                check_current_package_name("com.google.android.apps.tasks"),
                check_contain_input_text(content),
                check_google_tasks_name(content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548ac",
        description='In Android, Using "Calendar" app, add a new event with text "{content}" in date "{date}" all day.',
        attribute_dict={"content": "content", "date": "date"},
        output_type="None",
        evaluator_generator=lambda content, date: path_graph(
            [
                check_current_package_name("com.google.android.calendar"),
                check_contain_input_text(content),
                check_event_registered(date, content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548ag",
        description='In Android, Using "Contacts" app, add a contact with a mail "{mail}" with a name "{name}".',
        attribute_dict={"mail": "mail", "name": "name"},
        output_type="None",
        evaluator_generator=contact_evaluator_generator,
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548ap",
        description='In Android, Using "Contacts" app, find out the mail of contact named {name}.',
        attribute_dict={"name": "name"},
        output_type="mail",
        evaluator_generator=lambda name: path_graph(
            [
                check_current_package_name("com.android.contact"),
                check_contain_contact(name),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="0090f116-e02b-4562-a20d-b5df38be963a",
        description='In Android, Using "Gmail" app, send {mail} a message {content}.',
        attribute_dict={"content": "content", "mail": "mail"},
        output_type="None",
        evaluator_generator=mail_evaluator_generator,
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548ar",
        description='In Android, Using "Google Drive" app, create a new folder named {content}.',
        attribute_dict={"content": "content"},
        output_type="None",
        evaluator_generator=lambda content: path_graph(
            [
                check_current_package_name("com.google.android.apps.docs"),
                check_drive_registered(content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548ak",
        description='In Android, Using "Files" app, find the create date of {file_path}.',
        attribute_dict={"file_path": "file_path"},
        output_type="Date",
        evaluator_generator=lambda file_path: path_graph(
            [
                check_current_package_name("com.google.android.apps.photos"),
                check_file(file_path),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548an",
        description='In Android, Using "Setting" app, rename the device name of bluetooth as {name}.',
        attribute_dict={"content": "content"},
        output_type="None",
        evaluator_generator=lambda content: path_graph(
            [
                check_current_package_name("com.android.settings"),
                check_contain_input_text(content),
                check_bluetooth_name(content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548ah",
        description='In Android, Using "Clock" app, set the time of {place_name} in the clock, check the time gap between the city and current city.',
        attribute_dict={"place_name": "place_name"},
        output_type="content",
        evaluator_generator=lambda place_name: path_graph(
            [
                check_current_package_name("com.google.android.deskclock"),
                check_city_clock(place_name),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548aw",
        description='In Android, Using "Google Map" app, Find the address of {content}',
        attribute_dict={"content": "content"},
        output_type="content",
        evaluator_generator=lambda content: path_graph(
            [
                check_current_package_name("com.google.android.apps.maps"),
                check_location(content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="51b2463c-9904-4a32-81ba-507bfb89d61f",
        description='In Android, Using "Google Map" app, Find the city name of corresponding post code "{number}" in the country "{country}".',
        attribute_dict={"number": "number", "country": "country"},
        output_type="content",
        evaluator_generator=lambda number, country: path_graph(
            [
                check_current_package_name("com.google.android.apps.maps"),
                check_contain_input_text(country),
                check_contain_input_text(number),
                check_contain_city(number, country),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="2394b768-2ca7-45e9-b41e-2aa4e9573192",
        description='In android system, use the calendar app, find the title of an event in the date "{date}".',
        attribute_dict={"date": "date"},
        output_type="content",
        evaluator_generator=lambda date: path_graph(
            [
                check_current_package_name("com.google.android.calendar"),
                check_event(date),
            ],
            create_using=DiGraph,
        ),
    ),
    # TODO: The phone number page cannot be accesed by xml. figure out another way.
    # SubTask(
    #     id="fa9c0b01-9835-4932-824d-0990cb20e5f7",
    #     description='Using Settings app, find the phone number of this phone in the "About" panel.',
    #     attribute_dict={},
    #     output_type="phone_number",
    #     evaluator=lambda: path_graph(
    #         [
    #             check_current_package_name("com.android.settings"),
    #         ],
    #         create_using=DiGraph,
    #     ),
    # ),
]

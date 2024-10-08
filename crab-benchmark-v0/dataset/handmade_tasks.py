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
# ruff: noqa: E501 F405
import os
import re
import subprocess
import time
from datetime import datetime

import networkx as nx

from crab import Task, action, evaluator

from .android_subtasks import (
    check_current_package_name,
    check_google_tasks_name,
    check_message_text_box_contain,
    check_message_text_box_empty,
    check_note_content,
    get_xml_etree,
)
from .ubuntu_subtasks import *  # noqa: F403

_item_count_cache = None


@evaluator(env_name="android")
def check_calendar_in_today(env) -> bool:
    # Get today's date and format it as "Weekday DD Month YYYY"
    today_date_str = datetime.now().strftime("%A %d %B %Y")

    root = get_xml_etree(env)
    if root is None:
        return False
    # Construct the desired string with today's date
    date_string = f"{today_date_str}, Open Schedule View"
    date_node = root.xpath(f'//node[@content-desc="{date_string}"]')
    if not date_node or len(date_node) != 1:
        return False
    today_nodes = date_node[0].getparent().getchildren()
    item_count = len(today_nodes) - 2
    if item_count < 0:
        return False
    global _item_count_cache
    _item_count_cache = item_count
    return True


@action(env_name="ubuntu")
def get_file_bullet_points(file_path: str) -> int | None:
    # Check if the file exists
    if not os.path.exists(file_path):
        return None

    # Read the markdown text from the file
    try:
        with open(file_path, "r") as file:
            markdown_text = file.read()
    except Exception:
        return None

    # Regex to match empty checkboxes in markdown
    pattern = r"- \[ \]"
    # Find all matches
    matches = re.findall(pattern, markdown_text)
    # Return the number of empty checkboxes
    return matches


@evaluator(env_name="ubuntu", local=True)
def check_blluet_point_match_calendar(file_path: str, env) -> bool:
    matches = env._action_endpoint(get_file_bullet_points, {"file_path": file_path})
    global _item_count_cache
    if _item_count_cache is None or matches is None:
        return False
    return _item_count_cache == len(matches)


@evaluator(env_name="android")
def check_node_exist(node_query: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    node = root.xpath(f"//node[{node_query}]")
    if not node:
        return False
    return True


@evaluator(env_name="ubuntu")
def check_new_jpg_files_in_dir(directory) -> bool:
    # Get the current time
    current_time = time.time()
    # Time limit set to 3 minutes ago
    time_limit = current_time - 180

    # Iterate over files in the specified directory
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        # Check if the file is a .jpg and was modified within the last 3 minutes
        if file.endswith(".jpg") and os.path.getmtime(file_path) > time_limit:
            return True

    return False


@evaluator(env_name="ubuntu")
def check_text_list_in_current_window_name(texts: list[str]) -> bool:
    try:
        out = subprocess.check_output(
            ["xdotool", "getwindowfocus", "getwindowname"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return False
    for text in texts:
        if text not in out:
            return False
    return True


@evaluator(env_name="android")
def check_keep_notes_content(text: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    edit_node = root.xpath(
        '//node[@resource-id="com.google.android.keep:id/editor_bottom_bar"]'
    )
    if len(edit_node) != 1:
        return False
    content_node = root.xpath(
        '//node[@resource-id="com.google.android.keep:id/browse_note_interior_content"]'
    )
    if len(content_node) != 1:
        return False
    text_nodes = content_node[0].getchildren()
    if len(text_nodes) != 1:
        return False
    return text_nodes[0].get("text") == text


@evaluator(env_name="android")
def check_keep_notes_contain_fd(env) -> bool:
    global RESULT_fd0576be
    text = RESULT_fd0576be
    root = get_xml_etree(env)
    if root is None or text is None:
        return False
    edit_node = root.xpath(
        '//node[@resource-id="com.google.android.keep:id/editor_bottom_bar"]'
    )
    if len(edit_node) != 1:
        return False
    content_node = root.xpath(
        '//node[@resource-id="com.google.android.keep:id/browse_note_interior_content"]'
    )
    for node in content_node:
        text_nodes = node.getchildren()
        if len(text_nodes) != 1:
            continue
        if text in text_nodes[0].get("text"):
            return True
    return False


@evaluator(env_name="android")
def check_alarm_contains(time: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None or time is None:
        return False
    clock_node = root.xpath(
        '//node[@resource-id="com.google.android.deskclock:id/digital_clock"]'
    )
    for node in clock_node:
        if time == node.get("text"):
            return True
    return False


@evaluator(env_name="android", local=True)
def check_tap_text(text: str, env) -> bool:
    if env.trajectory:
        action_name, params, _ = env.trajectory[-1]
        if action_name == "tap":
            try:
                element_id = int(params["element"])
                element_label = env.element_label_map[element_id]
            except TypeError:
                return False
            if element_label is None:
                return False
            return text.lower() in element_label.lower()
    return False


def summarize_ubuntu_evaluator():
    result = nx.DiGraph()
    a = check_current_window_process("slack")
    b = check_current_package_name("com.google.android.apps.messaging")
    c = check_message_text_box_contain("agent")
    d = check_message_text_box_contain("github")
    e = check_message_text_box_empty()
    result.add_edges_from([(a, c), (a, d), (b, c), (b, d), (c, e), (d, e)])
    return result


def check_calendar_evaluator():
    result = nx.DiGraph()
    a = check_current_package_name("com.google.android.calendar")
    b = check_calendar_in_today()
    c = check_file_exist("/home/crab/assets/plan.md")
    d = check_blluet_point_match_calendar("/home/crab/assets/plan.md")
    result.add_edges_from([(a, b), (b, d), (c, d)])
    return result


def evaluator_97e6f333():
    result = nx.DiGraph()
    a = check_current_package_name("com.android.camera2")
    b = check_node_exist('@resource-id="com.android.camera2:id/rounded_thumbnail_view"')
    c = check_node_exist('@resource-id="com.android.camera2:id/filmstrip_layout"')
    d = check_current_package_name(
        "com.google.android.apps.photos/.upload.intent.UploadContentActivity"
    )
    e = check_node_exist('@resource-id="com.android.camera2:id/filmstrip_layout"')
    f = check_current_window_process("firefox")
    g = check_text_in_current_window_name("Photos - Google Photos — Mozilla Firefox")
    h = check_new_jpg_files_in_dir("/home/crab/Downloads")
    i = check_file_exist("/home/crab/assets/photo.jpg")
    j = check_text_list_in_current_window_name(["photo", "GIMP"])
    result.add_edges_from([(a, b), (b, c), (c, d), (d, e), (e, h)])
    result.add_edges_from([(f, g), (g, h)])
    result.add_edges_from([(h, i), (i, j)])
    return result


def evaluator_82efbd82():
    result = nx.DiGraph()
    a = download_and_verify_file(
        "https://media.cntraveller.com/photos/642aa1ad770beda2d4f5cc22/4:3/w_2664,h_1998,c_limit/Fiji-march2023issue-JackJohns15.jpg",
        "/home/crab/Downloads/raw.jpg",
    )
    b = check_text_in_current_window_name("GNU Image Manipulation Program")
    c = check_file_exist("/home/crab/Pictures/edited.jpg")
    d = is_image_2_brighter(
        "/home/crab/Downloads/raw.jpg", "/home/crab/Pictures/edited.jpg"
    )
    e = verify_background("/home/crab/Pictures/edited.jpg")
    result.add_edges_from([(a, b), (b, c), (c, d), (d, e)])
    return result


def evaluator_515a5467():
    result = nx.DiGraph()
    a = download_and_verify_file(
        "https://media.cntraveller.com/photos/642aa1ad770beda2d4f5cc22/4:3/w_2664,h_1998,c_limit/Fiji-march2023issue-JackJohns15.jpg",
        "/home/crab/Downloads/img_1.jpg",
    )
    b = download_and_verify_file(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Flag_of_Ethiopia.svg/250px-Flag_of_Ethiopia.svg.png",
        "/home/crab/Downloads/img_2.jpg",
    )
    c = check_text_in_current_window_name("GNU Image Manipulation Program")
    d = check_file_exist("/home/crab/Downloads/combined_editing.jpg")
    e = verify_combined_image(
        "/home/crab/Downloads/img_1.jpg",
        "/home/crab/Downloads/img_2.jpg",
        "/home/crab/Downloads/combined_editing.jpg",
        "right",
    )
    f = check_directory_exists("/home/crab/jpg")
    g = verify_files_copied("/home/crab/Downloads", "/home/crab/jpg", "jpg")
    result.add_edges_from([(a, c), (b, c), (c, d), (d, e), (e, f), (f, g)])
    return result


def evaluator_5a1eba49():
    result = nx.DiGraph()
    a = check_text_in_current_window_name("Firefox")
    b = check_contain_input_text("GPU")
    c = is_img_url_in_clipboard()
    d = download_from_clipboard_and_verify_file("/home/crab/Pictures/GPU.png")
    e = check_directory_exists("/home/crab/Pictures/png_files")
    f = verify_files_copied(
        "/home/crab/Pictures", "/home/crab/Pictures/png_files", "png"
    )
    result.add_edges_from([(a, b), (b, c), (c, d), (d, e), (e, f)])
    return result


def evaluator_c347f78a():
    file_path = "/home/crab/assets/content.txt"
    content = "An air quality health advisory is in effect Tuesday for New York City and the lower Hudson Valley, as well as western Connecticut and northern New Jersey, meaning it may not be safe for people with some conditions to be outside long."
    result = nx.DiGraph()
    a = check_current_window_process("gnome-terminal-server")
    b = is_process_open("vim")
    c = ~is_process_open("vim")
    d = check_file_content(file_path, content)
    e = check_contain_input_text("cat " + file_path)
    f = check_submit(content)
    result.add_edges_from([(a, b), (b, c), (c, d), (d, e), (e, f)])
    return result


def evaluator_bf83c176():
    result = nx.DiGraph()

    file_path_1 = "/home/crab/Desktop/waymo.jpg"
    file_path_2 = "/home/crab/Desktop/tesla.png"
    output_path = "/home/crab/Documents/self_driving.pdf"
    # Search for the first image and download it
    a1 = check_text_in_current_window_name("Firefox")
    b1 = check_contain_input_text("Waymo")
    c1 = is_img_url_in_clipboard()
    d1 = download_from_clipboard_and_verify_file(file_path_1)

    # Search for the second image and download it
    a2 = check_text_in_current_window_name("Firefox")
    b2 = check_contain_input_text("Tesla")
    c2 = is_img_url_in_clipboard()
    d2 = download_from_clipboard_and_verify_file(file_path_2)

    # Combine images into a PDF
    e = check_text_in_current_window_name("LibreOffice Impress")
    f = check_file_exist(output_path)
    g = verify_combined_image(file_path_1, file_path_2, output_path, "left")

    # Add edges to form the branches and connections
    result.add_edges_from([(a1, b1), (b1, c1), (c1, d1)])
    result.add_edges_from([(d1, a2), (a2, b2), (b2, c2), (c2, d2)])
    result.add_edges_from([(d2, e), (e, f), (f, g)])

    return result


def evaluator_74bb11dd():
    file_path_1 = "/home/crab/Documents/FR.ods"
    file_path_2 = "/home/crab/Documents/MX.ods"
    result = nx.DiGraph()

    # Search for the first country and save information to an ODS file
    a1 = check_text_in_current_window_name("Wikipedia — Mozilla Firefox")
    b1 = check_text_in_current_window_name("LibreOffice Calc")
    c1 = check_file_exist(file_path_1)
    d1 = verify_country_data_in_ods("France", file_path_1)

    # Search for the second country and save information to an ODS file
    a2 = check_text_in_current_window_name("Wikipedia — Mozilla Firefox")
    b2 = check_text_in_current_window_name("LibreOffice Calc")
    c2 = check_file_exist(file_path_2)
    d2 = verify_country_data_in_ods("Mexico", file_path_2)

    # Create new directory and copy ODS files to it
    e = check_directory_exists("/home/crab/Desktop/country_info")
    f = verify_files_copied(
        "/home/crab/Documents", "/home/crab/Desktop/country_info", "ods"
    )

    # Add edges to form the branches and connections
    result.add_edges_from([(a1, b1), (b1, c1), (c1, d1)])
    result.add_edges_from([(a2, b2), (b2, c2), (c2, d2)])
    result.add_edges_from([(d1, e), (d2, e), (e, f)])

    return result


TEXT_ca79febf = 'The rapid advancement of conversational and chat-based language models has led to remarkable progress in complex task-solving. However, their success heavily relies on human input to guide the conversation, which can be challenging and time-consuming. This paper explores the potential of building scalable techniques to facilitate autonomous cooperation among communicative agents and provide insight into their "cognitive" processes. To address the challenges of achieving autonomous cooperation, we propose a novel communicative agent framework named role-playing. Our approach involves using inception prompting to guide chat agents toward task completion while maintaining consistency with human intentions. We showcase how role-playing can be used to generate conversational data for studying the behaviors and capabilities of chat agents, providing a valuable resource for investigating conversational language models. Our contributions include introducing a novel communicative agent framework, offering a scalable approach for studying the cooperative behaviors and capabilities of multi-agent systems, and open-sourcing our library to support research on communicative agents and beyond. The GitHub repository of this project is made publicly available on: https://github.com/camel-ai/camel.'


def evaluator_ca79febf():
    result = nx.DiGraph()
    a = check_current_package_name("com.google.android.keep")
    b = check_keep_notes_content(TEXT_ca79febf)
    c = check_tap_text("select")
    d = check_tap_text("copy")
    e = check_current_package_name(
        "com.google.android.apps.docs.editors.docs/com.google.android.apps.docs.editors.homescreen.HomescreenActivity"
    )
    f = check_current_package_name(
        "com.google.android.apps.docs.editors.docs/com.google.android.apps.docs.editors.kix.KixEditorActivity"
    )
    g = check_tap_text("paste")
    h = check_current_window_process("firefox")
    i = check_text_in_current_window_name("Google Docs — Mozilla Firefox")
    j = check_text_in_current_window_name(
        "Untitled document - Google Docs — Mozilla Firefox"
    )
    result.add_edges_from([(a, b), (b, c), (c, d), (d, e), (e, f), (f, g), (g, j)])
    result.add_edges_from([(h, i), (i, j)])
    return result


def evaluator_dfabf84c():
    result = nx.DiGraph()
    keyword = "kaust"
    a = check_text_in_current_window_name("Mozilla Firefox")
    b = check_contain_input_text(keyword)
    c = is_img_url_in_clipboard()
    d = download_from_clipboard_and_verify_file("/home/crab/Desktop/download.jpg")
    e = check_current_package_name("com.google.android.keep")
    f = check_contain_input_text(keyword)
    g = check_note_content(keyword)
    result.add_edges_from([(a, b), (b, c), (c, d), (d, g)])
    result.add_edges_from([(b, e), (e, f), (f, g)])
    return result


def evaluator_aab5555e():
    result = nx.DiGraph()
    a = check_current_window_process("gnome-terminal-server")
    b = check_contain_input_text("uname -a")
    d = check_current_package_name("com.google.android.apps.messaging")
    e = check_message_text_box_contain("ubuntu")
    f = check_message_text_box_contain("x86")
    g = check_message_text_box_contain("linux")
    h = check_message_text_box_contain("crab")
    sink = check_message_text_box_empty()
    result.add_edges_from(
        [
            (a, b),
            (b, sink),
            (d, e),
            (d, f),
            (d, g),
            (d, h),
            (e, sink),
            (f, sink),
            (g, sink),
            (h, sink),
        ]
    )
    return result


RESULT_fd0576be = None


@action(env_name="ubuntu")
def get_root_usage() -> str:
    try:
        output = subprocess.check_output(["df", "/"], text=True)
        return output.split("\n")[1].split()[4][:-1]
    except Exception:
        return None


@evaluator(env_name="ubuntu", local=True)
def check_contain_input_text_and_get_df_result(text: str, env) -> bool:
    global RESULT_fd0576be
    RESULT_fd0576be = env._action_endpoint(get_root_usage, parameters={})
    if env.trajectory:
        inputs = [
            params["text"].lower()
            for action_name, params, _ in env.trajectory
            if action_name == "write_text"
        ]
        return any(text.lower() in input_text for input_text in inputs)

    return False


def evaluator_fd0576be():
    result = nx.DiGraph()
    a = check_current_window_process("gnome-terminal-server")
    b = check_contain_input_text_and_get_df_result("df")
    c = check_current_package_name("com.google.android.keep")
    d = check_keep_notes_contain_fd()
    result.add_edges_from([(a, b), (b, d), (c, d)])
    return result


def evaluator_7e08f7d4():
    result = nx.DiGraph()
    a = check_text_in_current_window_name("Mozilla Firefox")
    b = check_contain_input_text(
        "https://farm9.staticflickr.com/8293/7591378270_76059bc1cf_z.jpg"
    )
    c = check_current_package_name("com.android.deskclock.DeskClock")
    d = check_alarm_contains("7:00\u200aAM")
    result.add_edges_from([(a, b), (b, d), (c, d)])
    return result


def evaluator_4957e964():
    result = nx.DiGraph()
    a = check_current_window_process("gnome-terminal-server")
    b = check_contain_input_text("wget")
    c = check_contain_input_text(
        "https://farm8.staticflickr.com/7451/10001676353_fd762e02f0_z.jpg"
    )
    d = check_file_exist("/home/crab/Desktop/download.jpg")
    e = check_text_in_current_window_name("Image Viewer")
    f = check_current_package_name("com.google.android.apps.tasks")
    g = check_google_tasks_name("tennis")
    result.add_edges_from([(a, b), (b, c), (c, d), (d, e), (e, g), (f, g)])
    return result


# Hand-made environment setup guide:
# Ubuntu
# * Make sure the Ubuntu slack login, and the default channel has at least two messages

# Andorid
# * Make sure the first incomplete task in android "Tasks" application is a instruction to change the system to dark mode.
# * Make sure the init page of "Calendar" app is "Day" view. There should be at least one element today.


ubuntu_handmade_tasks = [
    Task(
        id="82efbd82-c941-4be9-9ac0-a495dc629e02",
        description='Download an image file from a given URL "https://media.cntraveller.com/photos/642aa1ad770beda2d4f5cc22/4:3/w_2664,h_1998,c_limit/Fiji-march2023issue-JackJohns15.jpg" to "/home/crab/Downloads/raw.jpg", then use GIMP (GNU Image Manipulation Program) to adjust the brightness of the image from "/home/crab/Downloads/raw.jpg" to be brighter and save the edited file to "/home/crab/Pictures/edited.jpg", and set the adjusted image "/home/crab/Pictures/edited.jpg" as the screen background of the system.',
        evaluator=evaluator_82efbd82(),
    ),
    Task(
        id="515a5467-b7ce-4cad-874d-da894361c1a3",
        description='Download two image files from given URLs "https://media.cntraveller.com/photos/642aa1ad770beda2d4f5cc22/4:3/w_2664,h_1998,c_limit/Fiji-march2023issue-JackJohns15.jpg" and "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Flag_of_Ethiopia.svg/250px-Flag_of_Ethiopia.svg.png" to "/home/crab/Downloads/img_1.jpg" and "/home/crab/Downloads/img_2.jpg", combine the first image ("/home/crab/Downloads/img_1.jpg") with the second image ("/home/crab/Downloads/img_2.jpg") using GIMP (GNU Image Manipulation Program) by placing the first image on the right side of the second image, and save the resulting combined image to "/home/crab/Downloads/combined_editing.jpg". Then, create a new directory "/home/crab/jpg" and copy all files with the specified "jpg" extension from "/home/crab/Downloads" to the newly created directory "/home/crab/jpg".',
        evaluator=evaluator_515a5467(),
    ),
    Task(
        id="5a1eba49-ed2d-4955-a684-32472090a45b",
        description='Use Firefox to search for an image using the keyword "GPU", copy the URL of the found image to the clipboard, download the image file from the URL stored in the clipboard to "/home/crab/Pictures/GPU.png", and create a new directory "/home/crab/Pictures/png_files" to copy all files with the specified "png" extension from "/home/crab/Pictures" to the newly created directory "/home/crab/Pictures/png_files".',
        evaluator=evaluator_5a1eba49(),
    ),
    Task(
        id="c347f78a-4643-43c8-b41e-e437b70a2c5e",
        description='Open a file at "/home/crab/assets/content.txt" using vim in a terminal, write the specified "An air quality health advisory is in effect Tuesday for New York City and the lower Hudson Valley, as well as western Connecticut and northern New Jersey, meaning it may not be safe for people with some conditions to be outside long." to it, then save and exit vim. Print the content of the file by printing it to the command line interface through a terminal, and finally, submit the printed content.',
        evaluator=evaluator_c347f78a(),
    ),
    Task(
        id="bf83c176-fa15-4057-996f-f75be4338c05",
        description='Use Firefox to search for an image using the keyword "Waymo" first, copy the URL of the image to the clipboard, and download the image to "/home/crab/Desktop/waymo.jpg". Then, search for another image using the keyword "Tesla", copy the URL of the image to the clipboard, and download the image to "/home/crab/Desktop/tesla.png". Finally, combine the two images using LibreOffice Impress, placing Image 1 from "/home/crab/Desktop/waymo.jpg" on the left side of Image 2 "/home/crab/Desktop/tesla.png", and save the resulting file in PDF format to "/home/crab/Documents/self_driving.pdf".',
        evaluator=evaluator_bf83c176(),
    ),
    Task(
        id="74bb11dd-89ca-43d0-8edf-fe7b5201ecf7",
        description='Use Firefox to search for information about the country "France" on Wikipedia. Extract the capital city and population, and save this information in an ODS file at "/home/crab/Documents/FR.ods" using LibreOffice Calc. Then, search for information about the country "Mexico" on Wikipedia, extract the capital city and population, and save this information in a separate ODS file at "/home/crab/Documents/MX.ods" using LibreOffice Calc. The format of the file are, first column for the country name, the second for the capital city name, and the third for the population without any header. Finally, create a new directory "/home/crab/Desktop/country_info" and copy all files with the specified "ods" extension from "/home/crab/Documents" to the newly created directory "/home/crab/Desktop/country_info".',
        evaluator=evaluator_74bb11dd(),
    ),
]

corss_environment_tasks = [
    Task(
        id="79832e15-5fd3-43b8-b3e3-66249edfe1db",
        description='Open slack in Ubuntu desktop, summarize the last two messages in current channel, then use "Messages" app in android phone to send the summary to the first contact in the list.',
        evaluator=summarize_ubuntu_evaluator(),
    ),
    Task(
        id="a3476778-e512-40ca-b1c0-d7aab0c7f18b",
        # You must set the first incomplete task to "In Ubuntu, switch the system to dark mode by "Settings" application"
        description='Open "Tasks" app on Android, check the first incomplete task, then perform the task according to its description',
        evaluator=nx.path_graph(
            [
                check_current_package_name("com.google.android.apps.tasks"),
                check_current_window_process("gnome-control-center"),
                check_color_scheme("prefer-dark"),
            ],
            create_using=nx.DiGraph,
        ),
    ),
    Task(
        id="914e6a48-8430-4a68-8328-c4e01db8926e",
        # You must create several tasks in google calendar today's view.
        description='Open "Calendar" app on Android, summarize all schedules today. Then, create a markdown file in Ubuntu at "/home/crab/assets/plan.md" with each event as a checkbox bullet point.',
        evaluator=check_calendar_evaluator(),
    ),
    Task(
        id="97e6f333-bedb-429b-8dd6-1855f99c312d",
        description="Take a photo through Android Camera, then upload it to Google Photos inside Camera App. Use Firefox inside Ubuntu desktop to download the photo to local disk, move it as `/home/crab/assets/photo.jpg`, finally open the photo in GIMP.",
        evaluator=evaluator_97e6f333(),
    ),
    Task(
        id="ca79febf-cae7-4669-8812-d3ec85ee2868",
        description="Open the first note in the Keep Notes app on Android, copy its contents, and paste them into a new document in Google docs. Then, open the newly created document in Firefox on Ubuntu.",
        evaluator=evaluator_ca79febf(),
    ),
    Task(
        id="dfabf84c-d05f-4e25-9f21-ba0f08107bd5",
        description='Use Firefox to search for an image using the keyword "kaust" and copy the URL of the image to the clipboard. Download a file from the URL stored in the clipboard to "/home/crab/Desktop/download.jpg". Then describe this image and save it in the Android Keep Notes app.',
        evaluator=evaluator_dfabf84c(),
    ),
    Task(
        id="aab5555e-4b72-4ebf-816a-59c1da2cec86",
        description="Check the all uname information of the system in Ubuntu, then explain the information to the first contact in the list of the Messages app in Android.",
        evaluator=evaluator_aab5555e(),
    ),
    Task(
        id="fd0576be-8b2c-45ce-b4a2-78659740879b",
        description="Check the current disk usage through command line in Ubuntu, check the root directory usage in percentage and save the information to a note in Keep Notes app in Android.",
        evaluator=evaluator_fd0576be(),
    ),
    Task(
        id="7e08f7d4-9b11-4aec-9b42-6cbde083fb4c",
        description='Use firefox on Ubuntu to openup the image "https://farm9.staticflickr.com/8293/7591378270_76059bc1cf_z.jpg", check the time of the clock in the image, then open the clock app in Android and set an alarm to the same as the image.',
        evaluator=evaluator_7e08f7d4(),
    ),
    Task(
        id="4957e964-5dd5-42f6-9d5d-f6a53a9a5d94",
        description='Use wget to download the image "https://farm8.staticflickr.com/7451/10001676353_fd762e02f0_z.jpg" to /home/crab/Desktop/download.jpg, what does the people in the image do? Create a task in the Tasks app in Android to remind you to do the same thing.',
        evaluator=evaluator_4957e964(),
    ),
]

handmade_tasks = ubuntu_handmade_tasks + corss_environment_tasks

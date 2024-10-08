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

import re

import networkx as nx
from lxml import etree
from lxml.etree import _Element
from networkx import DiGraph, path_graph

from crab import SubTask, evaluator
from crab.actions.android_actions import execute_adb


def get_xml_etree(env) -> _Element | None:
    xml_str = execute_adb("exec-out uiautomator dump /dev/tty", env)
    if "UI hierchary dumped to: /dev/tty" not in xml_str:
        return None
    xml_str = xml_str.removesuffix("UI hierchary dumped to: /dev/tty")
    return etree.fromstring(xml_str.encode("utf-8"))


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
    if mail_node is None:
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
    if not event_nodes:
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

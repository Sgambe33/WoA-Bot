import threading

import cv2
import keyboard
import pyautogui
import numpy as np
import logging
from time import sleep

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Image paths for events
arrival_event_path = "img/arrival_events/arrival_event.png"
cleartoland_path = "img/arrival_events/cleartoland.png"
freestand_path = "img/arrival_events/freestand.png"
selectstand_path = "img/arrival_events/selectstand.png"

departure_event_path = "img/departure_events/departure_event.png"
lineup_path = "img/departure_events/lineup.png"
pushback_path = "img/departure_events/pushback.png"
takeoff_path = "img/departure_events/takeoff.png"

ground_event_path = "img/ground_events/ground_event.png"
handling_event_path = "img/ground_events/handling_event.png"
addcrew_path = "img/ground_events/addcrew.png"
claim_path = "img/ground_events/claim.png"
claimandupgrade_path = "img/ground_events/claimandupgrade.png"
cleanplane_path = "img/ground_events/cleanplane.png"
disembark_path = "img/ground_events/disembark.png"
embark_path = "img/ground_events/embark.png"
finishhandling_path = "img/ground_events/finishhandling.png"
loadcargo_path = "img/ground_events/loadcargo.png"
loadluggage_path = "img/ground_events/loadluggage.png"
refuel_path = "img/ground_events/refuel.png"
starthandling_path = "img/ground_events/starthandling.png"
unloadcargo_path = "img/ground_events/unloadcargo.png"
unloadluggage_path = "img/ground_events/unloadluggage.png"
water_path = "img/ground_events/water.png"
food_path = "img/ground_events/food.png"
wastewater_path = "img/ground_events/wastewater.png"
def take_screenshot():
    """Take a screenshot and convert it to BGR format."""
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    return cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

def move_to_center():
    """Move the pointer to the center of the screen."""
    screen_width, screen_height = pyautogui.size()
    center = (screen_width // 2, screen_height // 2)
    pyautogui.moveTo(center)
    sleep(0.2)

def check_sim(image_path, threshold=0.8):
    """
    Checks if a template image is present on the screen.
    Returns a tuple: (found_flag, center_x, center_y).
    """
    screenshot = take_screenshot()
    template = cv2.imread(image_path)

    if template is None:
        return False, None, None

    # Convert images to grayscale for template matching
    gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    found = max_val >= threshold
    w, h = template.shape[1], template.shape[0]
    center_x = max_loc[0] + w // 2
    center_y = max_loc[1] + h // 2
    return found, center_x, center_y

def click_at(x, y):
    """Move to the given coordinates, click, then move pointer to the center."""
    pyautogui.moveTo(x, y)
    pyautogui.click()
    move_to_center()

def process_arrival_events():
    # Check and click freestand if available
    found_fs, fs_x, fs_y = check_sim(freestand_path)
    if found_fs:
        logging.info("Free stand event detected")
        click_at(fs_x, fs_y)
    # Check and click select stand if available
    found_ss, ss_x, ss_y = check_sim(selectstand_path)
    if found_ss:
        logging.info("Select stand event detected")
        click_at(ss_x, ss_y)
    # Check and click clear to land
    found_ct, ct_x, ct_y = check_sim(cleartoland_path)
    if found_ct:
        logging.info("Clear to land event detected")
        click_at(ct_x, ct_y)

def process_departure_events():
    for event_path, event_name in [
        (pushback_path, "Pushback"),
        (lineup_path, "Lineup"),
        (takeoff_path, "Takeoff")
    ]:
        sleep(0.2)  # Small delay to allow for UI updates
        found_event, ex, ey = check_sim(event_path)
        if found_event:
            logging.info(f"{event_name} event detected")
            click_at(ex, ey)
            return True

def multi_scale_match(image_path, scales=[0.6,0.7,0.8, 0.9, 1.0], threshold=0.8):
    screenshot = take_screenshot()
    template_orig = cv2.imread(image_path)
    if template_orig is None:
        logging.error(f"Template not found: {image_path}")
        return False, None, None

    best_match = (False, 0, None, None, None)  # (found, score, center_x, center_y, scale)
    for scale in scales:
        template = cv2.resize(template_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_screenshot, gray_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_match[1]:
            w, h = template.shape[1], template.shape[0]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            best_match = (max_val >= threshold, max_val, center_x, center_y, scale)

    found, score, cx, cy, used_scale = best_match
    logging.debug(f"{image_path} multi-scale match score: {score:.2f} at scale {used_scale}")
    return found, cx, cy



def process_ground_events():
    # Add crew if available, retry multiple times
    retries = 10
    while retries > 0:
        found_crew, crew_x, crew_y = check_sim(addcrew_path)
        if found_crew:
            logging.info("Add crew event detected")
            click_at(crew_x, crew_y)
        else:
            break
        retries -= 1

    # Start handling event if available
    found_sh, sh_x, sh_y = check_sim(starthandling_path)
    if found_sh:
        logging.info("Start handling event detected")
        click_at(sh_x, sh_y)
        return True

    # Define a list of handling tasks with their image paths and names
    handling_tasks = [
        (claim_path, "Claim"),
        (claimandupgrade_path, "Claim and upgrade"),
        (finishhandling_path, "Finish handling")
    ]
    for task_path, task_name in handling_tasks:
        found_task, task_x, task_y = check_sim(task_path)
        if found_task:
            logging.info(f"{task_name} event detected")
            click_at(task_x, task_y)
            return True


    ground_tasks = [
        (disembark_path, "Disembark"),
        (unloadluggage_path, "Unload luggage"),
        (unloadcargo_path, "Unload cargo"),
        (refuel_path, "Refuel"),
        (cleanplane_path, "Clean plane"),
        (loadcargo_path, "Load cargo"),
        (loadluggage_path, "Load luggage"),
        (water_path, "Water"),
        (food_path, "Food"),
        (wastewater_path, "Wastewater"),
        (embark_path, "Embark")
    ]
    for task_path, task_name in ground_tasks:
        found_task, task_x, task_y = check_sim(task_path)
        if found_task:
            logging.info(f"{task_name} event detected")
            click_at(task_x, task_y)

def bot_logic():
    """Main bot logic to process events."""
    try:
        while True:
            ret, x, y = check_sim("img/event.png")
            if ret:
                logging.info("Event detected")
                click_at(x, y)
                if process_arrival_events(): continue
                if process_departure_events(): continue
                if process_ground_events(): continue
                sleep(0.2)
            sleep(1)
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")

def main():
    logging.info("Starting bot...")
    t1 = threading.Thread(target=bot_logic, daemon=True)
    t1.start()
    try:
        while True:
            if keyboard.is_pressed('f12'):
                print("F12 pressed. Exiting...")
                break
            sleep(0.2)
    except:
        pass

if __name__ == "__main__":
    main()

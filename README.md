# WoA Bot

## Description
WoA bot is a simple Python program capable of automating the gameplay of World of Airports mobile game.
It works only on Android emulators such as BlueStacks.

## Demo
![Demo Video](./.github/readme_assets/demo.mp4)

## Features

- Automatic stand assignment
- Automatic pushback
- Automatic takeoff and lineup
- Automatic landing clearance
- Automatic ground services management
- Automatic rewards claimer and contract upgrader

## How to use

1. Install [BlueStacks](https://www.bluestacks.com/) or any other Android emulator and World of Airports.
2. Clone/Download this repo.
3. Install Python 3.10 or higher.
4. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the script:
   - main.py for better performance but inaccurate results (may result in delayed flights)
   - main2.py "brute force", is less efficient but generates more accurate results.
    ```bash
    python main.py
    ```
6. Make sure the game is running and in a focused window.
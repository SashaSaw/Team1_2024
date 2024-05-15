# Optical flow looming stimuli detection and avoidance for MiRo 

This project presents a ROS node implementation based off of the obstacle avoidance behaviours of locust.

# How to run

To run this project you can simply:

1. Clone this repository
2. Connect to your MiRo
3. Place MiRo in an environment - You must have a 'Looming Object' with a dense pattern for the optical flow looming stimuli detection to work - for my demonstration I used a printed greyscale camoflage pattern
4. Run roscore in a terminal
5. Navigate to directory Team1_2024/COM3528_assignment/scripts
6. run python3 optical_flow_node.py

# Installing the requirements

You should not need to install these requirements. Running python3 optical_flow_node.py should just work.
However if you get errors saying that you do not have some of the packages in the requirements then simply follow the instructions below to install them onto a virtual environment.

1. pip install virtualenv (if you don't already have virtualenv installed)

2. virtualenv <name of env> (to create your new environment)

3. source <name of env>/bin/activate (to enter the environement)

4. pip install -r requirements.txt

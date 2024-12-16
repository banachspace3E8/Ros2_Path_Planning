FROM osrf/ros:foxy-desktop

RUN apt-get update
RUN apt-get install -y git && apt-get install -y python3-pip

RUN source /opt/ros/foxy/setup.bash \
    && cd ~/ \
    && git clone https://github.com/HaiderAbasi/ROS2-Path-Planning-and-Maze-Solving.git

RUN apt-get install -y ros-foxy-joint-state-publisher
RUN apt-get install -y ros-foxy-robot-state-publisher
RUN apt-get install -y ros-foxy-gazebo-*
RUN pip install pygame
RUN pip install setuptools==58.1.0
RUN pip install opencv-contrib-python
RUN pip install numpy
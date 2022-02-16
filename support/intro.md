# The things to do before running particle_to_gmm

## Turtlebot3_simulation (gazebo) setup

1. Copy the `turtlebot3_simulations` folder to `catkin_ws/src`, while merging

2. `catkin_make`

## Turtlebot3_navigation (RViz)

done when you clone `particle_to_gmm` and build

## Maps

copy maps folder to `/home`

## Launching

For campus outdoor map, run the following:

    roslaunch turtlebot3_gazebo turtlebot3_campus2.launch  
    roslaunch particle_to_gmm turtlebot3_navigation_new.launch x:=7.91 y:=-44.55

For corridor indoor map, run the following:

    roslaunch turtlebot3_gazebo turtlebot3_corridor1.launch  
    roslaunch particle_to_gmm turtlebot3_navigation_new.launch


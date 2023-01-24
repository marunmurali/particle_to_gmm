# The things to do before running particle_to_gmm

Note that it's been out of date. 

## Turtlebot3_simulation (gazebo) setup

1. Copy the `turtlebot3_simulations` folder to `catkin_ws/src`, while merging

2. `catkin_make` (maybe)

<s>I don't know but maybe because Turtlebot3_simulation comes from the original repository, when I push the files don't go to this repository. 

I forked the repository and pushed. 
You can pull, merge folders and skip all existing files (cause there are only new files). </s>

Since it's not a full copy of the original repository, I did the changes so that it's no longer a subdirectory. 


## Turtlebot3_navigation (RViz)

done when you clone `particle_to_gmm` and build (if you have cloned turtlebot3_simulation into catkin_ws)

## Launching

### Gazebo and RViz

For campus outdoor map, run the following:

    roslaunch turtlebot3_gazebo turtlebot3_campus2.launch  
    roslaunch particle_to_gmm turtlebot3_navigation_new.launch x:=7.91 y:=-44.55

For corridor indoor map, run the following:

    roslaunch turtlebot3_gazebo turtlebot3_corridor1.launch  
    roslaunch particle_to_gmm turtlebot3_navigation_new.launch

### Particle_to_gmm nodes

Run the following: 

    roslaunch particle_to_gmm my_state_feedback.launch

(Particle_to_gmm and gmm_controller will both launch)
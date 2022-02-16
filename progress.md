# Research project progress note

## 2022/2/15

### What I’ve done

#### Made a relatively robust line following navigation node for Turtlebot3

* Current features:  
    Launched with .launch file  
    Can follow in different directions  
* Current problems:  
    Can’t figure out the mathematical model  
    Can’t let the program run in desired rate  
    Can’t prevent collision  

> It seems that simply the larger the feedback gain is, the quicker it goes towards the target line. 

#### (Trying to) make new map and launch file with “map2gazebo”

* Map2gazebo is only working on Ubuntu 18.04, so I ran it on Lenovo and copied the files  
* Finally got to know the way to load the map files (I don’t know if it is the only)  
* Now we have a much more complicated map available in gazebo (and RViz)! 

### What I am going to do:  

* Building the very detailed state feedback model  
* Upload the files so that you can test on your notebook

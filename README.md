# particle_to_gmm

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->

## General Information

Contains a python file running in ROS that can convert a set of pose messages (such as that are published from an AMCL localization node) into a GMM distribution. The GMM means and covariances are published as ros messages.

A visualisation node is also added to see the gmm as ellipses.


## Technologies Used
- ROS
  - ROS Navigation (for AMCL)
  - rqtplot
- Python
  - Numpy (for eigenvalues and eigenvectors)
  - Scikit (for GMM)
  <!-- - matplotlib -->


## Features
Available features are:
- Pose messages to GMM
- GMM visualizer
- GMM-based control


## Screenshots
/
<!-- ![Example screenshot](./img/screenshot.png) -->
<!-- If you have screenshots you'd like to share, include them here. -->


## Setup
<!-- What are the project requirements/dependencies? Where are they listed? A requirements.txt or a Pipfile.lock file perhaps? Where is it located? -->

### Requirements: 

1. A PC with moderate hardware
2. Ubuntu 20.04 
3. ROS Noetic Ninjemys
4. Turtlebot3 tutorial downloaded

<!-- Proceed to describe how to install / setup one's local environment / get started with the project. -->
For setup, please refer to [intro.md](./support/intro.md)


## Usage
<!-- How does one go about using it?
Provide various use cases and code examples here.

`write-your-code-here` -->

Also, refer to [intro.md](./support/intro.md)

## Project Status
<!-- Project is: _in progress_ / _complete_ / _no longer being worked on_. If you are no longer working on it, provide reasons why. -->

Project is _in progress_ (scheduled until 2023.1). 

For progress, please refer to [the Hackmd note](https://hackmd.io/@lihanjie/rkClIb9k9/edit). 

## Room for Improvement (TODO)
<!-- Include areas you believe need improvement / could be improved. Also add TODOs for future development. -->

- Implementation of RMPC method



## Acknowledgements
<!-- Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to... -->

Great appreciation to Mr.Suzuki and Mr.Okuda. 


## Contact

Created by [@marunmurali](https://github.com/marunmurali) - feel free to contact me!

Being modified by [@AkitaShigure](https://github.com/AkitaShigure). 

<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

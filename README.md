# QuantumGraphs

The current repository contains a package for quantum graphs numerical calculations and a GUI for simulations. Feel free to share your issues / submit new ideas for improvements / contribute to this open-source project.

## Getting started:
You can simulate some quantum graphs properties (sketching a graph with different boundary conditions, compute its eigenvalues and plot its eigenfunctions accordingly, display the secular determinant,...) by using the GUI created for this effect. There are two ways to do so:

#### 1. Via the application:
In this section, you don't need to have Python installed.

You can simulate the program via the application created for the project in the link:
https://drive.google.com/file/d/1HiQhbUaETrxe_C40L2CvYxYkm3zfIfFy/view?usp=sharing

By following the steps:
- Download and extract the zip folder.
- Open the .exe file (the application) directly in the folder.

Make sure the pycache folder as well as the application and the background picture are in the same folder while running the program.

#### 2. Via the file main.py:
##### a. Requirements:
In this section, you do need to have Python installed as well as the following libraries:
 - Numpy
 - math
 - cmath
 - matplotlib
 - mpl_toolkits 
 - pygame
 - tkinter

You can install those libraries with the pip command on your command-line interpreter, as in the following example:
##### pip install pygame

##### b. Running main.py:
Make sure the main.py file is on the same folder as QuantumGraphs.py and background.png.
Run the main.py file with the compiler of your choice.

## How to use the simulation:
#### 1. Main menu:
When you successfully managed to open the GUI (youhou !), you find yourself in front of a menu screen. 
You can display the credits/instructions of the program by clicking on it and clicking on QUIT to return the the main menu.

#### 2. Game state:
Click on GAME to start sketching your graph. You can add a vertex by clicking on any position of the screen. Release the mouse on another position to create another vertex (or choosing an existing one) and connect them (black edges represent the connectivity of the graph). If you want to play with some existing node boundary condition, click on it (blue means Neumann condition while green means Dirichlet condition). By default, the nodes are created with Neumann conditions (blue).

At each moment (including when you started the numerical computations' process), you can choose to go back to the main menu (BACK button) or restart the graph (RESTART button).

#### 3. Numerical computations:
When you finish sketching your graph, click on START. As requested, enter the number N of eigenvalues you want to compute, and click on the SEC_DET button if you want to display the secular determinant of the graph (from 0 to the N-th time the function crosses 0) then click on ENTER.

An axis of eigenvalues (represented by red nodes) appears (with a scale-relative real spacing between the values). Move the mouse on a node to display the respective eigenvalue. Click on a node to add it (turning yellow) / remove it (turning red) from the list of eigenvalues you are interested in plotting the respective eigenfunctions.
When you finish the choosing phase, click on ENTER. The eigenfunctions' subplots according to the eigenvalues chosen appear. TADA ! You can now continue sketching (going back to Game state).

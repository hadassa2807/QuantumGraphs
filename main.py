"""
This file presents the GUI for using the quantum graph package.
"""

# Needed libraries:
import pygame, sys
from pygame.locals import *
import math
import QuantumGraphs
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from QuantumGraphs import QuantumGraph

# Tupples
LEFT = 1
START = 0
INST = 1
CREDIT = 2
GAME = 3
PREDRAW = 4
DRAW = 5

# Colors
WHITE = (255, 255, 255, 255)
GREY = (70, 70, 70, 255)
LIGHT_GREY = (150, 150, 150, 255)
BLACK = (0, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
RED = (255, 0, 0, 255)

# Sizes
screen_width = 800
screen_height = 500

'''
The following class aims to represent the object "node", with characteristics: position 
within the grid (pos), its number (number), and the nodes connected to it (neighbors)
'''
class Node:
    # Initializing the parameters of the object
    def __init__(self, pos, number):
        self.pos = pos
        self.number = number
        self.neighbors = []
    # The function aims to add the the neigbors' list of the node, a newly connected node
    def connect_nodes(self, node):
        self.neighbors.append(node)


'''
The following function returns closest tenth to a number (x)
'''
def roundup(x):
    diff = x - (10*int(x/10))
    if diff >= 5:
        return 10*int(x/10) + 10
    else:
        return 10*int(x/10)

'''
The following function aims to create a grid of size (w,h) starting at position 
(startX, startY)
'''
def draw_grid(screen, w, h, startX, startY):
    pygame.draw.rect(screen, GREY, [startX, startY, w, h])
    WIDTH = 9
    for column in range(int(startX/10), int(startX/10 + w/10)):
        for row in range(int(startY/10), int(startY/10 + h/10)):
            color = WHITE
            pygame.draw.rect(screen, color, [(1 + WIDTH) * column, (1 + WIDTH) * row, WIDTH, WIDTH])

'''
The following function aims to return a node according to a given position 
(if such a node exists)
'''
def get_node(pos, Nodes):
    for item in Nodes:
        if (item.pos == pos):
            return item
    return None # Should not get here

'''
The following function aims to return the closest legal position for a node's
place (the closest point on the grid)
'''
def get_legal_pos(mouse_pos_beg):
    X = roundup(mouse_pos_beg[0])
    Y = roundup(mouse_pos_beg[1])
    pos = (X, Y)
    return pos

'''
The following function aims to create a new node and to return it
'''
def create_node(screen, mouse_pos_beg, Nodes, Coord_list, d_vertices, releasing):
     # We want the position of the node to be on the grid
     pos = get_legal_pos(mouse_pos_beg)
     X = pos[0]
     Y = pos[1]
     # We are on an existing node, if and only if at least one of the following positions is a blue pixel
     col_pos1 = (X+2, Y)
     col_pos2 = (X, Y+2)
     # If the node does not exist yet, we build it and add it to the list of nodes and of their positions
     if screen.get_at(col_pos1) != BLUE and screen.get_at(col_pos2) != BLUE:
        # If the node is neither blue nor green, it does not exist yet, therefore we build it (blue vertex
        # with Neumann condition) and add it to the list of nodes and their positions.
        if screen.get_at(col_pos1) != GREEN and screen.get_at(col_pos2) != GREEN:
            pygame.draw.circle(screen, BLUE, pos, 3)
            Start_node = Node(pos, len(Nodes))
            Nodes.append(Start_node)
            Coord_list.append([pos[0]/10, pos[1]/10])
            d_vertices.append(0)
        # If the node is green, we turn it blue and convert it into a Neumann condition 
        else:
            Start_node = get_node(pos, Nodes)
            if releasing == False:
                pygame.draw.circle(screen, BLUE, pos, 3)
                d_vertices[Start_node.number] = 0
     # If the node is blue, we turn it green and convert it into a Dirichlet condition 
     else:
        Start_node = get_node(pos, Nodes)
        if releasing == False:
            pygame.draw.circle(screen, GREEN, pos, 3)
            d_vertices[Start_node.number] = 1
        
     return Start_node

'''
The following function aims to use the relations between each node (connected or not) of the built graph
to create its adjacency matrix
'''
def create_adj_mat(Nodes):
    adj_mat = np.zeros( (len(Nodes),len(Nodes)) )
    for item in Nodes:
        for node_num in item.neighbors:
            # We want to assure the symmetry of the matrix
            adj_mat[item.number][node_num.number] = 1
            adj_mat[node_num.number][item.number] = 1
    print(adj_mat)
    return adj_mat

'''
The following function aims to add text at position (X,Y), in color black by default
but given to choice
'''
def add_text(screen, text, X, Y, color=(0,0,0), size=23):
    myfont = pygame.font.SysFont('Comic Sans MS', size)
    textsurface = myfont.render(str(text), False, color)
    screen.blit(textsurface,(X,Y))

'''
The following function aims to get (and print to the interface) the number of eigenvalue
the user wants to use for eigenfunctions' finding.
'''
def get_num_eigval(key_name, screen, Num_eigval, start, start_pressed, spacing):
    # To be sure we get a number or a letter, we check the key pressed is of length 1
    if (len(key_name)==1): 
        key_c = ord(key_name) 
    else:  
        key_c = None
    # We check if the pressed key is a number from 0 to 10
    if (key_c != None and key_c>=48 and key_c<=57):
        # We update the value of the eigenvalue's number according to the entered digit
        Num = int(key_name)
        if Num_eigval==None: 
            Num_eigval = Num
        else: 
            Num_eigval = Num_eigval * (10**spacing) + Num
            spacing+=1
        add_text(screen, key_name, 430+11*(spacing-1), 0)
    # If we pressed "Enter", we want to start drawing the eigenfunctions and use the written
    # number to be the eigenvalue's number we want to use for eigenfunctions' drawing.
    elif key_name=="return":
        if (Num_eigval==None): Num_eigval = 1
        start = True
    # If we pressed anything else, starting is not relevant anymore
    else:
        print("Not an integer")
        start_pressed == False
    return Num_eigval, start, start_pressed, spacing

'''
The following function aims to draw the eigenfunctions of the graph according to the Num_eigval-th
eigenvalue
'''
def start_QG(screen, Coord_list, Num_eigval, Nodes, screen_width, draw_sec_det, sd, DVertices):
    # Remove the sec det button and the eigenvalues' number question
    draw_grid(screen, sd.width+5, sd.height+5, sd.left, sd.top)
    draw_grid(screen, int(screen_width), 40, 0, 0)
    # Parameters needed from the quantum graph
    adj_mat = create_adj_mat(Nodes)
    
    # If a problem occurs while creating the quantum graph, we catch it to report to the user why
    # the process failed.
    try:
        QG = QuantumGraph(len(Nodes), adj_mat, Coord_list,Num_eigval,DVertices)
        ev = QG.EigenVal
        # Parameters for creating the line where the eigenvalues will be represented
        ev_placed = []
        last_val = len(ev)-2
        dist_pix = screen_width - 40
        dist_num = ev[last_val]-ev[0]
        number = 0
        # Eigenvalues' axis creation
        pygame.draw.line(screen, BLACK, (20,50), (dist_pix+20,50), 2)
        for i in range(0, last_val+1):
            if dist_num != 0: pos = (roundup(20 + (ev[i]-ev[0])*dist_pix/dist_num), 50)
            else: pos = (20, 50)
            pygame.draw.circle(screen, RED, pos, 3)
            # Writing the number of the eigenvalue upon the corresponding node
            if last_val < 10: rate = 1
            else: rate = math.floor((last_val)/10)
            if i == 0 or i == last_val or i % rate == 0:
                add_text(screen, str(i),pos[0]-5,23, BLACK, 20)
            eigval = Node(pos, number)
            ev_placed.append(eigval)
            number+=1
        # If the user chose to display the secular determinant, we listen to him
        if (draw_sec_det == True): QG.SecularDetPloting()
        return QG, ev_placed
    # Procedure while catching an exception (ending with exiting the program)
    except Exception as ex:
        # Infinite loop exception. Main cause: infinite loop while searching the eigenvalues.
        if type(ex).__name__ == "InfiniteLoop":
            mess = "Oops ! An infinite loop has been generated.\n\nPossible reason: a single cycle has been drawn.\n\nThe program will now exit."
        # Division by zero exception. Main cause: Dirichlet condition on illegal vertex, causing the S matrix to be singular (determinant equals to zero).
        elif type(ex).__name__ == "ZeroDivisionError": 
            mess = "Oops ! A division by zero occured.\n\nPossible reason: a vertex with a Dirichlet condition is not legal.\n\nThe program will now exit."
        # Other exception. Should not reach it.
        else: 
            print(type(ex).__name__) 
            mess = "Oops ! An error occured.\n\nThe program will now exit."
        Tk().wm_withdraw() #to hide the main window
        messagebox.showinfo('Exception box', mess)
        pygame.quit()
        sys.exit()
        return None, None
    

'''
The following function aims to create a new button
'''
def create_button(screen, button, text, color=LIGHT_GREY):
    pygame.draw.rect(screen, color, button)
    add_text(screen, text, button.left, button.top, size=20)

'''
The following function aims to display the original state of the drawing board
(grid with START and RESTART buttons, without any graph)
'''
def init_state(screen, start_button, restart_button, back_button):
    # Setting some default parameters
    Nodes = []
    number = 0
    Coord_list = []
    d_vertices = []
    start_pressed = False
    start = False

    # Setting platform and background
    screen=pygame.display.set_mode([screen_width,screen_height])
    screen.fill(GREY)
    draw_grid(screen, int(screen_width), int(screen_height), 0, 0)
    pygame.display.set_caption("Quantum Graph")
    create_button(screen, start_button,"START")
    create_button(screen, restart_button,"RESTART")
    create_button(screen, back_button,"BACK")
    return Nodes, number, Coord_list, start_pressed, start, d_vertices

"""
The following function aims to display the instructions of the GUI (how to use it).
"""
def display_instr(screen):
    num = 0
    step = 35
    NL = 15
    add_text(screen, "Welcome to Quantum Graph drawer !", 0, 0, WHITE)
    add_text(screen, "-> To start, click on GAME in the menu platform", 0, step+NL, WHITE)
    add_text(screen, "   To draw a node, click on any point of the grid", 0, 2*step+NL, WHITE)
    add_text(screen, "   To connect it to another node or to a new one,", 0, 3*step+NL, WHITE)
    add_text(screen, "   release the mouse from it to the place of the wanted node.", 0, 4*step+NL, WHITE)
    add_text(screen, "   Add a Dirichlet condition on the vertex of your choice by clicking on it.", 0, 5*step+NL, WHITE)
    add_text(screen, "-> When your graph is ready, click on START;", 0, 6*step+2*NL, WHITE)
    add_text(screen, "   Enter the number of eigenvalues you want to compute.", 0, 7*step+2*NL, WHITE)
    add_text(screen, "   If you want to draw the secular determinant click on SEC DET.", 0, 8*step+2*NL, WHITE)
    add_text(screen, "   Press ENTER and click on the eigenvalue you want to compute", 0, 9*step+2*NL, WHITE)
    add_text(screen, "   Now appear the eigenfunctions drawn upon your graph", 0, 10*step+2*NL, WHITE)
    add_text(screen, "-> Continue or draw a new graph by clicking on RESTART", 0, 11*step+3*NL, WHITE)

"""
The following function aims to display the credits of the project.
"""
def display_credit(screen):
    num = 0
    step = 40
    NL = 10
    add_text(screen, "CREDITS: ", 0, 0, WHITE)
    add_text(screen, "Supervisor: Pr. Rami Band", 0, step+NL, WHITE)
    add_text(screen, "Programmer: Hadassa Malka", 0, 2*step+NL, WHITE)
    add_text(screen, "Also contributed: Gilad Sofer, Salomon Malka", 0, 3*step+NL, WHITE)

"""
The following function aims to control the actions to do in the displays (credit and instructions) screens.
"""
def display_state(screen, quit_button, screen_state, background_image):
    screen.blit(background_image, [0, 0])
    # Choosing the display state
    if screen_state == INST: display_instr(screen)
    else: display_credit(screen)
    create_button(screen, quit_button, "QUIT", WHITE)
    while True:
       for event in pygame.event.get():
           mouse_pos = pygame.mouse.get_pos()
           # Exiting the program.
           if event.type == QUIT:
                pygame.quit()
                sys.exit()
           # Exiting display state
           if quit_button.collidepoint(mouse_pos) == True:
                create_button(screen, quit_button, "QUIT", LIGHT_GREY)
                if (event.type == MOUSEBUTTONDOWN):
                    return START
           # Creating the exiting display state button
           else: create_button(screen, quit_button, "QUIT", WHITE)
       pygame.display.update()
    return screen_state

"""
The following function aims to display the introduction screen of the program.
"""
def menu_game(screen, instr_button, game_button, credit_button, background_image):
    screen.blit(background_image, [0, 0])
    add_text(screen, "Welcome to Quantum Graph drawer !", 30, 30, WHITE, 30)
    create_button(screen, instr_button, "INSTRUCTIONS", WHITE)
    create_button(screen, game_button, "GAME", WHITE)  
    create_button(screen, credit_button, "CREDIT", WHITE) 

"""
The following function aims to control the possible states of the introduction screen of the program,
according to the user's actions.
"""
def start_state(screen, instr_button, game_button, credit_button, screen_state, background_image):
    menu_game(screen, instr_button, game_button, credit_button, background_image)
    while screen_state == START:
       for event in pygame.event.get():
           mouse_pos = pygame.mouse.get_pos()
           # Exiting the program.
           if event.type == QUIT:
                pygame.quit()
                sys.exit()
           # If the mouse collides with the buttons, the button becomes darker. 
           # If we click on the button, we change the state of the screen accordingly and return it.
           if instr_button.collidepoint(mouse_pos) == True:
                create_button(screen, instr_button, "INSTRUCTIONS", LIGHT_GREY)
                if (event.type == MOUSEBUTTONDOWN):
                   return INST
           else: create_button(screen, instr_button, "INSTRUCTIONS", WHITE)
           if credit_button.collidepoint(mouse_pos) == True:
                create_button(screen, credit_button, "CREDIT", LIGHT_GREY) 
                if (event.type == MOUSEBUTTONDOWN):
                   return CREDIT
           else: create_button(screen, credit_button, "CREDIT", WHITE) 
           if game_button.collidepoint(mouse_pos) == True:
                create_button(screen, game_button, "GAME", LIGHT_GREY)  
                if (event.type == MOUSEBUTTONDOWN):
                   return GAME
           else: create_button(screen, game_button, "GAME", WHITE)  
       pygame.display.update()

    return screen_state

'''
Main function
'''
def main():
    pygame.init()
    # Setting introduction screen buttons
    row = int(screen_height / 2)
    col = int(screen_width / 2 - 100)
    instr_button = Rect(col, row, 170, 25)
    game_button = Rect(col, row - 50, 65, 25)
    quit_button = Rect(screen_width - 65, 0, 65, 25)
    credit_button = Rect(col, row + 50, 75, 25)
    # Setting introduction screen buttons
    row = screen_height - 50
    col = int(screen_width / 2 - 50)
    start_button = Rect(col, row, 75, 25)
    SecDet_button = Rect(col-110, row, 85, 25)
    restart_button = Rect(col+100, row, 95, 25)
    back_button = Rect(screen_width - 60, row, 55, 25)
    # Setting initial conditions and state
    restart = draw_sec_det = back = False 
    start = start_pressed = start2 = start3 = False
    screen_state = START
    Start_node = End_node = None
    # Displaying introduction screen
    pygame.display.set_caption("Quantum Graph")
    screen=pygame.display.set_mode([screen_width,screen_height])
    background_image = pygame.image.load("background.jpg").convert()
    menu_game(screen, instr_button, game_button, credit_button, background_image)      
    
    # The game
    while True:
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            starting = start or start_pressed or start2 or start3
        	# Exiting the program
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            # Actions according to the state of the screen
            elif screen_state == START:
                screen_state = start_state(screen, instr_button, game_button, credit_button, screen_state, background_image)
            elif screen_state == INST:
                screen_state = display_state(screen, quit_button, screen_state, background_image)
            elif screen_state == CREDIT:
                screen_state = display_state(screen, quit_button, screen_state, background_image)
            elif screen_state == GAME:
                Nodes, number, Coord_list, start_pressed, start, d_vertices = init_state(screen, start_button, restart_button, back_button)
                screen_state = PREDRAW
            # Actions while pressing the mouse button down in the GAME state
            if (event.type == MOUSEBUTTONDOWN and event.button == LEFT and screen_state >= GAME):
                    mouse_pos_beg = pygame.mouse.get_pos()
                    # If we press START (while we are not already in the starting state and the graph is not empty),
                    # we start the starting process by asking the user to enter the number of eigenvalues he wants to compute.
                    if starting == False and start_button.collidepoint(mouse_pos_beg) == True: 
                        if Nodes != []:
                            add_text(screen, "Enter the requested eigenvalue number: ", 0, 0)
                            create_button(screen, SecDet_button, "SEC DET", LIGHT_GREY)
                            Num_eigval = None
                            start_pressed = True
                            spacing = 1
                    # Pressing the restart button clears the current graph
                    elif restart_button.collidepoint(mouse_pos_beg) == True:
                        restart = True 
                        start = start2 = start3 = False
                        Nodes, number, Coord_list, start_pressed, start, d_vertices = init_state(screen, start_button, restart_button, back_button)
                    # Pressing the back button brings the user to the introduction screen (clearing the graph in the same time)
                    elif back_button.collidepoint(mouse_pos_beg) == True:
                        back = True 
                        screen_state = START
                    # Else, we draw a node at the position of the mouse
                    elif starting == False:
                        back = False
                        restart = False
                        start_pressed = False
                        releasing = True
                        Start_node = create_node(screen, mouse_pos_beg, Nodes, Coord_list, d_vertices, releasing)
                    # Handling issues of drawing a node too quickly (transition state after introduction state)
                    if screen_state == PREDRAW:
                        Nodes, number, Coord_list, start_pressed, start, d_vertices = init_state(screen, start_button, restart_button, back_button)
                        screen_state = DRAW
                        first_node = True

            # If we clicked on start + wrote a number + pressed 'Enter', we start initializing a quantum graph object.
            # We ask the user to enter a number of eigenvalues he wants to compute, and draw an axis containing them.
            if start == True:
                QG, eigvals = start_QG(screen, Coord_list, Num_eigval, Nodes, screen_width,draw_sec_det, SecDet_button, d_vertices)
                start_pressed = False
                start = False
                start2 = True
                # If the quantum graph has not been computed (exception caught), we can continue drawing or modyfing the graph. 
                if QG == None:
                    start3 = True
                    start2 = False
            # After computing the axis with eigenvalues, we can go through one by placing the mouse on it (then we display the value 
            # of that eigenvalue) or clicking on it to start computing the eigenfunctions corresponding upon the graph. 
            elif start2 == True:
                mouse_pos_beg = pygame.mouse.get_pos()
                pos = get_legal_pos(mouse_pos_beg)
                eigval = get_node(pos, eigvals)
                if (eigval != None):
                    text = "eigenvalue = " + str(QG.EigenVal[eigval.number])
                    add_text(screen, text, int(screen_width/3), 0, size=20)
                    if event.type == MOUSEBUTTONDOWN:
                        start2 = False
                        start3 = True
                # If the mouse is not on an eigenvalue node, we don't display any value of eigenvalue.
                else: 
                    draw_grid(screen, int(screen_width), 30, 0, 0)
            # Computing the eigenfunctions and going back to drawing state (to continue drawing).
            elif start3 == True:
                 if QG != None: QG.EigFunc3D_aux(eigval.number)
                 draw_grid(screen, int(screen_width), 60, 0, 0)
                 draw_sec_det = False
                 start3 = False

            # Request of eigenvalue's number for the current drawn graph
            elif (start_pressed == True):
                # Getting a character from keyboard as input to complete an eigenvalues' number.
                if (event.type == pygame.KEYDOWN):
                    key_name = pygame.key.name(event.key) 
                    Num_eigval, start, start_pressed, spacing = get_num_eigval(key_name, screen, Num_eigval, start, start_pressed, spacing)
                # Choosing to display or not the secular determinant.
                elif (event.type == MOUSEBUTTONDOWN):
                    mouse_pos_beg = pygame.mouse.get_pos()
                    if SecDet_button.collidepoint(mouse_pos_beg) == True:
                        if (draw_sec_det == False): 
                            create_button(screen, SecDet_button, "SEC DET", GREEN)
                            draw_sec_det = True
                        else:
                            create_button(screen, SecDet_button, "SEC DET", LIGHT_GREY)
                            draw_sec_det = False

            # If we release the left button (mouse), we create a dot and connect it to the one created by pressing the button
            # (in the case we did not press the start button).
            elif event.type == MOUSEBUTTONUP and event.button == LEFT and starting == False:
                if screen_state == PREDRAW: restart = True
                elif (restart == False):
                    mouse_pos_end = pygame.mouse.get_pos()
                    mouse_pos_end = get_legal_pos(mouse_pos_end)
                    # We allow the node to change its color when the releasing bool is true, meaning it is a node already drawn 
                    # that we just pressed.
                    if Start_node != None and Start_node.pos == mouse_pos_end: releasing = False
                    else: releasing = True
                    End_node = create_node(screen, mouse_pos_end, Nodes, Coord_list, d_vertices, releasing)
                    if Start_node != None and Start_node.pos != End_node.pos:
                        Start_node.connect_nodes(End_node)
                        End_node.connect_nodes(Start_node)
                        pygame.draw.line(screen, BLACK, Start_node.pos, End_node.pos, 2)
                    if first_node == True: first_node = False
                    else: Start_node = None
                
        pygame.display.update()

# Running the main function
if __name__ == "__main__":
    main()
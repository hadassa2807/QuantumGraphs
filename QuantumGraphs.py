"""
This file presents the quantum graph class.
"""

# Needed libraries:
import inspect
import os
import sys
import importlib
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from random import randint

"""
This class represents the Infinite loop exception (we raise if the eigenvalues finder function enters an infinite loop)
"""
class InfiniteLoop(Exception):
    pass

"""
The quantum graph class
"""
class QuantumGraph:
    """
    The constructor aims to initialize every characteristics of the quantum graph: the number of nodes, the adjacency matrix,
    the number of edges from each vertex array, the matrix of the graph's characteristics, the S matrix, the minimal length from the
    array of length, and the eigenvalues of the graph. We can also initialize a specific number of eigenvalues we want to compute and
    an array of vertices with Dirichlet conditions.
    """
    def __init__(self,Nb_nodes,A,coord_list,Num_eigenval=None, DVertices=None,min = None,max = None):
        self.NbNodes = Nb_nodes
        self.adj_mat = A
        self.vertices = coord_list
        self.num_eigval = Num_eigenval
        if (DVertices==None):
            self.DCondition = np.zeros(Nb_nodes, dtype=int)
        else:
            self.DCondition=DVertices
        self.Nb_edges = self.NbOutingEdges()
        self.TotEdges = self.Nb_edges[Nb_nodes]
        self.transfer_mat = self.TM_simple()
        self.graph_characteristics = self.def_of_edges()
        self.SMatrix = self.S_mat(self.DCondition)
        self.minL = self.min_L()
        if min == None: self.startP = 0
        else: self.startP = min
        if max == None: self.endP = np.pi/self.minL
        else: self.endP = max
        if (Num_eigenval!=None): 
            self.EigenVal = self.k_eigenval_finder(Num_eigenval)
        else:
            self.EigenVal = self.Eigenvals_finder()


    # Aims to convert a 2D array into a matrix 
    def load(self,table):
        if len(table) != self.NbNodes or len(table[0]) != self.NbNodes:
            return ("error : Bad dimensions")
        for i in range(self.NbNodes):
            for j in range(self.NbNodes):
                self.adj_mat[i][j]=table[i][j]

    # Gets an item from the matrix
    def __getitem__(self,index):
        return self.adj_mat[index]

    # Sets a new value at place index into the matrix
    def __setitem__(self,index,value):
        self.adj_mat[index]=value

    # Represents the matrix in rows and columns
    def __repr__(self):
        repres=""
        for i in range(self.NbNodes):
            repres=repres+str(self.adj_mat[i])
            if i != self.NbNodes-1:
                repres=repres+"\n"
        return repres

    """
    The following method aims to create an array which contains, at each position <vertex>,
    the number of edges which are going out from the vertex.
    """
    def NbOutingEdges(self):
        x = 0
        y = 0
        a  = []
        # We count the number of edges (i,j) going out from each vertex (if the value of the adjacency matrix is positive at (i,j))
        for i in range(0,self.NbNodes):
            for j in range(0,self.NbNodes):
                if self.adj_mat[i][j] > 0:
                    x = x+1
            y = y + x
            a.append(x)
            x = 0
        # We add the total number of edges in the graph at the end of the array.
        a.append(y/2)
        return a

    """
    The transfer matrix; 
    The member(i,j) represents the numero of the edge between the node i and the node j;
    If i and j are not connected, the value of the matrix at (i,j) is zero;
    The diagonal of the matrix is filled with zeros. 
    The edges are entered in a "lexicographic" order according to the vertices' array given
    """
    def TM_simple(self):
        counter = 1
        TM = np.zeros( (self.NbNodes, self.NbNodes) )
        for i in range(0, self.NbNodes):
            for j in range(i, self.NbNodes):
                if (self.adj_mat[i][j] > 0):
                    TM[i][j] = int(counter)
                    TM[j][i] = TM[i][j]
                    counter+=1
                else:
                    TM[i][j] = 0
                    TM[j][i] = TM[i][j]
        return TM

    """
    The calculation of edges' lengths with given coordinates
    """
    def lengths_calc(self, x0, y0, x1, y1):
        return np.sqrt(((y1-y0)**2)+((x1-x0)**2))


    """
    The following method aims to create the matrix of characteristics of the edges of the graph, while: 
    The first row of the matrix represents the starting vertex of the edge numero column+1;
    The second row of the matrix represents the ending vertex of the edge numero column+1;
    The third row of the matrix represents the length of the edge numero column+1;
    For each i even, the i-th column represents the charactersitic of an edge and the column i+1 the invert of this edge
    """
    def def_of_edges(self):
        Nb_tot = int(self.TotEdges)
        GraphDetails = np.zeros( (3, 2*Nb_tot) )
        for i in range(0,self.NbNodes):
           for j in range(i,self.NbNodes):
               if self.transfer_mat[i][j] > 0:
                   ind = int(2*(self.transfer_mat[i][j]-1))
                   # starting vertex of the edge numero column+1
                   GraphDetails[0][ind] = i
                   GraphDetails[0][ind+1] = j
                   # ending vertex of the edge numero column+1
                   GraphDetails[1][ind] = j
                   GraphDetails[1][ind+1] = i
                   # length of the edge numero column+1
                   x0 = self.vertices[int(GraphDetails[0][ind])][0]
                   y0 = self.vertices[int(GraphDetails[0][ind])][1]
                   x1 = self.vertices[int(GraphDetails[1][ind])][0]
                   y1 = self.vertices[int(GraphDetails[1][ind])][1]
                   GraphDetails[2][ind] = self.lengths_calc(x0, y0, x1, y1)
                   GraphDetails[2][ind+1] = GraphDetails[2][ind]
        return GraphDetails

    """
    This method returns the minimal length of all the edges of the graph.
    """
    def min_L(self):
        minLengths = self.graph_characteristics[2][0]
        for num in range(1,int(self.Nb_edges[self.NbNodes]-1)):
            if self.graph_characteristics[2][num]<minLengths:
                minLengths = self.graph_characteristics[2][num]
        return minLengths

    """
    The S matrix of the quantum graph;
    """
    def S_mat(self, dv):
        Nb_tot = int(self.Nb_edges[self.NbNodes]) 
        # Initialize the matrix to be of dimension which is number of (directed) edges
        S = np.zeros( (2*Nb_tot, 2*Nb_tot) )
        for i in range(0,2*Nb_tot):
           for j in range(0,2*Nb_tot):
               # If both edges i and j are the inverse of each other then we set the value of the S matrix at (i,j) to 2/dv-1 
               if ((self.graph_characteristics[1][i] == self.graph_characteristics[0][j]) and (self.graph_characteristics[0][i] == self.graph_characteristics[1][j])):
                   # Dirichlet condition -> we set the value of the S matrix at (i,j) to -1
                   if (dv[int(self.graph_characteristics[0][j])]==1): 
                       S[i][j]=-1
                   else:
                       S[i][j]=(2/self.Nb_edges[int(self.graph_characteristics[1][i])])-1
               # Otherwise, if the edges are adjacent, then we set the value of the S matrix at (i,j) to 2/dv
               elif self.graph_characteristics[1][i] == self.graph_characteristics[0][j]:
                   if (dv[int(self.graph_characteristics[0][j])]==0):
                        S[i][j]=2/self.Nb_edges[int(self.graph_characteristics[1][i])]
                   # Dirichlet condition -> we set the value of the S matrix at (i,j) to 0
                   else: 
                       S[i][j]=0
        return S

    """
    In the following method secular_det we are going to calculate the secular determinant. 
    We start by finding the complex one complex_det, and as seen in the formula (63),
    we find the real function R_sec_det which has the same zeros k as complex_det,
    as they are the sacred eigenvalues we are looking for.
    """
    def secular_det(self, k):
        Nb_tot = int(self.Nb_edges[self.NbNodes])
        # Creation of the matrix D(k)
        D = np.zeros( (2*Nb_tot, 2*Nb_tot), dtype=complex )
        for row in range(0, 2*Nb_tot):
            for col in range(0, 2*Nb_tot):
                if (row == col):
                   D[row][col] = cmath.exp(1j*self.graph_characteristics[2][row]*k)
        # Calculation of the secular determinant of I - S*D(k)
        I = np.identity(2*Nb_tot)
        A = I - np.matmul(self.SMatrix,D)
        complex_det = np.linalg.det(A)

        # Calculation of the real part of the secular determinant
        S_det = np.linalg.det(self.SMatrix)
        if (S_det < 0):
            S_det_sqrt = math.sqrt(abs(S_det))*1j
        else:
            S_det_sqrt = math.sqrt(abs(S_det))
        Sum_L = 0
        for num in range(0, 2*Nb_tot, 2):
            Sum_L += self.graph_characteristics[2][num]
        R_sec_det = (cmath.exp(-1j*Sum_L*k)/S_det_sqrt)*complex_det
        return R_sec_det

    """
    The following method gets an interval (a,b) at which we want to search, and a number of iterations (N) to control 
    the ratio precision (running time of the research) and returns the value between the numbers a and b where the secular_det method
    crosses a zero (if there is such a value)
    """
    def bisection(self,a,b,N):
        # Failure control of the algorithm.
        if self.secular_det(a)*self.secular_det(b) >= 0:
            print("Bisection method fails.")
            print("reason of failure : ", self.secular_det(a), self.secular_det(b))
            return None
        # We set the starting point (a_n) and end point (b_n) of the interval
        a_n = a
        b_n = b
        # Now we go over many samples of the function to look for zeros
        for n in range(1,N+1):
            # We set a middle point m_n (to be updated at each step of the loop)
            m_n = (a_n + b_n)/2
            f_m_n = self.secular_det(m_n)
            # We check if m_n is correspunding to a zero (with a precision of 10^-6) and return it in that case.
            if abs(f_m_n) <= 10**-6 :
                return m_n
            # We check if m_n is at the right side of the wanted zero; 
            # if so, the interval we look into is now (m_n, b_n).
            elif self.secular_det(b_n)*f_m_n < 0:
                a_n = m_n
                b_n = b_n
            # We check if m_n is at the left side of the wanted zero; 
            # if so, the interval we look into is now (a_n, m_n).
            elif self.secular_det(a_n)*f_m_n < 0:
                a_n = a_n
                b_n = m_n
            # Should not reach that point (failure of the method).
            else:
                """print("Bisection method fails.")"""
                return None
        # In case of failure after the loop, we print the latest interval and the function at its middle.
        print("out of the loop", a_n, b_n, f_m_n)
        # We return the middle of the latest interval.
        return (a_n + b_n)/2

    """
    The following method aims to find the roots of the secular determinant; it runs through a whole period
    (max_period) step by step (step) and checks (with bisection method) if the current value is a root of the secular determinant;
    if so, we update the eigenvalues array (eigenval) with a new member (the root) and continue the search.
    """
    def Eigenvals_finder(self):
        # Setting parameter counter to count the number of eigenvalues found.
        eigenval = []
        eigenval.append(0)
        step = 0.001
        counter = 0
        for num in np.arange(self.startP, self.endP, step):
            x1 = num
            x2 = (num + step)
            if ((self.secular_det(x1) < 0 and self.secular_det(x2) > 0) or (self.secular_det(x1) > 0 and self.secular_det(x2) < 0)):
                if (self.bisection(x1, x2, 100) != None):
                    eigenval.append(self.bisection(x1, x2, 100))
                    counter+=1
        # Printing eigenvalues
        for num in range(0,counter):
            print("root of eigenvalue k number ", num, " = ",  eigenval[num])
        print("\n")
        return eigenval

    """
    The following method aims to find the first k roots of the secular determinant; it runs until k roots are found,
    step by step (step) and checks (with bisection method) if the current value is a root of the secular determinant;
    if so, we update the eigenvalues array (eigenval) with a new member (the root) and continue the search.
    """
    def k_eigenval_finder(self, k):
        # Setting parameter counter to count the number of eigenvalues found.
        eigenval = []
        eigenval.append(0)
        step = 0.001
        counter = 0
        num = 0
        while counter <= k:
            x1 = num
            x2 = (num + step)
            if ((self.secular_det(x1) < 0 and self.secular_det(x2) > 0) or (self.secular_det(x1) > 0 and self.secular_det(x2) < 0)):
                check_zero = self.bisection(x1, x2, 100)
                if (check_zero != None):
                    eigenval.append(check_zero)
                    counter+=1
            num+=step
            # Represents infinite loop
            if num >= 10 and counter < 5: 
                self.startP = 0
                self.endP = 2*np.pi/self.minL
                self.SecularDetPloting()
                raise InfiniteLoop()
        # Printing eigenvalues
        for nb in range(0,counter):
            print("root of eigenvalue k number ", nb, " = ",  eigenval[nb])
        print("\n")
        self.startP = 0
        self.endP = num
        return eigenval

    """
    The following method aims to plot the secular determinant to understand on a graph where its roots are.
    """
    def SecularDetPloting(self):
        f = np.vectorize(self.secular_det)
        plt.figure()
        plt.title("Secular determinant as a function of k")
        plt.xlabel("k")
        plt.ylabel("Secular determinant")
        rate = (self.endP - self.startP)*0.001 
        k = np.arange(self.startP, self.endP, rate)
        plt.plot(k, f(k), 'b', k, 0*k,'r')
        plt.show()

    """
    The following method aims to test the class on different quantum graphs
    """
    def Testing(self):
        print("Init = \n", self.adj_mat);
        print("Nb edges = \n", self.Nb_edges);
        print("graph_characteristics = \n", self.graph_characteristics);
        print("S matrix = \n", self.SMatrix);
        print("EigenVal = \n", self.EigenVal);
        self.SecularDetPloting();

    """
    The following method aims to calculate the coefficients of the quantum graph's
    eigenfunctions according to the parameter k (which is an eigenvalue of the graph)
    """
    def eigenvector_aux(self,k):
        Nb_tot = int(self.Nb_edges[self.NbNodes])
        # Creation of the matrix D(k)
        D = np.zeros( (2*Nb_tot, 2*Nb_tot), dtype=complex )
        for row in range(0, 2*Nb_tot):
            for col in range(0, 2*Nb_tot):
                if (row == col):
                   D[row][col] = cmath.exp(1j*self.graph_characteristics[2][row]*k)
        # Finding the eigenvalues and eigenvectors of I-S*D(k)
        I = np.identity(2*Nb_tot)
        A = I - np.matmul(self.SMatrix,D)
        x = np.linalg.eig(A)
        return x
    
    """
    The following method aims to plot the eigenfunctions of each edge of the graph, in the 3D space of the quantum graph.
    The eigenfunctions of all the edges are drawn in different graphs according to the eigenvalue chosen.
    If a specific number of eigenvalue is not specified, all the eigenvalues (in a specific interval of possibilities found
    or chosen previously) are taken in consideration to plot the possible combinations of eigenfunctions.
    """
    def EigFunc3D_aux(self,k, ax):    

        # Finding and plotting the vertices
        X = np.zeros(int(self.NbNodes))
        Y = np.zeros(int(self.NbNodes))
        for i in range (0,int(self.NbNodes)):
            X[i] = self.vertices[i][0]
            Y[i] = self.vertices[i][1]
        for i in range(0,int(self.NbNodes)):
            if self.DCondition[i] == 0:
                ax.scatter3D(X[i],Y[i], 0, c='b', marker='o')
            else:
                ax.scatter3D(X[i],Y[i], 0, c='g', marker='o')
                
        # Plotting the edges (connecting the vertices according to the adjacency matrix)
        for i in range(0, int(2*self.TotEdges), 2):
            self.connectpoints(X,Y,int(self.graph_characteristics[0][i]),int(self.graph_characteristics[1][i]))
        

        # Setting the current eigenvalue used and the coefficients of the eigenfunctions (in the array eigvec)
        # The coefficients are chosen as the items of the eigenvector corresponding to the lowest eigenvalue of the secular determinant
        Curr_eigval = self.EigenVal[k]
        coeffs = self.eigenvector_aux(Curr_eigval)
        eigval_array = coeffs[0]
        Ind_eigval = np.argmin(np.abs(eigval_array))
        M = coeffs[1]
        M = M.T
        eigvec = M[int(Ind_eigval)]
            
        # For each edge of the graph, we plot the eigenfunction associated with the first eigenvalue and the first eigenvector found
        for i in range(0, int(2*self.TotEdges), 2):

            # We take the parameters of the currently used edge (on it we would draw the eigenfunction) 
            edge_number = i
            edge_length = self.graph_characteristics[2][i]

            # We find the array corresponding to the edge, upon which we will draw the eigenfunction
            if (Curr_eigval != 0):
                nb_points = int(50*np.pi*k*edge_length/(Curr_eigval))
            else: nb_points = 30
            t = np.linspace(0, 1, nb_points)
            p0 = int(self.graph_characteristics[0][edge_number])
            p1 = int(self.graph_characteristics[1][edge_number])
            p1_x = self.vertices[p1][0]
            p0_x = self.vertices[p0][0]
            p1_y = self.vertices[p1][1]
            p0_y = self.vertices[p0][1]
            delta_x = p1_x - p0_x
            delta_y = p1_y - p0_y
            X = p0_x + delta_x*t
            Y = p0_y + delta_y*t
            t = t*edge_length

            # We calculate and draw the eigenfunction upon the edge
            eigenfun = (eigvec[edge_number]*np.exp(1j*Curr_eigval*(edge_length-t)) + eigvec[edge_number+1]*np.exp(1j*Curr_eigval*(t))).real          
            ax.plot(X, Y, eigenfun, color='red')
        
        # Setting title, labels, and plotting
        ax.set_title("For eigenvalue %i:" %(k))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


    """
    The following method aims to plot the eigenfunctions of each edge of the graph, in the 3D space of the quantum graph.
    The eigenfunctions of all the edges are drawn in different graphs according to the eigenvalue chosen.
    All the eigenvalues are taken in consideration to plot the possible combinations of eigenfunctions.
    """
    def EigFuncPlot3D(self):
        fig = plt.figure()
        # Setting the plotting space for the quantum graph
        if (int(np.sqrt(len(self.EigenVal))) == np.sqrt(len(self.EigenVal))):
            nb_cols = int(np.sqrt(len(self.EigenVal)))
        else:
            nb_cols = int(np.sqrt(len(self.EigenVal))) + 1
        nb_rows = math.ceil(len(self.EigenVal)/nb_cols)
        if self.num_eigval==None:
            for k in range(0, len(self.EigenVal)-1):
                ax = fig.add_subplot(nb_rows, nb_cols, k+1, projection="3d")
                self.EigFunc3D_aux(k, ax)
        else:
            print("here1: ", self.num_eigval)
            ax = fig.add_subplot(1,1,1,projection="3d")
            self.EigFunc3D_aux(self.num_eigval, ax)
            print("here2")
        plt.show()



    """
    The following method aims to connect two points given in a virtual space.
    x (horizontal axis) and y (vertical axis) are two vectors which contain the coordinates of each point of the graph.
    We want here to connect the points (x[p1],y[p1]) and (x[p2],y[p2])
    """
    def connectpoints(self,x,y,p1,p2):
        x1, x2 = x[p1], x[p2]
        y1, y2 = y[p1], y[p2]
        y = plt.plot([x1,x2],[y1,y2],'k-')

    """
    The following method aims to sketch the quantum graph.
    We start by placing the right number of nodes in random places (red points), and them we connect them according to the graph
    characteristics, thanks to the adjacency matrix.
    """
    def sketching(self):
        X = np.random.rand(int(self.NbNodes))
        Y = np.random.rand(int(self.NbNodes))
        for i in range(0,int(self.NbNodes)):
            plt.plot(X[i],Y[i], 'ro')
        for i in range(0, int(self.TotEdges)):
            self.connectpoints(X,Y,int(self.graph_characteristics[0][i]),int(self.graph_characteristics[1][i]))
        plt.show()

    """
    The following method aims to sketch the quantum graph using given coordinates.
    We start by placing the right number of nodes in given places (red points), and them we connect them according to the graph
    characteristics, thanks to the adjacency matrix.
    """
    def set_coordinates(self):
        X = np.zeros(int(self.NbNodes))
        Y = np.zeros(int(self.NbNodes))
        for i in range (0,int(self.NbNodes)):
            X[i] = self.vertices[i][0]
            Y[i] = self.vertices[i][1]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(0,int(self.NbNodes)):
            ax.scatter3D(X[i],Y[i], 0, c='r', marker='o')
        for i in range(0, int(self.TotEdges)):
            self.connectpoints(X,Y,int(self.graph_characteristics[0][i]),int(self.graph_characteristics[1][i]))


# Here are some tests as examples for using the package  
'''
# Tests: 

# Test on a star graph:
A = [ [0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0] ]
coordA = [ [0, 1], [0, 0], [1, 0], [1, 1] ]
DirichletA = [1, 0, 1, 0]

# Test on a V curved graph:
B = [ [0, 1, 0], [1, 0, 1], [0, 1, 0] ]
coordB = [ [0, 0], [1, 2], [2, 0] ]
DirichletB = [0, 0 ,0]

# Test on a V graph:
C = [ [0, 1, 0], [1, 0, 1], [0, 1, 0] ]
coordC = [ [0, 0], [1, 1], [np.pi, np.pi] ]
DirichletC = [1, 0, 0]

# Test on a quadrilatere + wings graph:
D = [ [0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 1, 0]]
coordD = [ [0, 1], [1, 1], [2, 0], [2, 2], [3, 1], [4, 1]]

# Test on a U graph:
E = [ [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0] ]
coordE = [ [0, 0], [1, 0], [1, 1], [0, 1] ]
DirichletE= [1, 0, 0, 1]

#Test on Dirichlet interval
F = [ [0, 1], [1, 0] ]
coordF = [ [0, 0], [0, 1] ]
DirichletF = [1, 1]

# Initilizing the quantum graphs
QuantGraph1 = QuantumGraph(4, A, coordA, 5, DirichletA);
QuantGraph2 = QuantumGraph(3, B, coordB, 6, DirichletB);
QuantGraph3 = QuantumGraph(3, C, coordC, 6, DirichletC);
QuantGraph6 = QuantumGraph(2, F, coordF, 4, DirichletF);
QuantGraph5 = QuantumGraph(4, E, coordE, 5, DirichletE)

# Ploting the respective eigenfunctions
QuantGraph1.EigFuncPlot3D()
QuantGraph2.EigFuncPlot3D()
QuantGraph3.EigFuncPlot3D()
QuantGraph6.EigFuncPlot3D()
QuantGraph5.EigFuncPlot3D()
'''


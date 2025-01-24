# encoding the two matrices
import numpy as np
A = [[1,2,3],[5,3,1],[7,9,8]]
B = [[3,4],[12,4],[8,14]]
 
# retrieving the sizes/dimensions of the matrices
p = len(A)
q = len(A[0])
 
t = len(B)
r = len(B[0])
 
if(q!=t):
   print("Error! Matrix sizes are not compatible")
   quit()
 
# creating the product matrix of dimensions p×r
# and filling the matrix with 0 entries
C = []
for row in range(p):
   curr_row = []
   for col in range(r):
       curr_row.append(0)
   C.append(curr_row)
 
 
# performing the matrix multiplication
for i in range(p):
   for j in range(r):
       curr_val = 0
       for k in range(q):
           curr_val += A[i][k]*B[k][j]
       C[i][j] = curr_val
 
print(C)


# three dimensional arrays
m1 = ([1, 6, 5],[3 ,4, 8],[2, 12, 3]) 
m2 = ([3, 4, 6],[5, 6, 7],[6,56, 7]) 
m1 = np.dot(m1,m2) 
print(m1) 

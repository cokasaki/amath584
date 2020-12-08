
import numpy as np
from scipy import linalg
from copy import deepcopy

def LU(A):
	(m,n) = A.shape
	U = deepcopy(A)
	P = np.eye(m)
	L = np.eye(m)

	for i in range(m):
		# do the pivot
		max_j = i + np.argmax(np.abs(U[i:,i]))

		# make the pivot matrix
		Pi = np.eye(m)
		Pi[i,i] = 0
		Pi[max_j,max_j] = 0
		Pi[i,max_j] = 1
		Pi[max_j,i] = 1

		# do the pivot 
		U = Pi @ U

		# add the pivot to P
		P = P @ Pi

		# add the pivot to L
		L= Pi @ L @ Pi

		# make the eliminator Linv
		Li = np.eye(m)
		Li[(i+1):m,i] = -U[(i+1):m,i]/U[i,i]

		# make the inverse of the eliminator
		Linv = np.eye(m)
		Linv[(i+1):m,i] = U[(i+1):m,i]/U[i,i]

		# do the elimination (we already did the pivot)
		U = Li @ U

		# add the inverse-eliminator to L
		L = L @ Linv

	L = linalg.tril(L)
	U = linalg.triu(U)

	return (P,L,U)

test3by3 = np.random.normal(size=(3,3))
(P,L,U) = LU(test3by3)


print(np.sum(np.abs(test3by3 - P@L@U)))

testbig = np.random.normal(size=(100,100))
(P,L,U) = LU(testbig)


print(np.sum(np.abs(testbig - P@L@U)))











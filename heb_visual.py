import cupy as cp

axon = cp.array([[1, 0], [1, 0]])
y1 = cp.array([[1, 0]]) 
y2 = cp.dot(y1, axon) # [[1, 0]]
y1_transposed = y1.transpose()  # [[1]
                                #  [0]]
update = 0.001 * cp.dot(y1_transposed, y2) #[[0.001, 0.0
                                            #  0.0, 0.0]]
new_axon = update + axon
print(new_axon) #[[1.001 0.   ]
                #[1.    0.   ]]

import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x_arr=np.array(x,dtype=np.float64)
        w1_arr=np.array(W1,dtype=np.float64)
        b1_arr=np.array(b1,dtype=np.float64)
        W2_arr=np.array(W2,dtype=np.float64)
        b2_arr=np.array(b2,dtype=np.float64)
        y_arr=np.array(y_true,dtype=np.float64)


        #---forward ----
        #--layer1---
        z1=np.dot(w1_arr,x_arr)+b1_arr
        a1=np.maximum(0.0,z1)

        #----layer2--
        z2=np.dot(W2_arr,a1)+ b2_arr

        predictions=z2
        #--loss compution---
        loss=np.mean((predictions-y_arr)**2)

        #---Backward Pass ----
        #Derivative of MSE : 2/N *(predictions - y_true)
        
        n_out=len(y_arr)
        dL_dpred=(2.0/n_out)*(predictions-y_arr)

        #Gradient for layer 2
        dL_dz2=dL_dpred
        dW2=np.outer(dL_dz2,a1)
        db2=dL_dz2

        #---Gradients for Layer 1
        #---Backpropagate through W2: dL__da1=W2*dL_dz2
        dL_da1=np.dot(W2_arr.T,dL_dz2)

        dL_dz1=dL_da1*(z1>0).astype(np.float64)

        dW1=np.outer(dL_dz1,x_arr)
        db1=dL_dz1

        return {
            'loss': round(float(loss),4),
            'dW1': np.round(dW1,4).tolist(),
            'db1':np.round(db1,4).tolist(),
            'dW2':np.round(dW2,4).tolist(),
            'db2':np.round(db2,4).tolist()
        }


import numpy as np

import torch

bias = -4

weights = np.array([1,-1,1,
                    -1,1,-1,
                    1,-1,1
                    ])

def RelU(output):
    return max(0, output)

def BinaryStep(output, Threshold):
    if output>=Threshold:
        return 1
    else:
        return 0

def test(weights, bias):
    input1 = np.array([1, 0 , 1,
                  0, 1, 0,
                  1, 0, 1])
    input2 = np.array([0, 0, 1,
                       0, 1, 0,
                       1, 0, 1])
    input3 = np.array([0, 0, 0,
                       0, 1, 1,
                       1, 0, 1])
    input4 = np.array([1, 1, 1,
                       1, 1, 1,
                       1, 1, 1])
    input5 = np.array([1, 0, 0,
                       0, 1, 0,
                       1, 0, 0])
    output1 = RelU(np.matmul(input1, weights)+bias)
    output2 = RelU(np.matmul(input2, weights) + bias)
    output3 = RelU(np.matmul(input3, weights) + bias)
    output4 = RelU(np.matmul(input4, weights) + bias)
    output5 = RelU(np.matmul(input5, weights) + bias)

    if output1==1:
        print("Test 1: Passed")
    else:
        print("Test 1: Failed")
    if output2 == 0:
        print("Test 2: Passed")
    else:
        print("Test 2: Failed")
    if output3 == 0:
        print("Test 3: Passed")
    else:
        print("Test 3: Failed")
    if output4 == 0:
        print("Test 4: Passed")
    else:
        print("Test 4: Failed")
    if output5 == 0:
        print("Test 5: Passed")
    else:
        print("Test 5: Failed")


test(weights, bias)
input = torch.FloatTensor([[1,0,1,
                            0,1,0,
                            1,0,1]])

neuron = torch.nn.Linear(9, 1, bias = True)
class SingleLayerLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerLinear, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, input):
        result = self.linear(input)
        return result

model = SingleLayerLinear(9,1)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr =0.01)

X = torch.FloatTensor([[1,0,1,
                        0,1,0,
                        1,0,0]])
Y = torch.FloatTensor([[1]])

epochs = 5
loss = []

for epoch in range(epochs):
    y_hat = model(X)
    print(y_hat)
    loss_epoch = loss_function(y_hat, Y)
    loss_epoch.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss.append(loss_epoch.item())

print(epochs)










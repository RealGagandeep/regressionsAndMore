import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import io
print(sys.argv)

import numpy as np
from numpy.random import normal as gaussian   # Return a normal distributed array of given size

          #****************IMPORTANT INFORMATION*****************#

# eta = 0.001, batchsize is = 1, stopping criteria = 3*10**-7 ; Time: 4 to 9s
# eta = 0.001, batchsize is = 100, stopping criteria = 3*10**-5 ; Time: 2 to 7s
# eta = 0.001, batchsize is = 10000, stopping criteria = 3*10**-5 ; Time: 3m to 4m 14s
# eta = 0.1, batchsize is = 1000000, stopping criteria = 3*10**-3 ; Time: 3m 17s to 5m 2s

#***********************************************************************************#

# url = sys.argv[1]+'/X.csv'
# # url = 'https://raw.githubusercontent.com/RealGagandeep/DataSet-for-ML/main/data/q2/q2test.csv'
# data = pd.read_csv(url)

# arr = pd.DataFrame(data)
# Dataset = arr.to_numpy()

# url = sys.argv[1]+'/X.csv'
url = 'https://raw.githubusercontent.com/RealGagandeep/DataSet-for-ML/main/data/q2/X.csv'
data = pd.read_csv(url)

arr = pd.DataFrame(data)
xTest = arr.to_numpy()


# url = sys.argv[3]+'/X.csv'
# # url = 'https://raw.githubusercontent.com/RealGagandeep/DataSet-for-ML/main/data/q2/true_Y.csv'
# data = pd.read_csv(url)

# arr = pd.DataFrame(data)
# yValidate = arr.to_numpy()


# sampleDatax1 = Dataset[:,0]
# sampleDatax2 = Dataset[:,1]
testDatax1 = xTest[:,0]
testDatax2 = xTest[:,1]

# sampleDatay = Dataset[:,2]
# testDatay = yValidate[:]
eta = .001                  # learning rate
size = 10**6                # Size of generated dataset
batch = 100                 # Batch size
stoppingCriteria = 3*10**-5 # Stopping criteria [L1 norm]
theta0 = 0                  # Initial assumed theta0
theta1 = 0                  # Initial assumed theta1
theta2 = 0                  # Initial assumed theta2
caseCounter = 1
crieteria = 0

def diff(thetaOld,thetaNew):                       # Function to find difference
  return abs(thetaOld - thetaNew)

def batc(y,batch,k):                               # Funtion to split in batches
  z = []
  for i in range(k,k+batch):
    z.append(y[i])
  return np.array(z)

def finalError(sampleDatay,sampleDatax1,sampleDatax2,theta0,theta1,theta2):
  sum = 0
  for i in range(len(sampleDatay)):
    sum = sum + ((sampleDatay[i] - (theta0 + theta1*sampleDatax1[i] + theta2*sampleDatax2[i]))**2)
  return (sum)/len(sampleDatay)

x0 = gaussian(3,0,size)                            # Initialized values of x0
x1 = gaussian(3,2,size)                            # Initialized values of x1
x2 = gaussian(-1,2,size)                           # Initialized values of x2
noise = gaussian(0,1.414,size)                     # Gaussian noise

y = x0 + 1*x1 + 2*x2 + noise                       # Making a dataset of 1M values using theta(s)

val0=[]
val1=[]
val2=[]
k = 0                                              # Data points counter
iteration = 0                                      # Iteration counter

while(1):
  # if(caseCounter==0):                              # For first case i.e. batch of size '1'
  #   eta = 0.001
  #   batch = 1
    # stoppingCriteria = 2*10**-7
  if(caseCounter==1):                            # For first case i.e. batch of size '100'
    eta = 0.001
    batch = 100
    stoppingCriteria = 3*10**-5
  # elif(caseCounter==2):                            # For first case i.e. batch of size '10000'
  #   eta = 0.001
  #   batch = 10000
  #   stoppingCriteria = 6*10**-5
  # elif(caseCounter==3):                            # For first case i.e. batch of size '1 M'
  #   eta = 0.1
  #   batch = 1000000
  #   stoppingCriteria = 3*10**-3
  # else:
  #   break

  iteration += 1                                   # Iteration incremator
  yPredict = []
  if(k>999999):                                    # If training goes beyond len(y) then loop again
    k = 0
  for i in range(k,k + batch):                              # Making batches of yPredict
    yPredict.append(theta0 + theta1*x1[i] + theta2*x2[i])

  del0 = -(batc(y,batch,k) - yPredict)                      # Calculating the gradient of the batches
  del1 = -(batc(y,batch,k) - yPredict)*batc(x1,batch,k)     # Calculating the gradient of the batches
  del2 = -(batc(y,batch,k) - yPredict)*batc(x2,batch,k)     # Calculating the gradient of the batches

  thetaOld0 = theta0
  thetaOld1 = theta1                              #*** Storing the value of theta before updation for diff. calculation ***#
  thetaOld2 = theta2

  theta0 = theta0 - eta*np.mean(del0)
  theta1 = theta1 - eta*np.mean(del1)             #*** Updation of theta(s) ***#
  theta2 = theta2 - eta*np.mean(del2)

  thetaNew0 = theta0
  thetaNew1 = theta1                              #*** Storing the value of theta after updation for diff. calculation ***#
  thetaNew2 = theta2

  val0.append(theta0)
  val1.append(theta1)
  val2.append(theta2)

  difference = (diff(thetaOld0,thetaNew0) + diff(thetaOld1,thetaNew1) +diff(thetaOld2,thetaNew2)) # Calculating diff. for stopping criteria

  #print(difference, k, theta0, theta1, theta2)
  k+=batch
  if(difference<stoppingCriteria):
    crieteria += 1
  if(crieteria>=5):
    yValidation = []
    yValidation.append(theta0 + theta1*testDatax1 + theta2*testDatax2)
    #print(difference, iteration, theta0, theta1, theta2)
    print(f'For batch size: {batch}, L1Norm: {difference}, iterations: {iteration}, theta0: {theta0}, theta1: {theta1}, theta2: {theta2}')
    #print(f'error(MSE) for test data(q2test.csv) is: {finalError(sampleDatay,sampleDatax1,sampleDatax2,theta0,theta1,theta2)}')
    #print(f'error(MSE) for original hypothesis function is: {finalError(3+1*sampleDatax1+2*sampleDatax2,sampleDatax1,sampleDatax2,theta0,theta1,theta2)}')
    print(f'The value of validation data is {(np.array(yValidation)).T}')
    print('**********************************************************************************************************************************************')
    with open("result_2.txt", "a") as f:
      print(f'For batch size: {batch}, L1Norm: {difference}, iterations: {iteration}, theta0: {theta0}, theta1: {theta1}, theta2: {theta2}', file=f)
      #print(f'error(MSE) for test data(q2test.csv) is: {finalError(sampleDatay,sampleDatax1,sampleDatax2,theta0,theta1,theta2)}', file=f)
      #print(f'error(MSE) for original hypothesis function is: {finalError(3+1*sampleDatax1+2*sampleDatax2,sampleDatax1,sampleDatax2,theta0,theta1,theta2)}', file=f)
      print(f'The value of validation data is {(np.array(yValidation)).T}', file=f)
      print('**********************************************************************************************************************************************', file=f)
    caseCounter+=1
    crieteria = 0
    k = 0
    iteration = 0
    theta0 = 0
    theta1 = 0
    theta2 = 0



fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# defining all 3 axes
z = val2
x = val0
y = val1

# plotting
ax.plot3D(x, y, z, 'green')
ax.set_title('3D line plot geeks for geeks')
plt.show()

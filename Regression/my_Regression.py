import math
import torch.optim as optim
import sys
#sys.path.append('../')
from my_BayesByBackProp import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def train(net, optimizer, data, target, NUM_BATCHES, epoch):
    net.train()
    for i in range(NUM_BATCHES):
        net.zero_grad()
        x = data[i].reshape((-1, 1))
        y = target[i].reshape((-1,1))
        loss = net.BBB_loss(x, y)
        loss.backward()
        optimizer.step()
        if (epoch)%50 == 0:
           print("loss", loss)


#Hyperparameter setting
TRAIN_EPOCHS = 900
SAMPLES = 5
TEST_SAMPLES = 10
BATCH_SIZE = 200
NUM_BATCHES = 10
TEST_BATCH_SIZE = 50
CLASSES = 1
PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)

print('Generating Data set.')

if torch.cuda.is_available():
    Var = lambda x, dtype=torch.cuda.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor
else:
    Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor

# new noise model
def noise_model(x):
    return 0.45*(x+0.5)**2

x = np.random.rand(NUM_BATCHES, BATCH_SIZE) - 0.5
y = -(x+0.5)*np.sin(3 * np.pi *x) + np.random.normal(0, noise_model(x))

x_test = np.linspace(-1, 1, TEST_BATCH_SIZE)   
y_test = -(x_test+0.5)*np.sin(3 * np.pi *x_test)

def BBB_Regression(x,y,x_test,y_test):

    print('BBB Training Begins!')

    X = Var(x)
    Y = Var(y)
    X_test = Var(x_test)

    #Declare Network
    # Defines the BNN architecture by calling the BayesianNetwork class constructor.
    net = BayesianNetwork(inputSize = 1,\
                        CLASSES = CLASSES, \
                        layers=np.array([16,16,16]), \
                        activations = np.array(['relu','relu','relu','none']), \
                        SAMPLES = SAMPLES, \
                        BATCH_SIZE = BATCH_SIZE,\
                        NUM_BATCHES = NUM_BATCHES,\
                        hasScalarMixturePrior = True,\
                        PI = PI,\
                        SIGMA_1 = SIGMA_1,\
                        SIGMA_2 = SIGMA_2)

    #Declare the optimizer
    #optimizer = optim.SGD(net.parameters(),lr=1e-3,momentum=0.95)

    optimizer = optim.Adam(net.parameters())

    #Declare the optimizer
    #optimizer = optim.SGD(net.parameters(),lr=1e-3,momentum=0.95)

    for epoch in range(TRAIN_EPOCHS):
        if (epoch)%10 == 0:
            print('Epoch: ', epoch)
        train(net, optimizer,data=X,target=Y,NUM_BATCHES=NUM_BATCHES, epoch=epoch)

    print('Training Ends!')

    # Testing
    # Computes mean prediction and standard deviation for the test data by calling the forward method of the BNN for a given number of test samples.
    outputs = torch.zeros(TEST_SAMPLES+1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
    #print(outputs.shape)
    #print((net.forward(X).shape))
    #print(TEST_SAMPLES)
    for i in range(TEST_SAMPLES):
        outputs[i] = net.forward(X_test, infer = True)
    outputs[TEST_SAMPLES] = net.forward(X_test, infer = True)
    pred_mean = outputs.mean(0).data.cpu().numpy().squeeze(1) #Compute mean prediction
    pred_std = outputs.std(0).data.cpu().numpy().squeeze(1) #Compute standard deviation of prediction for each data point

    #Visualization
    plt.fill_between(x_test, pred_mean - 3 * pred_std, pred_mean + 3 * pred_std,
                        color='cornflowerblue', alpha=.5, label='+/- 3 std')
    plt.scatter(x, y,marker='x', c='black', label='target')
    plt.plot(x_test, pred_mean, c='red', label='Prediction')
    plt.plot(x_test, y_test, c='grey', label='truth')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Results/My_Regression_BBB_DVI.png')
    plt.savefig('Results/My_Regression_BBB_DVI.eps', format='eps', dpi=1000)
    plt.clf()

    #Save the trained model
    torch.save(net.state_dict(), './My_Regression_DVI.pth')


BBB_Regression(x,y,x_test,y_test)
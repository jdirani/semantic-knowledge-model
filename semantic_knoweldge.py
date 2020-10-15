# Semantic Knowledge - Model Extension

# Based on model from:
# McClelland, J. L., & Rogers, T. T. (2003). The parallel distributed processing approach to semantic cognition. Nature Reviews Neuroscience, 4(4), 310.

# Extends the McCelland & Rogers (2003) model to include part-whole relations where
# the parts are also represented as full concepts by the model.
# e.g bird -> wings. "wings" also repreented as a concept, not just as a feature.
# Does the neural  net model learn a hierarchical structure? e.g. Bird -> wings -> feathers.

# Import libraries
from __future__ import print_function
import matplotlib, torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.nn.functional import sigmoid, relu
from scipy.cluster.hierarchy import dendrogram, linkage
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's first load in the names of all the items, attributes, and relations into Python lists.


with open('data/sem_items.txt','r') as fid:
    names_items = np.array([l.strip() for l in fid.readlines()])
with open('data/sem_relations.txt','r') as fid:
    names_relations = np.array([l.strip() for l in fid.readlines()])
with open('data/sem_attributes.txt','r') as fid:
    names_attributes = np.array([l.strip() for l in fid.readlines()])

nobj = len(names_items)
nrel = len(names_relations)
nattributes = len(names_attributes)

print('List of items:')
print(names_items)
print('')
print("List of relations:")
print(names_relations)
print('')
print("List of attributes:")
print(names_attributes)
print('')
print('# of items: ', nobj)
print('# of relations: ', nrel)
print('# of attributes: ', nattributes)


# In[11]:


# Check all the items are also attributes
len([value for value in names_items if value in names_attributes] ) == len(names_items)


# Next, let's load in the data matrix from a text file too. The matrix `D` has a row for each training pattern. It is split into a matrix of input patterns `input_pats` (item and relation) and their corresponding output patterns `output_pats` (attributes). The are `N` patterns total in the set.
#
# For each input pattern, the first 30 elements indicate which item is being presented, and the next 7 indicate which relation is being queried. Each element of the output pattern corresponds to a different attribute. All patterns use 1-hot encoding.

# In[12]:


D = np.loadtxt('data/sem_data.txt')
input_pats = D[:,:nobj+nrel]
input_pats = torch.tensor(input_pats,dtype=torch.float)
output_pats = D[:,nobj+nrel:]
output_pats = torch.tensor(output_pats,dtype=torch.float)
N = input_pats.shape[0] # number of training patterns
input_v = input_pats[0,:].numpy().astype('bool')
output_v = output_pats[0,:].numpy().astype('bool')


print('Example input pattern:')
print(input_v.astype('int'))
print('Example output pattern:')
print(output_v.astype('int'))
print("")
print("Which encodes...")
print('Item ',end='')
print(names_items[input_v[:30]])
print('Relation ',end='')
print(names_relations[input_v[30:]])
print('Attributes ',end='')
print(names_attributes[output_v])

assert len(input_v.astype('int')) == nobj+nrel
assert len(output_v.astype('int')) == nattributes


# <div class="alert alert-success" role="alert">
# <h3> Build the network </h3>
# Use the ReLu activation function ("relu") for the Representation and Hidden Layers, with a Logistic/Sigmoid activation function for the Attribute Layer ("sigmoid").
# <br><br>
# Need PyTorch's "nn.Linear" function for constructing the layers, and the "relu" and "sigmoid" activation functions.
# </div>

# In[13]:


class Net(nn.Module):
    def __init__(self, rep_size, hidden_size):
        super(Net, self).__init__()
        # Input
        #  rep_size : number of hidden units in "Representation Layer"
        #  hidden_Size : number of hidden units in "Hidden Layer"

        self.fc1 = nn.Linear(nobj, rep_size)  # items to representations
        self.fc2 = nn.Linear(rep_size + nrel, hidden_size) # representations and relations to hidden
        self.fc3 = nn.Linear(hidden_size, nattributes) # hidden to attributes


    def forward(self, x):
        # Defines forward pass for the network on input patterns x
        #
        # Input can take these two forms:
        #
        #   x: [nobj+nrel 1D Tensor], which is a single input pattern as a 1D tensor
        #      (containing both object and relation 1-hot identifier) (batch size is B=1)
        #   OR
        #   x : [B x (nobj+nrel) Tensor], which is a batch of B input patterns (one for each row)
        #
        # Output
        #   output [B x nattribute Tensor], which is the output pattern for each input pattern B on the Attribute Layer
        #   hidden [B x hidden_size Tensor], which are activations in the Hidden Layer
        #   rep [B x rep_size Tensor], which are the activations in the Representation LAyer
        x = x.view(-1,nobj+nrel) # reshape as size [B x (nobj+nrel) Tensor] if B=1
        x_item = x[:,:nobj] # input to Item Layer [B x nobj Tensor]
        x_rel = x[:,nobj:] # input to Relation Layer [B x nrel Tensor]


        rep = relu(self.fc1(x_item))
        rep_rel = torch.cat((rep,x_rel),1)
        hidden = relu(self.fc2(rep_rel))
        output = sigmoid(self.fc3(hidden))

        # -----
        return output, hidden, rep


# We provide a completed function `train` for stochastic gradient descent. The network makes online (rather than batch) updates, adjusting its weights after the presentation of each input pattern.

# In[14]:


def train(mynet,epoch_count,nepochs_additional=5000):
    # Input
    #  mynet : Net class object
    #  epoch_count : (scalar) how many epochs have been completed so far
    #  nepochs_additional : (scalar) how many more epochs we want to run
    mynet.train()
    for e in range(nepochs_additional): # for each epoch
        error_epoch = 0.
        perm = np.random.permutation(N)
        for p in perm: # iterate through input patterns in random order
            mynet.zero_grad() # reset gradient
            output, hidden, rep = mynet(input_pats[p,:]) # forward pass
            target = output_pats[p,:]
            loss = criterion(output, target) # compute loss
            loss.backward() # compute gradient
            optimizer.step() # update network parameters
            error_epoch += loss.item()
        error_epoch = error_epoch / float(N)
        if e % 50 == 0:
            print('epoch ' + str(epoch_count+e) + ' loss ' + str(round(error_epoch,3)))
    return epoch_count + nepochs_additional


# We provide some useful functions for extracting the activation pattern on the Representation Layer for each possible item. We provide two functions `plot_rep` and `plot_dendo` for visualizing these activation patterns.

# In[15]:


def get_rep(net):
    # Extract the hidden activations on the Representation Layer for each item
    #
    # Input
    #  net : Net class object
    #
    # Output
    #  rep : [nitem x rep_size numpy array], where each row is an item
    input_clean = torch.zeros(nobj,nobj+nrel)
    for idx,name in enumerate(names_items):
        input_clean[idx,idx] = 1. # 1-hot encoding of each object (while Relation Layer doesn't matter)
    output, hidden, rep = net(input_clean)
    return rep.detach().numpy()

def plot_rep(rep1,rep2,rep3,names):
    #  Compares Representation Layer activations of Items at three different times points in learning (rep1, rep2, rep3)
    #  using bar graphs
    #
    #  Each rep1, rep2, rep3 is a [nitem x rep_size numpy array]
    #  names : [nitem list] of item names
    #
    nepochs_list = [nepochs_phase1,nepochs_phase2,nepochs_phase3]
    nrows = nobj
    R = np.dstack((rep1,rep2,rep3))
    mx = R.max()
    mn = R.min()
    depth = R.shape[2]
    count = 1
    plt.figure(1,figsize=(4.2,8.4))
    for i in range(nrows):
        for d in range(R.shape[2]):
            plt.subplot(nrows, depth, count)
            rep = R[i,:,d]
            plt.bar(range(rep.size),rep)
            plt.ylim([mn,mx])
            plt.xticks([])
            plt.yticks([])
            if d==0:
                plt.ylabel(names[i])
            if i==0:
                plt.title("epoch " + str(nepochs_list[d]))
            count += 1
    plt.show()

def plot_dendo(rep1,rep2,rep3,names):
    #  Compares Representation Layer activations of Items at three different times points in learning (rep1, rep2, rep3)
    #  using hierarchical clustering
    #
    #  Each rep1, rep2, rep3 is a [nitem x rep_size numpy array]
    #  names : [nitem list] of item names
    #
    nepochs_list = [nepochs_phase1,nepochs_phase2,nepochs_phase3]
    linked1 = linkage(rep1,'single')
    linked2 = linkage(rep2,'single')
    linked3 = linkage(rep3,'single')
    mx = np.dstack((linked1[:,2],linked2[:,2],linked3[:,2])).max()+0.1
#     plt.figure(2,figsize=(7,12))
    plt.figure(2,figsize=(14,18))
    plt.subplot(3,1,1)
    dendrogram(linked1, labels=names, color_threshold=0)
    plt.ylim([0,mx])
    plt.title('Hierarchical clustering; ' + "epoch " + str(nepochs_list[0]))
    plt.ylabel('Euclidean distance')
    plt.subplot(3,1,2)
    plt.title("epoch " + str(nepochs_list[1]))
    dendrogram(linked2, labels=names, color_threshold=0)
    plt.ylim([0,mx])
    plt.subplot(3,1,3)
    plt.title("epoch " + str(nepochs_list[2]))
    dendrogram(linked3, labels=names, color_threshold=0)
    plt.ylim([0,mx])
    plt.show()


# The next script initializes the neural network and trains it for 4000 epochs total. It trains in three stages, and the item representations (on the Representation Layer) are extracted after 500 epochs, 1500 epochs, and then at the end of training (4000 epochs).

# ## Using same architecture as original model: rep_size = 8, hidden=15

# In[8]:


learning_rate = 0.1
criterion = nn.MSELoss() # mean squared error loss function
mynet = Net(rep_size=8,hidden_size=15)
optimizer = torch.optim.SGD(mynet.parameters(), lr=learning_rate) # stochastic gradient descent

nepochs_phase1 = 500
nepochs_phase2 = 1000
nepochs_phase3 = 2500
epoch_count = 0
epoch_count = train(mynet,epoch_count,nepochs_additional=nepochs_phase1)
rep1 = get_rep(mynet)
epoch_count = train(mynet,epoch_count,nepochs_additional=nepochs_phase2-nepochs_phase1)
rep2 = get_rep(mynet)
epoch_count = train(mynet,epoch_count,nepochs_additional=nepochs_phase3-nepochs_phase2)
rep3 = get_rep(mynet)


# Finally, let's visualize the Representation Layer at the different stages of learning.

# In[33]:


plot_rep(rep1,rep2,rep3,names_items)
plot_dendo(rep1,rep2,rep3,names_items)


# ## ====================================================================<br>====================================================================

# ## (1) Using rep_size = nobj=37, hidden = 40 = average of rep+rel and output size

# In[34]:


rep_size = nobj
hidden_size = 40

learning_rate = 0.1
criterion = nn.MSELoss() # mean squared error loss function
mynet = Net(rep_size=8,hidden_size=15)
optimizer = torch.optim.SGD(mynet.parameters(), lr=learning_rate) # stochastic gradient descent

nepochs_phase1 = 500
nepochs_phase2 = 1000
nepochs_phase3 = 2500
epoch_count = 0
epoch_count = train(mynet,epoch_count,nepochs_additional=nepochs_phase1)
rep1 = get_rep(mynet)
epoch_count = train(mynet,epoch_count,nepochs_additional=nepochs_phase2-nepochs_phase1)
rep2 = get_rep(mynet)
epoch_count = train(mynet,epoch_count,nepochs_additional=nepochs_phase3-nepochs_phase2)
rep3 = get_rep(mynet)

plot_dendo(rep1,rep2,rep3,names_items)



# ## ====================================================================<br>====================================================================

# ## Save trained model
import pickle

pickle.dump(mynet, 'mynet.pickle')
file.close()


# ## ====================================================================<br>====================================================================

# ## EXPLORE etc..


def get_single_input_vec(input_obj, input_rel, total_nobj):

    obj_idx = np.where(names_items==input_obj)[0][0]
    rel_idx = np.where(names_relations==input_rel)[0][0] + total_nobj
    input_single = torch.zeros(nobj+nrel)
    input_single[[obj_idx, rel_idx]] = 1
    return input_single

# test 1
input_single = get_single_input_vec(input_obj='skin', input_rel='Part_of',total_nobj= nobj)
output, hidden, rep = mynet(input_single)

out = output[0].detach().numpy()
idx_attributes = np.where(out > 0.2) # 0.2 arbitrary here
names_attributes[idx_attributes]

# cpp-neuronet
Small example object oriented neural net on C++
Object oriented neural network
Using C++17
classes:

class Neuron
class NeuronsLayer
class SkyNet

Version 1:
weghts input by user
Mashine learning is not avalable

Version 2:
Network has method Learning now.
You have to make input dataset and exepected values for each case, next call Learning.
For Decision may be use sigma function only, because learning uses derivative from this function.

Version 3:
Neural network was refactored for many decisions, not only one. Now Neural network is templated with output type of desicions (i.e. bool, double etc.)
Serialisation of wheiths coefficients was added. There is possiblity to save wheight after learnin. So Neural Network need once to be learning and then just use saved wgeights.
Protobuf library was used for serialisation.
Now all project build with CMake.
As example Added learning for MNIST dataset for hanwritten digits recognising. MNIST datsets(http://yann.lecun.com/exdb/mnist/) is train-images.idx3-ubyte and train-images.idx3-ubyte train-labels.idx1-ubyte.train-labels.idx1-ubyte 


Whats new in future versions?
Fast learning. Structure of Neural network was ctreated just for example neural working. Because this reason learnin is slow. I will fix it



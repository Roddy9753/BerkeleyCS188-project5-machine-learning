from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


#(x,y) 向量x二分类，分类标签为y
#输入维度为向量x的维度，输出维度为1。没有隐藏层
class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        self.w = None #Initialize your weights here
        #随机生成1*dimentions的初始链接权重。这里dimentions指的是数据集中每个点的维度
        self.w = Parameter(ones(1, dimensions)) 


    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w
    

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        #实现模型输入的变量x（一个tensor）和刚刚的链接权重self.w之间的点积。实现神经网络每个节点中的加法器的功能。
        return tensordot(x, self.w) 


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        #当上面加法器的点积结果为非负数的时候，返回1；否则返回-1。总体功能类似于一个激活函数。
        if self.run(x) >= 0 :
            return 1
        else :
            return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            #更新参数直到所有的预测结果y'都与标准输出y相同
            all_correct = False #是否全部预测正确
            while not all_correct:
                all_correct = True
                for data_block in dataloader: #对训练集中的每一个数据而言
                    x = data_block['x'] #输入的向量
                    y = data_block['label'] #标准输出的标签
                    if self.get_prediction(x) != y: #出现训练错误
                        all_correct = False
                        self.w += y * x #更新链接权重参数



#建模预测sin(x)
class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()
        #构建一层隐藏层的大小
        self.hidden_layer_node_num = 100
        #定义隐藏层和输出层两个线性层
        self.hidden_layer = Linear(1,self.hidden_layer_node_num)
        self.output_layer = Linear(self.hidden_layer_node_num,1)



    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #经过两层的前向传播，得到预测的y值
        hidden_layer_out = relu(self.hidden_layer(x))
        y_predicted = self.output_layer(hidden_layer_out)
        return y_predicted

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        #前向传播得到的预测值和标准输出
        return mse_loss(self.forward(x), y)
 
  

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"
        #先定义优化器
        optimizer = optim.Adam(self.parameters(), lr=0.01) #当前参数，学习率
        #开始将dataset分为不同的批块
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        while 1: #训练直到损失函数达到要求
            for data_block in dataloader: #对dataset中的每一个data块来说
                #先重置，优化器梯度清零
                optimizer.zero_grad()
                #得到训练集的标准输入输出
                x=data_block["x"]
                y=data_block["label"]
                #计算损失函数
                lose = self.get_loss(x, y)
                #更新权重
                lose.backward()#反向传播，计算梯度
                optimizer.step()#更新权重
            #一次对全部训练集训练结束后，检查是否达到可以训练结束的条件，即损失函数大小
            if self.get_loss(dataset[:]['x'], dataset[:]['label']) < 0.0200: #对于所有数据的损失函数
                break



#手写体识别
class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        #定义神经网络的隐藏层和输出层
        self.hidden_layer1 = Linear(input_size,200)
        self.hidden_layer2 = Linear(200,100)
        self.output_layer = Linear(100,output_size)



    #对一个batch块的输入，返回的是这个块中所有输入，被分在在各个类中的概率
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        #对batch_size个x，每个进行前向传播，再将结果存入batch_size个y中去
        hidden_layer_out1 = relu(self.hidden_layer1(x))
        hidden_layer_out2 = relu(self.hidden_layer2(hidden_layer_out1))
        y_predicted = self.output_layer(hidden_layer_out2)
        return y_predicted


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        #一个batch的损失函数
        return cross_entropy(self.run(x),y)

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        #先定义优化器
        optimizer = optim.Adam(self.parameters(), lr=0.01) #当前参数，学习率
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)#学习率调整策略
        #开始将dataset分为不同的批块
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        while 1: #训练直到损失函数达到要求
            for data_block in dataloader: #对dataset中的每一个data块来说
                #先重置，优化器梯度清零
                optimizer.zero_grad()
                #得到训练集的标准输入输出
                x=data_block["x"]
                y=data_block["label"]
                #计算损失函数
                lose = self.get_loss(x, y)
                #更新权重
                lose.backward()#反向传播，计算梯度
                optimizer.step()#更新权重
            scheduler.step()
            #一次对全部训练集训练结束后，检查是否达到可以训练结束的条件，即损失函数大小
            if dataset.get_validation_accuracy() >= 0.975: #对于所有数据的损失函数
                break



#语言识别  RNN
class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        # Initialize your model parameters here
        self.hidden_layer_node_num = 128
        self.hidden_layer = Linear(self.num_chars, self.hidden_layer_node_num)
        self.output_layer = Linear(self.hidden_layer_node_num, len(self.languages))


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        #对于一批（8个）单词来说，比如他有三个字母，先有一个8*47的矩阵，表达这一批单词里每一个单词第一个字母的独热码
        #就这样一个字母一个字母的给到这个函数，直到三次以后，每个单词的全部字母全部被给到神经网络了
        #输入：对于上述的例子来说，就是三个8*47的独热码矩阵，表达每个单词三位上每一位的独热码
        #输出：经过隐藏层后，得到8*5的每种语言的预测结果
        "*** YOUR CODE HERE ***"
        #先定义一个RNN的中间变量，他可以涵盖每次循环之前所有(字母)的信息
        tmp = tensor([0.0] * self.hidden_layer_node_num)
        #开始对一批单词的每一个字母去循环一次
        for letter_apperance_matrix_i in xs:
            tmp = relu(self.hidden_layer(letter_apperance_matrix_i) + tmp)
        #循环神经网络结束后进入输出层
        output = self.output_layer(tmp)
        return output


    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return cross_entropy(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        #开始训练
        for epoch in range(20):
            for batch in dataloader:
                #取数据
                batch_xs = batch['x']
                batch_y = batch['label']
                batch_xs = movedim(batch_xs, 0, 1)
                #清空梯度
                optimizer.zero_grad()
                #对一批，训练一下，计算损失
                loss = self.get_loss(batch_xs, batch_y)
                #更新权重
                loss.backward()
                optimizer.step()


        
#设计的卷积层，将来的作用是self.hidden_layer=Convolve(input,weight)
def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape    #上面两个表示input和weight是[0]*[1]的矩阵
    Output_Tensor = tensor(())  #设立一个空“矩阵”，存储卷积后的结果
    "*** YOUR CODE HERE ***"
    #用weight矩阵，对大矩阵input做卷积
    #重置输出矩阵的大小，为(输入大矩阵维度-卷积核小矩阵维度)
    Output_Tensor = empty((input_tensor_dimensions[0] - weight_dimensions[0] + 1, input_tensor_dimensions[1] - weight_dimensions[1] + 1))
    for i in range(input_tensor_dimensions[0] - weight_dimensions[0] + 1): #在列上滑动
        for j in range(input_tensor_dimensions[1] - weight_dimensions[1] + 1): #在行上滑动
            #在每一次滑动的块的位置，对第[i][j]个重叠部分的矩阵进行卷积
            coved_matrix = (input[i:i+weight_dimensions[0], j:j+weight_dimensions[1]] * weight) #对重叠部分求对应位置相乘的新矩阵
            Output_Tensor[i, j] = coved_matrix.sum() #对上一步得到的矩阵来每个位置求和
    "*** End Code ***"
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.


    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        #一个隐藏层
        self.hidden_layer = Linear(28*28, 200)
        self.output_layer = Linear(200, output_size)


    def run(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        hidden_layer_output = relu(self.hidden_layer(x))
        output = self.output_layer(hidden_layer_output)
        return output

 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        return cross_entropy(self.run(x), y)

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        #先定义优化器
        optimizer = optim.Adam(self.parameters(), lr=0.01) #当前参数，学习率
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)#学习率调整策略
        #开始将dataset分为不同的批块
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        while 1: #训练直到损失函数达到要求
            for data_block in dataloader: #对dataset中的每一个data块来说
                #先重置，优化器梯度清零
                optimizer.zero_grad()
                #得到训练集的标准输入输出
                x=data_block["x"]
                y=data_block["label"]
                #计算损失函数
                lose = self.get_loss(x, y)
                #更新权重
                lose.backward()#反向传播，计算梯度
                optimizer.step()#更新权重
            scheduler.step()
            #一次对全部训练集训练结束后，检查是否达到可以训练结束的条件，即损失函数大小
            if dataset.get_validation_accuracy() >= 0.975: #对于所有数据的损失函数
                break
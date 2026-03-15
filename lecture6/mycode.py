import numpy as np
import weakref
import graph_util as g
#定义一个变量类
class Variable:
    __array_priority__ = 999

    def __init__(self,input_data,name=None):
        #防止用户输入错误类型
        if input_data is not None and not isinstance(input_data,np.ndarray):
            raise TypeError("Variable类型的数据类型不正确，请输入np.ndarray类型的数据，而不是{}类型".format(type(input_data)))
        self.name = name
        self.value = input_data
        self.grad = None #梯度，默认为None
        self.creator = None #记录当前变量的创造函数

    #设置变量梯度
    def set_grad(self,grad):
        self.grad = grad
    
    #重置变量梯度
    def clear_grad(self):
        self.grad = None

    #默认不保留中间变量的导数
    def backward(self,retain_grad=True):
        if self.grad is None: 
            self.grad = Variable(np.ones_like(self.value)) #初始化梯度为1(向量1)
        #创建一个列表来存储需要处理的函数
        funcs = []
        visited = set() #用于跟踪已访问的函数，避免重复处理

        # 使用栈（或队列）广度优先收集所有函数（确保不重复）
        def collect_funcs(f):
            if f in visited:
                return
            visited.add(f)
            for x in f.input_variable:
                if x.creator is not None:
                    collect_funcs(x.creator)
            funcs.append(f)

        if self.creator is not None:
            collect_funcs(self.creator)

        # 按 generation 降序排序（从输出到输入）
        funcs.sort(key=lambda f: f.generation, reverse=True)

        for f in funcs:
            output_grads = [y().grad for y in f.output_variable]
            gradList = f.backward(*output_grads)
            if not isinstance(gradList, tuple):
                gradList = (gradList,)
            for i, x in enumerate(f.input_variable):
                if x.grad is None:
                    x.grad = gradList[i]
                else:
                    x.grad = x.grad + gradList[i]  # 梯度累加
            #中间变量不需要就置空
            if (not retain_grad):
                for y in f.output_variable:
                    y().grad = None
    #Variable的层数
    @property
    def generation(self):
        if self.creator is None:
            return 0
        else:
            return self.creator.generation+1
        
    #Variable的形状
    @property
    def shape(self):
        return self.value.shape
    
    #Variable的维度
    @property
    def ndim(self):
        return self.value.ndim
    
    #Varibale的大小
    @property 
    def size(self):
        return self.value.size
    
    #Variable的数据类型
    @property
    def dtype(self):
        return self.value.dtype
    
    @property
    def T(self):
        return self.transpose()
    
    #返回“最外围”的长度
    def __len__(self):
        return len(self.value)
    
    #格式化直接打印
    def __repr__(self):
        if self.value is None:
            return "variable(None)"
        p = str(self.value).replace('\n','\n' + ' ' * 9) #打印对齐
        return "variable(" + p + ")"
    
    #Variable内部方法，对齐ndarray用法
    def reshape(self,*shape):
        #支持输入一个（3，2）元组，也支持分别输入维度参数
        if len(shape) == 1 and isinstance(shape[0],(tuple,list)):
            shape = shape[0]
        return reshape(self,shape) #调用全局的reshape函数

    #转置
    def transpose(self):
        return transpose(self)
    
    def matmul(self,other):
        return matmul(self,other)
    
    #重载@运算符，等价于matmul函数
    def __matmul__(self,other):
        return matmul(self,other)
    
    def __rmatmul__(self,other):
        return matmul(other,self)

    #运算符重载

    def __add__(self,other):
        return add(self,other)
    
    def __radd__(self,other):
        return add(other,self)
    
    def __sub__(self,other):
        return sub(self,other)
    
    def __rsub__(self,other):
        return sub(other,self)
    
    def __mul__(self,other):
        return mul(self,other)
    
    def __rmul__(self,other):
        return mul(other,self)
    
    def __pow__(self,other):
        return pow(self,other)
    
    # def __rpow__(self,other):
    #     return pow(other,self)
    
    def __truediv__(self,other):
        return div(self,other)
    
    def __rtruediv__(self,other):
        return div(other,self)
    
    def __neg__(self):
        return neg(self)
    
    def __abs__(self):
        return abs(self)

class Parameter(Variable):
    
    def clear_grad(self):
        self.grad = None

#将标量输入转化为矢量
def as_array(input_data):
    if np.isscalar(input_data):
        return np.array(input_data) #标量转化为矢量
    return input_data

#将np.ndarray转换为Variable类型
def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(obj)

#定义一个函数类
class Function:
    def __init__(self):
        self.generation = 0   #层级属性
    #输入和输出都是Variable类型,输入接收任意数量的位置参数，并把他们打包成一个数组
    #一个剥壳，计算，套壳的过程
    def __call__(self,*input_variable:Variable):
        #入参可能是非Variable类型，需要先转换为Variable类型
        input_variable = [as_variable(temp_x) for temp_x in input_variable]
        max_gen = max([x.generation for x in input_variable])
        self.generation = max_gen + 1
        xs = [x.value for x in input_variable] #从输入变量元组中取出所有变量的值
        ys = self.forward(*xs) #将列表拆开，作为多个独立参数传给函数
        # 如果ys不是元组，要额外处理
        if not isinstance(ys,tuple):
            ys = (ys,)
        output_variable_list = [Variable(as_array(y)) for y in ys] #将计算结果封装成Variable类型
        for output_variable in output_variable_list:
            output_variable.creator = self #保存输出变量的创建函数 
        self.input_variable = input_variable #保存输入变量，用于反向传播
        self.output_variable = [weakref.ref(out) for out in output_variable_list] #保存输出变量，用于反向传播
        #返回多元素列表或者单元素
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]
    
    #正向计算，输入和输出都是ndarray类型
    def forward(self,*input_x):
        raise NotImplementedError() #尚未实现forward方法，抛出异常
    
    #反向传播，输入和输出都是Variable类型
    def backward(self,input_dy:Variable):
        raise NotImplementedError() #尚未实现backward方法，抛出异常

#求平方子类，继承自Function类
class Square(Function):
    #求平方函数
    def forward(self,input_x): 
        return input_x ** 2 #返回输入的平方
    
    def backward(self,input_dy):
        #已知求平方函数是单输入函数，那么需要把只包含一个元素的元组解包，再计算
        (x,) = self.input_variable
        return 2 * x.value * input_dy
#优化的求平方函数
def square(input_variable:Variable):
    input_variable = as_array(input_variable)
    return Square()(input_variable)

#求指数子类，继承自Function类
class Exp(Function):
    #求指数函数
    def forward(self,input_x):
        return np.exp(input_x)
    #反向传播函数，输入和输出都是Variable类型
    def backward(self,input_dy):
        (out_dy,) = self.output_variable
        return input_dy * out_dy().value
#优化的求指数函数
def exp(input_variable:Variable):
    input_variable = as_array(input_variable)
    return Exp()(input_variable)


#求正弦子类，继承自Function类
class Sin(Function):
    #求正弦函数
    def forward(self,input_x):
        return np.sin(input_x)
    #反向传播函数，输入和输出都是Variable类型
    def backward(self,input_dy):
        (x,) = self.input_variable
        return np.cos(x.value) * input_dy
#优化的求正弦函数
def sin(input_variable:Variable):
    return Sin()(input_variable)

#求余弦子类，继承自Function类
class Cos(Function):
    #求余弦函数
    def forward(self,input_x):
        return np.cos(input_x)
    #反向传播函数，输入和输出都是Variable类型
    def backward(self,input_dy):
        (x,) = self.input_variable
        return -np.sin(x.value) * input_dy
#优化的求余弦函数
def cos(input_variable:Variable):
    return Cos()(input_variable)

#求双曲正切子类，继承自Function类
class Tanh(Function):
    def forward(self,input_x):
        return np.tanh(input_x)
    
    def backward(self,input_dy):
        (out_dy,) =self.output_variable 
        return input_dy *(1-(out_dy.value)**2) 
#简化的求双曲正切函数
def tanh(input_variable:Variable):
    return Tanh()(input_variable)

#对数函数子类，继承自Function类
class Log(Function):
    def forward(self,input_x):
        return np.log(input_x)
    
    def backward(self,input_dy):
        (x,) = self.input_variable
        return input_dy/x
#简化的对数函数
def log(input_variable:Variable):
    return Log()(input_variable)    

#求绝对值子类，继承自Function类
class Abs(Function):
    #求绝对值函数
    def forward(self,input_x):
        return np.abs(input_x)
    #反向传播函数，输入和输出都是Variable类型
    def backward(self,input_dy):
        (input_x,) = self.input_variable
        return np.sign(input_x.value) * input_dy
#优化的求绝对值函数
def abs(input_variable:Variable):
    input_variable = as_array(input_variable)
    return Abs()(input_variable)

#取负值子类，继承自Function类
class Neg(Function):
    #求负函数
    def forward(self,input_x):
        return -input_x
    #反向传播函数，输入和输出都是Variable类型
    def backward(self,input_dy):
        return -1 * input_dy
#优化的取负值函数
def neg(input_variable:Variable):
    input_variable = as_array(input_variable)
    return Neg()(input_variable)

#求幂函数子类，继承自Function类
class Pow(Function):
    #接收幂次参数
    def __init__(self,power):
        if isinstance(power, Variable):
            power = power.value
        self.power = power
    #求幂函数
    def forward(self,input_x):
        return np.power(input_x,self.power)
    #反向传播函数，输入和输出都是Variable类型
    def backward(self,input_dy):
        (input_x,) = self.input_variable
        temp = self.power * (input_x ** (self.power - 1)) * input_dy
        return temp
#优化的求幂函数
def pow(input_variable:Variable,power:Variable):
    input_variable = as_array(input_variable)
    return Pow(power)(input_variable)

#加法类
class Add(Function):
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self,input1,input2):
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        return input1 + input2

    #backward方法的返回值个数和forward方法的输入参数数量一致
    def backward(self,input_dy:Variable):
        input_dy1,input_dy2 = input_dy,input_dy
        #处理广播情况
        if self.input1_shape!=self.input2_shape:
            input_dy1 = sum_to(input_dy1,self.input1_shape)
            input_dy2 = sum_to(input_dy2,self.input2_shape)
        return input_dy1,input_dy2
#简化后的add函数
def add(x0:Variable,x1:Variable):
    x1 = as_array(x1)
    x0 = as_array(x0)
    return Add()(x0,x1)

#减法类
class Sub(Function):
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self,input1,input2):
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        return input1 - input2
    
    def backward(self,input_dy):
        input_dy1,input_dy2 = input_dy,input_dy
        if self.input1_shape!=self.input2_shape:
            input_dy1 = sum_to(input_dy1,self.input1_shape)
            input_dy2 = sum_to(input_dy2,self.input2_shape)
        return input_dy1,-input_dy2
#简化后的sub函数
def sub(x0:Variable,x1:Variable):
    x1 = as_array(x1)
    x0 = as_array(x0)
    return Sub()(x0,x1)

#乘法类
class Mul(Function):
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self,input1,input2):
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        return input1 * input2
    
    def backward(self,input_dy:Variable):
        (x0,x1) = self.input_variable
        dy_1 = input_dy*x1
        dy_2 = input_dy*x0
        input_dy1,input_dy2 = input_dy,input_dy
        if self.input1_shape!=self.input2_shape:
            dy_1 = sum_to(dy_1,self.input1_shape)
            dy_2 = sum_to(dy_2,self.input2_shape)
        return dy_1 ,dy_2
#简化后的mul函数
def mul(x0:Variable,x1:Variable):
    x0 = as_array(x0)
    x1 = as_array(x1)
    return Mul()(x0,x1)

#除法类
class Div(Function):
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self,input1,input2):
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        return input1 / input2
    
    def backward(self,input_dy):
        (x0,x1) = self.input_variable
        dy_1 = input_dy/x1
        dy_2 = input_dy*x0*(-1)/(x1**2)
        if self.input1_shape!=self.input2_shape:
            dy_1 = sum_to(dy_1,self.input1_shape)
            dy_2 = sum_to(dy_2,self.input2_shape)
        return dy_1,dy_2
#简化后的div函数
def div(x0:Variable,x1:Variable):
    x1 = as_array(x1)
    x0 = as_array(x0)
    return Div()(x0,x1)
    
#数值微分，传入函数和变量，返回函数在这个变量处的微分
def numerical_differentiation(func,input_val,eps=1e-4):
    x0 = Variable(input_val.value - eps)
    x1 = Variable(input_val.value + eps)
    y0 = func(x0)
    y1 = func(x1)
    return (y1.value - y0.value) / (2 * eps)

#反向传播的逆操作工具函数
def util_sum_to(input_x,target_shape):
    y = input_x
    #处理广播对齐过程中新增的维度：input_x比target_shape多出来的“前导维度”
    while y.ndim > len(target_shape):
        y = y.sum(axis=0)
    for i,sx in enumerate(target_shape):
        if sx == 1:
            y = y.sum(axis=i,keepdims=True)
    return y

#变换形状子类
class Reshape(Function):
    def __init__(self,target_shape):
        self.origin_shape = None #先声明
        self.target_shape = target_shape

    def forward(self,x:np.array):
        self.origin_shape = x.shape #记录原始形状，反向传播时可以使用
        return np.reshape(x,self.target_shape)
    
    def backward(self,dy:Variable):
        #return as_variable(np.reshape(dy.value,self.origin_shape))  #方法一
        return reshape(dy,self.origin_shape)                         #方法二
#简化的变换形状函数
def reshape(input_x:Variable,shape):
    if input_x.shape == shape:
        return as_variable(input_x)
    return Reshape(shape)(as_array(input_x))

#转置子类
class Transpose(Function):
    def forward(self,input_x):
        return np.transpose(input_x)
    
    def backward(self,dy:Variable): 
        return transpose(dy)
#简化后的转置方法    
def transpose(input_x:Variable):
    return Transpose()(as_array(input_x))

#广播类
class BroadcastTo(Function):
    def __init__(self,target_shape):
        self.origin_shape = None #先声明
        self.target_shape = target_shape

    def forward(self,input_x):
        self.origin_shape = input_x.shape
        return np.broadcast_to(input_x,self.target_shape)
    
    def backward(self,dy):
        return sum_to(dy,self.origin_shape)
#简化后的广播函数
def broadcast_to(input_x:Variable,shape):
    if input_x.shape == shape:
        return as_variable(input_x)
    return BroadcastTo(shape)(as_array(input_x))

#求和类
class SumTo(Function):
    def __init__(self,target_shape):
        self.origin_shape = None
        self.target_shape = target_shape
    
    def forward(self,input_x:np.ndarray):
        self.origin_shape = input_x.shape #保存原始形状
        return util_sum_to(input_x,self.target_shape)
    
    def backward(self,dy:Variable):
        return broadcast_to(dy,self.origin_shape)
#简化后的广播求和
def sum_to(input_x:Variable,shape):
    if input_x.shape == shape:
        return as_variable(input_x)
    return SumTo(shape)(as_array(input_x))

class Sum(Function):
    """
    沿指定轴计算张量的元素总和
    """

    def __init__(self,axis=None,keepdims=False):
        self.axis = axis
        self.keepdims = keepdims #对齐numpy用法
        self.outout_shape_kept = None
        self.origin_shape = None

    def forward(self,input_x):
        """
        执行前向传播
        1.保存输入形状'self.origin_shape'，用于反向传播
        2.计算并保存'self.output_shape_kept'，记录如果forward阶段用了输出本该是什么
        shape，从而在backward阶段reshape
        3.使用np.sum执行实际的求和操作
        """
        self.origin_shape =  input_x.shape
        #如果不传axis，即把所有元素加起来，得出一个标量
        if self.axis is None:
            self.output_shape_kept = tuple(np.ones(input_x.ndim,dtype = int))
        else:
            #分别处理axis为int和tuple的情况
            if isinstance(self.axis,int):
                axis_tuple = (self.axis,)
            else:
                axis_tuple = self.axis
            #归一化轴索引（确保为正整数）
            normalized_axis = [ax % input_x.ndim for ax in axis_tuple]
            shape_list = list(input_x.shape)
            for ax in normalized_axis:
                shape_list[ax] = 1
            self.output_shape_kept = tuple(shape_list)
        #执行求和操作
        y = np.sum(input_x,axis=self.axis,keepdims=self.keepdims)
        return y

    def backward(self,dy):
        """
        执行反向传播
        1. 通过reshape调整梯度形状
        2.通过广播机制将梯度广播回原始输入形状
        """
        #将梯度reshape为“keepdims = True”时的形状
        dy_reshaped = reshape(dy,self.output_shape_kept)
        #将梯度广播回原始形状
        dx = broadcast_to(dy_reshaped,self.origin_shape)
        return dx
#简化后的通用求和函数
def sum(input_x,axis=None,keepdims=False):
    return Sum(axis,keepdims)(input_x)

#矩阵乘法类
class MatMul(Function):
    def forward(self,input_x,input_W):
        return input_x @ input_W
    
    def backward(self,dy:Variable):
        input_x,input_w = self.input_variable
        dx = matmul(dy,input_w.T)
        dW = matmul(input_x.T,dy)
        return dx,dW
#简化的矩阵乘法函数
def matmul(input_x:Variable,input_W:Variable):
    #调用Matmul方法，会跳转到基类的call方法，提取输入，然后取.value（类型为ndarray）作为forward的输入
    #因此forward的输入为ndarray类型，直接使用numpy的@运算即可
    return MatMul()(input_x,input_W)

#线性计算类
class Linear(Function):
    #入参和出参都是ndarray类型
    def forward(self,x,W,b):
        y = x.dot(W)
        if b is not None: 
            y += b
        return y
    
    def backward(self, gy:Variable):
        x,W,b = self.input_variable
        db = None if b.value is None else sum_to(gy,b.shape)
        dx = matmul(gy,W.T)
        dW = matmul(x.T,gy)
        return dx,dW,db
#简化后的线性计算函数
def linear(x,W,b=None):
    return Linear()(x,W,b)

#均方误差类
class MeanSquaredError(Function):
    def forward(self,input_x0,input_x1):
        diff = input_x0-input_x1
        y = (diff ** 2).sum()/len(diff)
        return y
    
    def backward(self,dy):
        x0,x1 = self.input_variable
        diff = x0 - x1
        dx0 = dy*diff*(2. /len(diff))
        dx1 = -dx0
        return dx0,dx1
#简化的均方误差计算函数
def mean_squared_error(x0,x1):
    return MeanSquaredError()(x0,x1)

#绝对差
def abs_loss(x0,x1):
    diff = abs(x1-x0)
    return sum(diff) / len(diff)

#验证矩阵反向传播(计算x梯度)
def numerical_gradient_matrix_x(f,x,W,eps=1e-4):
    #获取x的原始数据
    x_data = x.value
    grad = np.zeros_like(x_data)

    #对x的每个元素进行扰动
    for idx in np.ndindex(x_data.shape):
        x_plus = x_data.copy()
        x_minus = x_data.copy()
        #正向扰动
        x_plus[idx]=x_plus[idx]+eps
        y1=f(Variable(as_array(x_plus)),W)
        #负向扰动
        x_minus[idx] = x_minus[idx] - eps
        y2 = f(Variable(as_array(x_minus)),W)
        #中心差分法计算梯度
        temp = (y1-y2).value
        grad[idx] = temp/(2*eps)
    return grad

#验证矩阵反向传播（记录W梯度）
def numerical_gradient_matrix_w(f,x,W,eps=1e-4):
    #获取W的原始数据
    W_data = W.value
    grad = np.zeros_like(W_data)
    #对W的每个元素进行扰动
    for idx in np.ndindex(W_data.shape):
        W_plus = W_data.copy()
        W_minus = W_data.copy()
        #正向扰动
        W_plus[idx] = W_plus[idx] + eps
        y1 = f(x,Variable(as_array(W_plus)))
        #负向扰动
        W_minus[idx] = W_minus[idx] - eps
        y0 = f(x,Variable(as_array(W_minus)))
        #中心差分法计算梯度
        temp = (y1-y0).value
        grad[idx] = temp/(2*eps)
    return grad

#Sigmoid激活函数类
class Sigmoid(Function):
    def forward(self,input_x):
        return  1/(1+np.exp(-input_x))
    
    #反向传播函数，输入和输出都是Variable类型
    def backward(self,input_dy):
        (out_dy,) = self.output_variable
        return input_dy * out_dy().value*(1-out_dy().value)
#简化的sigmoid激活函数
def sigmoid(x:Variable):
    return Sigmoid()(x)

#ReLu激活函数类
class Relu(Function):
    def forward(self,input_x):
        self.mask = (x<=0) #保存x<=0的位置掩码，用于反向传播
        y = x.copy()
        y[self.mask]=0 #将<=0的元素置零
        return y
    def backward(self,input_dy):
        (x,)=self.input_variable
        #当x大于0时梯度为input_dy,否则梯度为0
        return input_dy * (x.value>0)
#简化的relu激活函数
def relu(x:Variable):
    return Relu()(x)

#Layer层
class Layer:
    def __init__(self):
        self.params_name = set() #初始化为无序的集合

    def __setattr__(self, name, value):
        #只收集Parameter,不搜集Variable和其他类型
        if isinstance(value,(Parameter,Layer)):
            self.params_name.add(name)
        #调用父类的setattr方法，否则不会真正为属性赋值
        super().__setattr__(name,value)

    def __call__(self,*inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs,tuple):
            outputs = (outputs,)
        #tuple不变，转换为list类型
        self.inputs,self.outputs = list(inputs),list(outputs)
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self,inputs):
        raise NotImplementedError
    
    #递归获取所有变量,yield方式
    def params(self):
        for name in self.params_name:
            #生成器可以一个一个返回参数
            obj =  self.__dict__[name]
            if isinstance(obj,Layer): #如果是Layer层，递归yield
                yield from obj.params()
            else:
                yield obj

    #清楚所有参数的梯度
    def clear_grads(self):
        for param in self.params():
            param.clear_grad()

#线性层
class LinearLayer(Layer):
    #可以不显式指定入参input_size，在forward中根据输入动态确定
    def __init__(self,output_size,input_size=None,need_bias=True,dtype=np.float32):
        super().__init__()
        self.input_size,self.output_size,self.dtype=input_size,output_size,dtype
        self.W = Parameter(None,name="W")
        if self.input_size is not None:
            self._init_W()
        self.b = None
        if need_bias:
            self.b = Parameter(np.zeros(output_size).astype(dtype),name="b")
    
        
    #延迟初始化
    def _init_W(self):
    #根据输入和输出的维度，初始化权重矩阵W，使用了Xavier初始化方法，避免梯度爆炸，梯度消失等问题
        self.W.value = np.random.randn(self.input_size, self.output_size).astype(self.dtype) * np.sqrt(
        2.0 / (self.input_size + self.output_size))
    #前向线性计算
    def forward(self,inputs):
        #如果之前没有指定输入维度，这里根据第一个输入的形状动态确定，并且初始化权重矩阵W
        if self.W.value is None:
            if self.input_size is None:
                self.input_size = inputs.shape[1]
                self._init_W()
            self._init_W()
        return linear(inputs,self.W,self.b)

#模型类
class Model(Layer):
    def plot(self,*inputs,to_file="model.png"):
        y = self.forward(*inputs)
        return g.plot_dot_graph(y,verbose=True,to_file=to_file)

#两层网络
class TwoLayerNet(Model):
    def __init__(self,hidden_size,output_size,dtype=np.float32):
        super().__init__()
        self.l1 = LinearLayer(hidden_size,dtype=dtype)
        self.l2 = LinearLayer(output_size,dtype=dtype)

    def forward(self,x):
        h = sigmoid(self.l1(x))
        return self.l2(h)

#任意N层网络模型，即MLP模型
class MultiLayerNet(Model):
    def __init__(self,hidden_size:list,output_size,dtype=np.float32):
        super().__init__()
        self.layers = []
        for i in range(len(hidden_size)):
            self.layers.append(LinearLayer(hidden_size[i],dtype=dtype))
            self.layers.append(sigmoid)
        #最后一层的输出形状是 output_size
        self.layers.append(LinearLayer(output_size,dtype=dtype))
    
    def forward(self,x):
        #最后一层不需要激活函数，所以循环到倒数第二层
        for layer in self.layers[:-1]:
            x = layer(x)
        #最后一层不需要激活函数
        return self.layers[-1](x)

    def params(self):
        #首先获取通过属性注册的参数
        yield from super().params()
        #然后遍历self.layers中的每个Layer对象，获取其参数
        for layer in self.layers:
            if isinstance(layer,Layer):
                yield from layer.params()


if __name__ == '__main__':
    # x = np.random.randn(100,1)
    # y = np.sin(2*np.pi*x)+np.random.randn(100,1)
    # l1 = LinearLayer(10) #输出维度是10
    # l2 = LinearLayer(1) #输出维度是1

    # model = Layer()
    # model.l1 = LinearLayer(10)
    # model.l2 = LinearLayer(1)
    # def predict(x):
    #     h = model.l1(x)
    #     y = sigmoid(h)
    #     return model.l2(y)
    
    # lr = 0.2 #学习率
    # iters = 100000 #迭代次数

    # for i in range(iters):
    #     y_predict = predict(x)
    #     loss = mean_squared_error(y,y_predict)
    #     model.clear_grads()
    #     loss.backward()
    #     for p in model.params():
    #         p.value -= lr*p.grad.value
    #     if i%100 ==0:
    #         print(f"iter {i},loss: {loss.value:.4f}")
    # x = np.random.randn(100,1)
    # y = np.sin(2*np.pi*x)+np.random.randn(100,1)

    # lr = 0.2
    # max_iter = 10000
    # hidden_size = 10
    # model = TwoLayerNet(hidden_size,1)
    # for i in range(max_iter):
    #     #前向传播
    #     y_predict = model(x)
    #     #计算损失
    #     loss =  mean_squared_error(y_predict,y)
    #     #重置权重并反向传播
    #     model.clear_grads()
    #     loss.backward()

    #     for param in model.params():
    #         param.value -= lr*param.grad.value
        
    #     if i % 100 == 0:
    #         print(f"迭代{i}: 损失{loss.value:.4f}")
    
    # model.plot(x)

        # 单层网络训练
    def train_single_layer_net(x, y, lr, iters, output_size, loss_func):
        model = LinearLayer(output_size)
        lastLoss = Variable(np.array(0))
        for epoch in range(iters):
            y_predit = model(x)
            loss = loss_func(y, y_predit)
            loss.backward()  # 损失函数反向传播
            # 更新参数
            for param in model.params():
                param.value -= lr * param.grad.value
            model.clear_grads()
            # # 打印损失值
            # if epoch % 100 == 0:
            #     print(f"{epoch}: loss={loss.value:.4f}")
            lastLoss = loss.value

        # 打印最后的损失值
        print(f"单层模型的最终损失值是 {lastLoss:.4f}")
        return model


    # 双层网络训练, 多了一个 hidden_size 参数
    def train_two_layer_net(x, y, lr, iters, hidden_size, output_size, loss_func):
        model = TwoLayerNet(hidden_size, output_size)
        lastLoss = Variable(np.array(0))
        for epoch in range(iters):
            y_predit = model(x)
            loss = loss_func(y, y_predit)
            loss.backward()  # 损失函数反向传播
            # 更新参数
            for param in model.params():
                param.value -= lr * param.grad.value
            model.clear_grads()
            # # 打印损失值
            # if epoch % 100 == 0:
            #     print(f"{epoch}: loss={loss.value:.4f}")
            lastLoss = loss.value
        # 打印最后的损失值
        print(f"双层模型的最终损失值是 {lastLoss:.4f}")
        return model


    # 多层网络[MLP]，多了一个 hidden_sizes 参数，是一个列表，每个元素是隐藏层的神经元数量
    def train_multi_layer_net(x, y, lr, iters, hidden_sizes, output_size, loss_func):
        model = MultiLayerNet(hidden_sizes, output_size)
        lastLoss = Variable(np.array(0))
        for epoch in range(iters):
            y_predit = model(x)
            loss = loss_func(y, y_predit)
            loss.backward()  # 损失函数反向传播
            # 更新参数
            for param in model.params():
                param.value -= lr * param.grad.value
            model.clear_grads()
             # 打印损失值
            if epoch % 100 == 0:
                print(f"{epoch}: loss={loss.value:.4f}")
            lastLoss = loss.value
        # 打印最后的损失值
        print(f"多层模型的最终损失值是 {lastLoss:.4f}")
        return model


    # 训练数据，从 -3 到 3 等间隔取 100 个点，然后 reshape 成 100 * 1 的向量
    x = Variable(np.linspace(0, 3, 100).reshape(100, 1))  # (100, 1)
    y = exp(x)  # 真实值

    lr = 0.03  # 学习率
    iters = 10000  # 迭代次数
    hidden_size = 50  # 双层网络时，中间隐藏层的神经元数量
    hidden_sizes = [hidden_size, hidden_size]  # 多层网络时，每个隐藏层的神经元数量，例如这里3层
    output_size = 1

    # 分别使用单层/双层/多层网络进行训练， 对比效果
    #model_1_trained = train_single_layer_net(x, y, lr, iters, output_size, abs_loss)
    #model_2_trained = train_two_layer_net(x, y, lr, iters, hidden_size, output_size, abs_loss)
    model_3_trained = train_multi_layer_net(x, y, lr, iters, hidden_sizes, output_size, abs_loss)

    # 预测
    test_x = Variable(np.array([1.5]))
   # y_predit_1 = model_1_trained(test_x)  # 模型1的预测值
    #y_predit_2 = model_2_trained(test_x)  # 模型2的预测值
    y_predit_3 = model_3_trained(test_x)  # 模型3的预测值
    y = exp(test_x)  # 真实值
   # print(f"单层模型预测的结果是 {y_predit_1.value} 真实值 {y.value}")
   # print(f"双层模型预测的结果是 {y_predit_2.value} 真实值 {y.value}")
    print(f"多层模型预测的结果是 {y_predit_3.value} 真实值 {y.value}")
    model_3_trained.plot(x)
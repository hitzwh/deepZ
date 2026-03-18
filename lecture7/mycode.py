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
    
    def __getitem__(self, item):
        return get_item(self,item)
    
    def __log__(self):
        return log(self)


#参数类，继承自Variable类
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
        self.mask = (input_x<=0)
        y = input_x.copy()
        y[self.mask]=0 #将<=0的元素置零
        return y
    def backward(self,input_dy):
        (x,)=self.input_variable
        #当x大于0时梯度为input_dy,否则梯度为0
        return input_dy * (x.value>0)
#简化的relu激活函数
def relu(x:Variable):
    return Relu()(x)

#切片类
class GetItem(Function):
    def __init__(self,slices):
        self.slices = slices
        self.x_shape = None
    
    def forward(self,x):
        self.x_shape = x.shape
        y = x[self.slices] #x,y是ndarray类型
        return y
    
    #切片操作，反向传播只需要把对应切片梯度赋过去即可，
    def backward(self,dy:Variable):
        #构造一个与原始输入形状相同的0数组
        dx = np.zeros(self.x_shape,dtype=dy.dtype)
        #np.add.at 可以实现“稀疏加法”,用于切片梯度还原
        np.add.at(dx,self.slices,dy.value)
        #最终要返回的Variable对象
        return Variable(dx)
#简化的切片函数
def get_item(x,slices):
    return GetItem(slices)(x)

#裁剪类
class Clip(Function):
    def __init__(self,x_min,x_max):
        self.x_min = x_min
        self.x_max = x_max
    
    def forward(self,input_x):
        return np.clip(input_x,self.x_min,self.x_max)
    
    def backward(self, dy:Variable):
        (x,) = self.input_variable
        mask = (x.value>=self.x_min) * (x.value<=self.x_max)
        dx = dy * mask
        return dx
#简化的clip函数
def clip(x,x_min,x_max):
    return Clip(x_min,x_max)(x)

#softmax和交叉熵一起算
def softmax_cross_entropy_simple(x,t):
    x,t = as_variable(x),as_variable(t)
    N = x.shape[0] #一般x的i的第一个维度是批量数据个数batch size
    p = softmax_simple(x)
    p = clip(p,1e-5,1.0) #防止0和1溢出问题
    log_p  = log(p)
    tlog_p = log_p[np.arange(N),t.value] #Python高级索引
    return -1 * sum(tlog_p)/N

#最大类
class Max(Function):
    def __init__(self,axis=None,keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.x_shape = None
        self.argmax = None

    def forward(self,x):
        #使用np的max函数计算最大值，注意指定轴方向和是否保持梯度
        y = np.max(x,axis=self.axis,keepdims=self.keepdims)
        self.x_shape = x.shape #记录输入数组的形状，供反向传播使用
        #记住哪些元素是最大值，因为反向传播时只有这些位置能获得梯度
        if  self.axis is None:
            #(x==y)得到的是一个布尔数组，形状与x相同
            #他的每个元素都是True或False，对应x中是否等于最大值y
            self.argmax = x == y
        else:
            if self.keepdims:
                y_broadcast = y
            else:
                y_broadcast = np.expand_dims(y,axis=self.axis)
            self.argmax = (x==y_broadcast)
        return y
    
    def backward(self,dy):
        dx = np.zeros(self.x_shape, dtype=dy.dtype)
        # 将 dy 扩展（广播）为与 x 相同的形状
        if self.axis is None:
            # 全局最大值：dy 是标量，直接赋值给所有最大值位置
            dx[self.argmax] = dy.value
        else:
            # 指定了轴：需要将 dy 扩展为与 x 相同的形状
            if self.keepdims:
                # keepdims=True 时，dy 形状如 (2,1)，需要广播到原始形状
                dy_broadcast = np.broadcast_to(dy.value, self.x_shape)
            else:
                # keepdims=False，dy 比 x 少一维，需要在 axis 处插入一维再广播
                dy_temp = np.expand_dims(dy.value, axis=self.axis)
                dy_broadcast = np.broadcast_to(dy_temp, self.x_shape)
            # 现在 dy_broadcast 的形状与 x 相同，可以用于广播赋值
            dx[self.argmax] = dy_broadcast[self.argmax]
        return Variable(dx)
#简化后的max函数
def mymax(x,axis=None,keepdims=False):
    return Max(axis,keepdims)(x)

#最小类
class Min(Function):
    def __init__(self,axis=None,keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.x_shape = None
        self.argmin = None

    def forward(self,x):
        #使用np的min函数计算最大值，注意指定轴方向和是否保持梯度
        y = np.min(x,axis=self.axis,keepdims=self.keepdims)
        self.x_shape = x.shape #记录输入数组的形状，供反向传播使用
        #记住哪些元素是最大值，因为反向传播时只有这些位置能获得梯度
        if  self.axis is None:
            #(x==y)得到的是一个布尔数组，形状与x相同
            #他的每个元素都是True或False，对应x中是否等于最大值y
            self.argmin = x == y
        else:
            if self.keepdims:
                y_broadcast = y
            else:
                y_broadcast = np.expand_dims(y,axis=self.axis)
            self.argmin = (x==y_broadcast)
        return y
    
    def backward(self,dy):
        dx = np.zeros(self.x_shape, dtype=dy.dtype)
        # 将 dy 扩展（广播）为与 x 相同的形状
        if self.axis is None:
            # 全局最小值：dy 是标量，直接赋值给所有最小值位置
            dx[self.argmin] = dy.value
            return Variable(dx)
        else:
            # 指定了轴：需要将 dy 扩展为与 x 相同的形状
            if self.keepdims:
                # keepdims=True 时，dy 形状如 (2,1)，需要广播到原始形状
                dy_broadcast = np.broadcast_to(dy.value, self.x_shape)
            else:
                # keepdims=False，dy 比 x 少一维，需要在 axis 处插入一维再广播
                dy_temp = np.expand_dims(dy.value, axis=self.axis)
                dy_broadcast = np.broadcast_to(dy_temp, self.x_shape)
        # 现在 dy_broadcast 的形状与 x 相同，可以用于广播赋值
        dx[self.argmin] = dy_broadcast[self.argmin]
        return Variable(dx)
#简化后的min函数
def mymin(x,axis=None,keepdims=False):
    return Min(axis,keepdims)(x)

#softmax类
class SoftMax(Function):
    def __init__(self,axis=1):
        self.axis = axis
    def forward(self,input_x):
        #防止数据溢出，进行缩放
        x_shift = input_x-input_x.max(axis=self.axis,keepdims=True)
        y = np.exp(x_shift)
        y/= y.sum(axis=self.axis,keepdims=True)
        return y
    
    def backward(self,dy):
        y = self.output_variable[0]
        dx = y*dy
        sum_dx = dx.sum(axis=self.axis,keepdims=True)
        dx -= y*sum_dx
        return dx
#简化的softman函数
def softmax(x,axis=1):
    return SoftMax(axis)(x)

#另一种计算方式，数学上稳定，避免溢出
def logsumexp(x,axis=1):
    m = x.max(axis=axis,keepdims=True)
    y = x - m 
    np.exp(y,out = y)
    s = y.sum(axix=axis,keepdims=True)
    np.log(s,out=s)
    m += s
    return m #最终返回log ∑ exp(xi)


#softmax与交叉熵同时计算
class SoftmaxCrossEntropy(Function):
    #返回标量损失值
    def forward(self,x,t):
        N = x.shape[0]
        log_z = logsumexp(x,axis=1)
        log_p = x - log_z #计算每个类别的对数概率
        log_p = log_p[np.arange[N],t.ravel()] #共N个元素，0,t[0];1,t[1]....
        y = -log_p.sum() / np.float32(N) #平均交叉熵损失
        return y
    
    def backward(self,dy):
        #反向传播： dL/dx = （y-one_hot(t)）/N
        x,t = self.input_variable
        N, _ =  x.shape

        dy *= 1/N
        y = softmax(x)

        #构造one-hot
        one_hot = np.zeros_like(y,dtype=np.float32)
        one_hot[np.arrange[N],t.value] = 1
        #softmax + crossentropy的合成梯度
        y = (y-one_hot) *dy #p-one_hot即是预测与真实之间的差异
        #正确类别的梯度是负数，错误类别的梯度是正数
        return y,None
#简化的交叉熵softmax计算
def softmax_cross_entropy(x,t):
    return SoftmaxCrossEntropy()(x,t)

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

#优化器类
class Optimizer:
    def __init__(self,model):
        self.target = model
        self.hooks = [] 

    def add_hook(self,hook):
        self.hooks.append(hook)

    def updates(self):
        params = self.target.params()
        #过滤掉梯度为None的参数
        params = [p for p in params if p.grad is not None]

        #调用钩子函数，用于权重衰减、梯度裁剪等工作
        for hook in self.hooks:
            hook(params)
    
        #逐个更新参数
        for param in params:
            self.update_one(param)
    
    #每个参数的更新方法，需要在子类实现
    def update_one(self,param):
        raise NotImplementedError
    
#随机梯度下降类，继承自Optimizer类
class SGD(Optimizer):
    def __init__(self,model,lr = 0.01):
        super().__init__(model)
        self.lr = lr
    
    def update_one(self, param):
        param.value -= self.lr * param.grad.value

#Momentum类
class Momentum(Optimizer):
    def __init__(self,model,lr=0.1,momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.v = {} #保存每个参数的动量项

    def update_one(self,param):
        if param.grad is None: #某些层可能不需要训练
            return
        grad = param.grad.value
        #初始化动量
        if param not in self.v:
            self.v[param] = np.zeros_like(grad) #初始化参数对应动量
        #提取动量
        v = self.v[param] 
        #计算动量更新
        #v[:]是Numpy的切片赋值语法，表示原地赋值
        #新速度 = 旧速度*学习率-学习率*梯度
        v[:] = self.momentum*v - self.lr*grad
        #参数更新，相比直接-=梯度，可以更好地保留速度、减少震荡
        param.value += v

#简单的softmax函数，假设参数x为二维数据
def softmax_simple(x,axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y,axis=axis,keepdims=True)
    return y/sum_y
      
if __name__ == '__main__':
    # a = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    # y = a[1]
    # y.backward()
    # print(y, a.grad)  # variable([4 5 6]) [[0 0 0] [1 1 1]]

    # 随机4个手写数字识别的输出结果，形状 (4, 10)
    # loss = softmax_cross_entropy_simple(np.random.rand(4, 10), np.array([2, 6, 9, 1]))
    # print(loss)
    pass


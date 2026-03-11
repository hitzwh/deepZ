import numpy as np
import weakref
import lecture5.graph_util as graph_util 
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

    #默认不保留中间变量的导数
    def backward(self,retain_grad=False):
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
            return self.creator.generation
        
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
    
    def backward(self,dy): 
        temp = dy.value
        return np.transpose(temp)

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

# ========== 辅助函数：逐元素数值梯度（用于标量输出） ==========
def numerical_gradient(f, x, eps=1e-4):
    """
    计算函数 f 在 x 处的数值梯度（f 必须返回标量 Variable）。
    x : Variable
    f : 函数，接受一个 Variable 并返回一个标量 Variable
    返回与 x.value 形状相同的 numpy 数组
    """
    x_val = x.value.copy()
    grad = np.zeros_like(x_val)
    it = np.nditer(x_val, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = x_val[idx]

        # 正扰动
        x_plus = x_val.copy()
        x_plus[idx] = orig + eps
        y_plus = f(Variable(x_plus)).value

        # 负扰动
        x_minus = x_val.copy()
        x_minus[idx] = orig - eps
        y_minus = f(Variable(x_minus)).value

        grad[idx] = (y_plus - y_minus) / (2 * eps)
        it.iternext()
    return grad

def test_sum():
    print("\n" + "="*60)
    print("开始测试 Sum 类与 sum 函数")
    print("="*60)

    # 测试用例： (输入形状, axis, keepdims, 是否为标量输出)
    test_cases = [
        # 1D 数组
        ((5,), None, False, True),
        ((5,), 0, False, True),          # axis=0 对1D等价于所有轴
        ((5,), 0, True, False),           # keepdims=True 输出形状 (1,)
        # 2D 数组
        ((3, 4), None, False, True),
        ((3, 4), 0, False, False),
        ((3, 4), 1, True, False),
        ((3, 4), (0, 1), False, True),    # 所有轴求和 -> 标量
        ((3, 4), (0, 1), True, False),    # keepdims=True 输出形状 (1,1)
        # 3D 数组
        ((2, 3, 4), None, False, True),
        ((2, 3, 4), 0, False, False),
        ((2, 3, 4), 1, True, False),
        ((2, 3, 4), (0, 2), False, False),
        ((2, 3, 4), (0, 1, 2), False, True),  # 所有轴求和 -> 标量
    ]

    for i, (shape, axis, keepdims, is_scalar_out) in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1}: shape={shape}, axis={axis}, keepdims={keepdims} ---")
        
        # 创建随机输入（转换为 Variable）
        np.random.seed(42 + i)  # 可重现
        data = np.random.randn(*shape).astype(np.float32)
        x = Variable(data)
        
        # 1. 正向传播验证
        y = sum(x, axis=axis, keepdims=keepdims)
        y_expected = np.sum(data, axis=axis, keepdims=keepdims)
        
        # 提取 y 的值（可能是 Variable 或 ndarray）
        if isinstance(y, Variable):
            y_val = y.value
        else:
            y_val = y  # 理论上 sum 总是返回 Variable，但保险
        
        assert np.allclose(y_val, y_expected), \
            f"正向结果错误: 得到 {y_val}, 期望 {y_expected}"
        print("✅ 正向传播结果正确")
        
        # 2. 反向传播验证（解析梯度应全为 1）
        y.backward()  # 默认 dy = 1
        x_grad = x.grad.value if x.grad is not None else None
        expected_grad = np.ones_like(data)
        assert x_grad is not None, "梯度未计算"
        assert np.allclose(x_grad, expected_grad), \
            f"解析梯度错误: 得到 {x_grad}, 期望 {expected_grad}"
        print("✅ 反向传播梯度正确（全1）")
        
        # 3. 如果输出是标量，额外进行数值梯度验证
        if is_scalar_out:
            # 定义函数 f(x) = sum(x, axis, keepdims) 返回标量
            def f(x_var):
                return sum(x_var, axis=axis, keepdims=keepdims)
            
            num_grad = numerical_gradient(f, x)
            assert np.allclose(num_grad, expected_grad, atol=1e-2), \
                f"数值梯度与解析梯度不匹配: 数值 {num_grad}, 解析 {expected_grad}"
            print("✅ 数值梯度验证通过")
        else:
            print("⏩ 输出非标量，跳过数值梯度验证")
    
    print("\n" + "="*60)
    print("所有 sum 测试通过！")
    print("="*60)

def temp_fun(x, y):
    return pow(x + 1, 2) * neg(y) - abs(x - y)

if __name__ == '__main__':
    # x = as_variable(np.array([[1, 2, 3], [4, 5, 6]]))
    # y = transpose(x)
    # z = transpose(x)

    # print(y)  # variable([[1 4] [2 5] [3 6]])
    # print(z)  # variable([[1 4] [2 5] [3 6]])
    # y.backward(True)
    # z.backward(True) 
    # print(y.grad)
    # print(z.grad)
    # print(x.grad)

    # y = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    # z = Variable(np.array([[[1, 2, 3], [4, 5, 6]]]))
    # print(y.transpose())  # variable([[1 4] [2 5] [3 6]])
    # print(z.T)  # variable([[[1] [4]] [[2] [5]]  [[3] [6]]])
    # x = np.array([1, 2, 3])  # 形状 shape 是 (3,)  维度数 ndim 是 1
    # y = np.broadcast_to(x, (2, 3))   # 形状shape (2,3) 维度数 ndim 是 2
    # print(y)  # [[1 2 3] [1 2 3]]

    x1 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    x2 = Variable(np.array([[1, 2, 3]]))
    y = x1 + x2
    y.backward()
    print("y:", y)  # [[2,4,6],[5,7,9]]
    print("x1.grad:", x1.grad)  # [[1 1 1] [1 1 1]]
    print("x2.grad:", x2.grad)  # [[1 1 1] [1 1 1]] 但实际上应该是 [2,2,2]
        # ========== 测试减法 ==========
    print("\n" + "="*50)
    print("测试减法：x1 - x2")
    x1 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    x2 = Variable(np.array([[1, 2, 3]]))
    y_sub = x1 - x2
    y_sub.backward()
    print("y_sub:", y_sub)  # 期望：[[0,0,0], [3,3,3]]
    print("x1.grad (减法):", x1.grad)  # 期望：[[1,1,1], [1,1,1]] （因为每个元素对y的贡献都是1）
    print("x2.grad (减法):", x2.grad)  # 期望：[[-1,-1,-1], [-1,-1,-1]] ？注意广播时x2被复制了两行，梯度应为每列求和，但这里由于x2形状是(1,3)，正确的梯度应该是[[-2,-2,-2]]（求和后形状(1,3)）。但当前实现Sub.backward返回(input_dy1, -input_dy2)，且广播处理只对输入形状不同的情况求和，所以x2.grad应该是[[-1,-1,-1],[-1,-1,-1]]？这取决于sum_to函数的效果。预期正确实现应是：梯度累加后形状恢复为(1,3)，值为[[-2,-2,-2]]。根据您提供的代码，可能因为sum_to函数问题导致结果不符，但测试语句本身应该输出实际结果供分析。

    # ========== 测试乘法 ==========
    print("\n" + "="*50)
    print("测试乘法：x1 * x2")
    x1 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    x2 = Variable(np.array([[1, 2, 3]]))
    y_mul = x1 * x2
    y_mul.backward()
    print("y_mul:", y_mul)  # 期望：[[1*1,2*2,3*3], [4*1,5*2,6*3]] = [[1,4,9], [4,10,18]]
    print("x1.grad (乘法):", x1.grad)  # 期望：x2广播后的值：[[1,2,3], [1,2,3]]
    print("x2.grad (乘法):", x2.grad)  # 期望：x1广播后按列求和：x1 = [[1,2,3],[4,5,6]]，对列求和得[[5,7,9]]，形状(1,3)

    # ========== 测试除法 ==========
    print("\n" + "="*50)
    print("测试除法：x1 / x2")
    x1 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    x2 = Variable(np.array([[1, 2, 3]]))
    y_div = x1 / x2
    y_div.backward()
    print("y_div:", y_div)  # 期望：[[1/1,2/2,3/3], [4/1,5/2,6/3]] = [[1,1,1], [4,2.5,2]]
    print("x1.grad (除法):", x1.grad)  # 期望：1/x2广播后的值：[[1/1,1/2,1/3], [1/1,1/2,1/3]] = [[1,0.5,0.3333], [1,0.5,0.3333]]
    print("x2.grad (除法):", x2.grad)  # 期望：-x1/(x2^2)广播后按列求和：x1/(x2^2) = [[1/1^2,2/2^2,3/3^2], [4/1^2,5/2^2,6/3^2]] = [[1,0.5,0.3333], [4,1.25,0.6667]]，求和后得[[-5, -1.75, -1.0]]，形状(1,3)

    #测试转置
    x = as_variable(np.array([[1,2,3],[4,5,6]]))
    trans = Transpose()
    res = trans.backward(x)
    print(res)

    #测试sum
    test_sum()

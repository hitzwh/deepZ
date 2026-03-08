import numpy as np
import weakref
import graph_util 
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
            if not retain_grad:
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
    
    #返回“最外围”的长度
    def __len__(self):
        return len(self.value)
    
    #格式化直接打印
    def __repr__(self):
        if self.value is None:
            return "variable(None)"
        p = str(self.value).replace('\n','\n' + ' ' * 9) #打印对齐
        return "variable(" + p + ")"
    
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
    
    #正向计算，输入和输出都是Variable类型
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
    #反向传播函数，输入和输出都是ndarray类型
    def backward(self,input_dy):
        (out_dy,) = self.output_variable
        return input_dy * out_dy
#优化的求指数函数
def exp(input_variable:Variable):
    input_variable = as_array(input_variable)
    return Exp()(input_variable)


#求正弦子类，继承自Function类
class Sin(Function):
    #求正弦函数
    def forward(self,input_x):
        return np.sin(input_x)
    #反向传播函数，输入和输出都是非Variable类型
    def backward(self,input_dy):
        return np.cos(self.input_variable.value) * input_dy
#优化的求正弦函数
def sin(input_variable:Variable):
    return Sin()(input_variable)

#求余弦子类，继承自Function类
class Cos(Function):
    #求余弦函数
    def forward(self,input_x):
        return np.cos(input_x)
    #反向传播函数，输入和输出都是非Variable类型
    def backward(self,input_dy):
        return -np.sin(self.input_variable.value) * input_dy
#优化的求余弦函数
def cos(input_variable:Variable):
    return Cos()(input_variable)

#求绝对值子类，继承自Function类
class Abs(Function):
    #求绝对值函数
    def forward(self,input_x):
        return np.abs(input_x)
    #反向传播函数，输入和输出都是非Variable类型
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
    #反向传播函数，输入和输出都是非Variable类型
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
    def forward(self,input1,input2):
        return input1 + input2

    #backward方法的返回值个数和forward方法的输入参数数量一致
    def backward(self,input_dy:Variable):
        return input_dy,input_dy
#简化后的add函数
def add(x0:Variable,x1:Variable):
    x1 = as_array(x1)
    x0 = as_array(x0)
    return Add()(x0,x1)

#减法类
class Sub(Function):
    def forward(self,input1,input2):
        return input1 - input2
    
    def backward(self,input_dy):
        return input_dy,-input_dy
#简化后的sub函数
def sub(x0:Variable,x1:Variable):
    x1 = as_array(x1)
    x0 = as_array(x0)
    return Sub()(x0,x1)

#乘法类
class Mul(Function):
    def forward(self,input1,input2):
        return input1 * input2
    
    def backward(self,input_dy):
        (x0,x1) = self.input_variable
        return input_dy*x1,input_dy*x0 
#简化后的mul函数
def mul(x0:Variable,x1:Variable):
    x0 = as_array(x0)
    x1 = as_array(x1)
    return Mul()(x0,x1)

#除法类
class Div(Function):
    def forward(self,input1,input2):
        return input1 / input2
    
    def backward(self,input_dy):
        (x0,x1) = self.input_variable
        return input_dy/x1,input_dy*x0*(-1)/(x1**2)
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

# 通用测试函数：减少重复代码，统一测试逻辑
def test_function(func, input_value, func_name, dy=1.0):
    """
    测试单个函数的正向计算、反向传播、数值微分对比
    :param func: 函数实例（如Square()）
    :param input_value: 输入数值（如1.0）
    :param func_name: 函数名称（用于打印）
    :param dy: 反向传播的输入梯度（默认1.0，即对输出求导）
    """
    print("="*50)
    print(f"测试函数：{func_name}，输入值：{input_value}")
    print("-"*50)
    
    # 1. 正向计算
    x = Variable(input_value)
    y = func(x)
    print(f"1. 正向计算结果：y = {func_name}({input_value}) = {y.value:.6f}")
    
    # 2. 反向传播（解析梯度）
    grad_analytic = func.backward(dy)
    print(f"2. 反向传播梯度（解析解）：dy/dx = {grad_analytic:.6f}")
    print(f"   输入变量的grad属性：x.grad = {x.grad}")
    
    # 3. 数值微分（数值解）
    grad_numerical = numerical_differentiation(func, x)
    print(f"3. 数值微分结果（数值解）：dy/dx ≈ {grad_numerical:.6f}")
    
    # 4. 对比解析解和数值解的误差
    error = abs(grad_analytic - grad_numerical)
    print(f"4. 解析解与数值解的误差：{error:.6e}")
    print(f"   误差是否小于1e-5：{'是' if error < 1e-5 else '否'}")
    print()


if __name__ == '__main__':
    # 运算符重载测试
    print("\n" + "=" * 60)
    print("\n运算符重载测试")
    print("-" * 40)

    # 测试加法运算符
    print("\n1. 加法运算符测试")
    x = as_variable(as_array(2))
    y = as_variable(as_array(3))
    z = x + y
    print(f"x + y = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试反向加法运算符
    x = as_variable(as_array(2))
    z = 3.0 + x
    print(f"3.0 + x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试乘法运算符
    print("\n2. 乘法运算符测试")
    x = as_variable(as_array(2))
    y = as_variable(as_array(3))
    z = x * y
    print(f"x * y = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试反向乘法运算符
    x = as_variable(as_array(2))
    z = 3.0 * x
    print(f"3.0 * x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试减法运算符
    print("\n3. 减法运算符测试")
    x = as_variable(as_array(5))
    y = as_variable(as_array(3))
    z = x - y
    print(f"x - y = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试反向减法运算符
    x = as_variable(as_array(3))
    z = 5.0 - x
    print(f"5.0 - x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试除法运算符
    print("\n4. 除法运算符测试")
    x = as_variable(as_array(6))
    y = as_variable(as_array(3))
    z = x / y
    print(f"x / y = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试反向除法运算符
    x = as_variable(as_array(2))
    z = 6.0 / x
    print(f"6.0 / x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试幂运算符
    print("\n5. 幂运算符测试")
    x = as_variable(np.array([[2,2],[2,2]]))
    z = x ** Variable(np.array([[1,2],[3,4]]))
    print(f"x ** z = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 注意，不实现反向幂运算

    # 测试取负运算符
    print("\n6. 取负运算符测试")
    x = as_variable(as_array(5))
    z = -x
    print(f"-x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试绝对值运算符
    print("\n7. 绝对值运算符测试")
    x = as_variable(as_array(-5))
    z = abs(x)
    print(f"abs(x) = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试组合运算符
    print("\n8. 组合运算符测试")
    x = as_variable(as_array(2))
    y = as_variable(as_array(3))
    z = (x + y) * (x - y)
    print(f"(x + y) * (x - y) = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试平方函数
    print("\n9. 平方函数测试")
    x = as_variable(as_array(5))
    z = square(x)
    print(f"x² = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    print("\n" + "=" * 60)
    print("计算两个函数在 x=1, y=2 处的梯度")
    print("\n函数1: z = x² + y²")
    x1 = as_variable(as_array(1))
    y1 = as_variable(as_array(2))
    z1 = add(square(x1), square(y1))
    print(f"输入: x = {x1.value}, y = {y1.value} 结果: z = {z1.value}")
    z1.backward()
    print(f"梯度: ∂z/∂x = {x1.grad.value} ∂z/∂y = {y1.grad.value}\n")
    print("-" * 40)
    print("函数2: z = 0.26(x² + y²) - 0.48xy")
    x2 = as_variable(as_array(1))
    y2 = as_variable(as_array(2))
    z2 = sub(mul(Variable(as_array(0.26)), add(square(x2), square(y2))),
             mul(Variable(as_array(0.48)), mul(x2, y2)))
    print(f"输入: x = {x2.value}, y = {y2.value} 结果: z = {z2.value}")
    z2.backward()
    print(f"\n梯度: ∂z/∂x =  {x2.grad.value} ∂z/∂y = {y2.grad.value}")

        # 之前的复杂函数
    def goldstein(x, y):
        z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
            (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
        return z

    x = Variable(np.array(1.0), "x")
    y = Variable(np.array(1.0), "y")
    z = goldstein(x, y)
    z.name = 'z'  # 给变量命名
    z.backward()

    graph_util.plot_dot_graph(z, verbose=True, to_file='goldstein.png')
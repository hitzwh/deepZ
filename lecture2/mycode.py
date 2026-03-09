import numpy as np

#定义一个变量类
class Variable:
    def __init__(self,input_data):
        #防止用户输入错误类型
        if input_data is not None and not isinstance(input_data,np.ndarray):
            raise TypeError("Variable类型的数据类型不正确，请输入np.ndarray类型的数据，而不是{}类型".format(type(input_data)))

        self.value = input_data
        self.grad = None #梯度，默认为None
        self.creator = None #记录当前变量的创造函数

    #设置变量梯度
    def set_grad(self,grad):
        self.grad = grad

    def backward(self):
        if self.grad is None: 
            self.grad = np.ones_like(self.value) #初始化梯度为1
        #创建一个列表来存储需要处理的函数
        funcs = []
        visited = set() #用于跟踪已访问的函数，避免重复处理

        #后序遍历收集所有函数
        def add_func(temp_func):
            if temp_func not in visited:
                # 添加输入变量的创建函数
                visited.add(temp_func)
                for temp_x in temp_func.input_variable:
                    if temp_x.creator is not None:
                        add_func(temp_x.creator)
                funcs.append(temp_func)
        
        if self.creator is not None:
            add_func(self.creator)
        #只有非用户输入的变量才有creator
        for f in funcs[::-1]:
            output_grads = [y.grad for y in f.output_variable] #取出当前节点函数所有输出的梯度
            gradList = f.backward(*output_grads) #计算每个输入元素的梯度
            #统一成元组
            if not isinstance(gradList,tuple):
                gradList = (gradList,)
            for i, x in enumerate(f.input_variable): #对输入中的多个参数逐个赋值
                if x.grad is None:
                    x.grad = gradList[i]
                else:
                    x.grad  = x.grad + gradList[i]

    #运算符重载
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
    
    def __rpow__(self,other):
        return pow(other,self)
    
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

#定义一个函数类
class Function:
    #输入和输出都是Variable类型,输入接收任意数量的位置参数，并把他们打包成一个数组
    #一个剥壳，计算，套壳的过程
    def __call__(self,*input_variable:Variable):
        xs = [x.value for x in input_variable] #从输入变量元组中取出所有变量的值
        ys = self.forward(*xs) #将列表拆开，作为多个独立参数传给函数
        # 如果ys不是元组，要额外处理
        if not isinstance(ys,tuple):
            ys = (ys,)
        output_variable_list = [Variable(as_array(y)) for y in ys] #将计算结果封装成Variable类型
        for output_variable in output_variable_list:
            output_variable.creator = self #保存输出变量的创建函数 
        self.input_variable = input_variable #保存输入变量，用于反向传播
        self.output_variable = output_variable_list #保存输出变量，用于反向传播
        #返回多元素列表或者单元素
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]
    
    #正向计算，输入和输出都是Variable类型
    def forward(self,*input_x):
        raise NotImplementedError() #尚未实现forward方法，抛出异常
    
    #反向传播，输入和输出都是ndarray类型
    def backward(self,input_dy):
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
    return Square()(input_variable)

#求指数子类，继承自Function类
class Exp(Function):
    #求指数函数
    def forward(self,input_x):
        return np.exp(input_x)
    #反向传播函数，输入和输出都是ndarray类型
    def backward(self,input_dy):
        (x,) = self.input_variable
        return np.exp(x.value) * input_dy
#优化的求指数函数
def exp(input_variable:Variable):
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
        return np.sign(self.input_variable.value) * input_dy
#优化的求绝对值函数
def abs(input_variable:Variable):
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
    return Neg()(input_variable)

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
    return Neg()(input_variable)

#求幂函数子类，继承自Function类
class Pow(Function):
    #接收幂次参数
    def __init__(self,power):
        self.power = power
    #求幂函数
    def forward(self,input_x):
        return np.power(input_x,self.power)
    #反向传播函数，输入和输出都是非Variable类型
    def backward(self,input_dy):
        return self.power * (self.input_variable.value ** (self.power - 1 )) * input_dy
#优化的求幂函数
def pow(input_variable:Variable,power):
    return Pow(power)(input_variable)

#加法类
class Add(Function):
    def forward(self,input1,input2):
        return input1 + input2

    #backward方法的返回值个数和forward方法的输入参数数量一致
    def backward(self,input_dy):
        return input_dy,input_dy
#简化后的add函数
def add(x0:Variable,x1:Variable):
    return Add()(x0,x1)

#减法类
class Sub(Function):
    def forward(self,input1,input2):
        return input1 - input2
    
    def backward(self,input_dy):
        return input_dy,-input_dy
#简化后的sub函数
def sub(x0:Variable,x1:Variable):
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
    # x0 = Variable(np.array((3,2)))
    # x1 = Variable(np.array((3,2)))
    # y = add(x0, x1)
    # print(y.value)
    # x0 = Variable(np.array(2))
    # x1 = Variable(np.array(3))
    # z = add(square(x0), square(x1))
    # z.backward()
    # print(z.value) # 2^2 + 3^2 = 13
    # print(x0.grad) # 4
    # print(x1.grad) # 6

    # x0 = Variable(np.array(3))
    # z = add(x0,x0)
    # z.backward()
    # print(f"x0.grad = {x0.grad}")
    # x0.set_grad(None)
    # y = add(x0,x0)
    # y.backward()
    # print(f"x0.grad = {x0.grad}")

    # x = Variable(np.array(1.0))
    # a = square(x)
    # b = exp(a)
    # c = exp(a)
    # y = add(b, c)
    # y.backward()
    # print(x.grad)  # 16.30969097075427  正确的梯度应该是 10.873127495050205

    x = Variable(np.array(1))
    y = Variable(np.array(2))
    z = add(square(x),square(y))
    z.backward()
    print(z.value)
    print(x.grad)
    print(y.grad)

    x.set_grad(None)
    y.set_grad(None)
    z2 = sub(mul(Variable(np.array(0.26)),z),mul(Variable(np.array(0.48)),mul(x,y)))
    z2.backward()
    print(z2.value)
    print(x.grad)
    print(y.grad)

    
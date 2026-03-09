import numpy as np

#定义一个变量类
class Variable:
    def __init__(self,input_data):
        self.value = input_data
        self.grad = None #梯度，默认为None

#定义一个函数类
class Function:
    #输入和输出都是Variable类型
    #一个剥壳，计算，套壳的过程
    def __call__(self,input_variable:Variable):
        x = input_variable.value #取出输入Variable中value属性的值
        y = self.forward(x) #具体计算，所有子类必须实现forward方法
        self.input_variable = input_variable #保存输入变量，用于反向传播
        output_variable = Variable(y) #将计算结果封装为Variable类型
        return output_variable #返回一个Variable对象
    
    #正向计算，输入和输出都是Variable类型
    def forward(self,input_x):
        raise NotImplementedError() #尚未实现forward方法，抛出异常
    
    #反向传播，输入和输出都是非Variable类型
    def backward(self,input_dy):
        raise NotImplementedError() #尚未实现backward方法，抛出异常

#求平方子类，继承自Function类
class Square(Function):
    #求平方函数
    def forward(self,input_x): 
        return input_x ** 2 #返回输入的平方
    
    def backward(self,input_dy):
        self.input_variable.grad = 2 * self.input_variable.value * input_dy #计算梯度，保存到输入变量的grad属性中
        return 2 * self.input_variable.value * input_dy

#求指数子类，继承自Function类
class Exp(Function):
    #求指数函数
    def forward(self,input_x):
        return np.exp(input_x)
    #反向传播函数，输入和输出都是非Variable类型
    def backward(self,input_dy):
        return np.exp(self.input_variable.value) * input_dy

#求正弦子类，继承自Function类
class Sin(Function):
    #求正弦函数
    def forward(self,input_x):
        return np.sin(input_x)
    #反向传播函数，输入和输出都是非Variable类型
    def backward(self,input_dy):
        return np.cos(self.input_variable.value) * input_dy

#求余弦子类，继承自Function类
class Cos(Function):
    #求余弦函数
    def forward(self,input_x):
        return np.cos(input_x)
    #反向传播函数，输入和输出都是非Variable类型
    def backward(self,input_dy):
        return -np.sin(self.input_variable.value) * input_dy

#求绝对值子类，继承自Function类
class Abs(Function):
    #求绝对值函数
    def forward(self,input_x):
        return np.abs(input_x)
    #反向传播函数，输入和输出都是非Variable类型
    def backward(self,input_dy):
        return np.sign(self.input_variable.value) * input_dy

#取负值子类，继承自Function类
class Neg(Function):
    #求负函数
    def forward(self,input_x):
        return -input_x
    #反向传播函数，输入和输出都是非Variable类型
    def backward(self,input_dy):
        return -1 * input_dy
    
#求幂函数子类，继承自Function类
class Pow(Function):
    #接收幂次参数
    def __init__(self,power):
        self.power = power
    #求幂函数
    def forward(self,input_x):
        return input_x ** self.power
    #反向传播函数，输入和输出都是非Variable类型
    def backward(self,input_dy):
        return self.power * (self.input_variable.value ** (self.power - 1 )) * input_dy

#组合封装Square，Exp，Square三个函数
class CompositeFunction(Function):
    def __init__(self):
        self.A = Square()
        self.B = Exp()
        self.C = Square()
    
    def forward(self,input_x):
        temp_val = Variable(input_x)
        a_output = self.A(temp_val)
        b_output = self.B(a_output)
        c_output = self.C(b_output)
        return c_output.value
    
    def backward(self,input_dy):
        dx = self.C.backward(input_dy)
        dx = self.B.backward(dx)
        dx = self.A.backward(dx)
        self.input_variable.grad = dx
        return dx
    
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

# 主函数，测试所有自定义函数
if __name__ == '__main__':
    # 测试参数说明：dy=1.0 表示对输出y求导（即计算dy/dx），是最常用的测试场景
    
    # 1. 测试平方函数 Square(x) = x²
    test_function(Square(), 2.0, "Square")
    
    # 2. 测试指数函数 Exp(x) = e^x
    test_function(Exp(), 1.0, "Exp")
    
    # 3. 测试正弦函数 Sin(x) = sin(x)（选x=π/4，cos(π/4)=√2/2≈0.7071）
    test_function(Sin(), np.pi/4, "Sin")
    
    # 4. 测试余弦函数 Cos(x) = cos(x)（选x=π/4，-sin(π/4)≈-0.7071）
    test_function(Cos(), np.pi/4, "Cos")
    
    # 5. 测试绝对值函数 Abs(x) = |x|（选x=3.0，sign(3)=1；避免x=0，因为0处不可导）
    test_function(Abs(), 3.0, "Abs")
    
    # 6. 测试取负值函数 Neg(x) = -x
    test_function(Neg(), 5.0, "Neg")
    
    # 7. 测试幂函数 Pow(x, 3) = x³（选x=2.0，导数3*2²=12）
    test_function(Pow(3), 2.0, "Pow(x^3)")
    
    # 8. 测试组合函数 CompositeFunction = (exp(x²))²
    # 解析梯度：d/dx [(exp(x²))²] = 2*exp(x²)*exp(x²)*2x = 4x*exp(2x²)
    # 当x=1.0时，梯度=4*1*exp(2*1)=4*7.389056≈29.556224
    test_function(CompositeFunction(), 1.0, "Composite (exp(x²))²")
    
    print("="*50)
    print("所有函数测试完成！")
class Calculator:
  result = 0;

  # 현재 코드에서는 불필요하나 이해를 위해 생성자 작성
  def __init__(self, initValue = 0):
    self.result = initValue;

  def add(self, value):
    self.result += value;
    return self.result;

  def sub(self, value):
    self.result -= value;
    return self.result;

  def mul(self, value):
    self.result *= value;
    return self.result;

  def div(self, value):
    self.result /= value;
    return self.result;

calculator = Calculator(10);

print(calculator.add(3));
print(calculator.sub(4));
print(calculator.mul(5));
print(calculator.div(2));
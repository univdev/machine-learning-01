class Calculator:
  result = 0;

  # 현재 코드에서는 불필요하나 이해를 위해 생성자 작성
  def __init__(self):
    self.result = 0;

  def add(self, value):
    self.result += value;
    return self.result;

calculator1 = Calculator();
calculator2 = Calculator();

print(calculator1.add(3));
print(calculator1.add(4));

print(calculator2.add(3));
print(calculator2.add(7));
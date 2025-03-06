class FourCalculator:
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


# FourCalculator를 CLI로 실행하면 아래가 실행되지만 다른 파일에서 참조하여 사용할 경우 아래 커맨드가 실행되지 않습니다.
if __name__ == "__main__":
  calculator = FourCalculator(10);

  print(calculator.add(3));
  print(calculator.sub(4));
  print(calculator.mul(5));
  print(calculator.div(2));
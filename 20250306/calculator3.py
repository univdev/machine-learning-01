result1 = 0;
result2 = 0;

def add1(value):
  global result1;
  result1 += value;
  return result1;

def add2(value):
  global result2;
  result2 += value;
  return result2;

print(add1(3));
print(add1(4));

print(add2(3));
print(add2(7));
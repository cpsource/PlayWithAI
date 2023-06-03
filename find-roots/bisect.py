def bisection(f, a, b, tol=1e-6):
  """
  Finds the root of the function f using the bisection method.

  Args:
    f: The function to find the root of.
    a: The lower bound of the interval.
    b: The upper bound of the interval.
    tol: The desired tolerance for the root.

  Returns:
    The root of the function f.
  """

  while abs(a - b) > tol:
    c = (a + b) / 2
    if f(c) < 0:
      a = c
    else:
      b = c
  return (a + b) / 2


def main():
  f = lambda x: 3*x**5 - 5*x**4 - 3*x**3 - 7*x - 10
  a = 0
  b = 1e-6
  root = bisection(f, a, b)
  print(root)


if __name__ == "__main__":
  main()

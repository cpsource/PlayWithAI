def newton_raphson(f, fp, x_0, tol=1e-6):
  """
  Finds the root of the function f using the Newton-Raphson method.

  Args:
    f: The function to find the root of.
    x_0: The initial guess for the root.
    tol: The desired tolerance for the root.

  Returns:
    The root of the function f.
  """

  while True:
    x_1 = x_0 - f(x_0) / fp(x_0)
    if abs(x_1 - x_0) < tol:
      return x_1
    x_0 = x_1


def main():
  f = lambda x: 3*x**5 - 5*x**4 - 3*x**3 - 7*x - 10
  fp= lambda x: 15*x**4 - 20*x**3 - 9*x**2 - 7
  x_0 = -1
  root = newton_raphson(f,fp, x_0)
  print(root)
  print(f(root))

if __name__ == "__main__":
  main()

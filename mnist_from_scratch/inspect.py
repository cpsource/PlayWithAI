import inspect

def explain_variable(variable):
  """
  Fully explains a variable.

  Args:
    variable: The variable to explain.

  Returns:
    The explanation of the variable.
  """

  explanation = ""
  explanation += "The variable '{}' is of type '{}'.".format(
      variable, type(variable))
  explanation += "\nIt is defined in line {} of the file '{}'.".format(
      inspect.getsourcelines(variable)[1], inspect.getfile(variable))
  explanation += "\nIts value is '{}'.".format(variable)
  return explanation


if __name__ == "__main__":
  variable = [10]
  explanation = explain_variable(variable)
  print(explanation)

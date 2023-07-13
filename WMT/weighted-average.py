def weighted_average(data_point, previous_points):
  """Returns the weighted average of a data point and the last three points.

  Args:
    data_point: The new data point.
    previous_points: A list of the last three data points.

  Returns:
    The weighted average of the data point and the last three points.
  """

  weight_new = 1.0
  weight_old = 0.5
  if len(previous_points) > 0:
    weight_old = 1.0 / len(previous_points)
  weighted_average = (weight_new * data_point) + (weight_old * sum(previous_points))
  return weighted_average/4


if __name__ == "__main__":
  data_point = 10.0
  previous_points = [5.0, 7.0, 8.0]
  weighted_average = weighted_average(data_point, previous_points)
  print(weighted_average)

  data = [5.0, 7.0, 8.0, 10.0]
  print((7+5)/2)
  print((7+8)/2)
  

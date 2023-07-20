import datetime

def compare_dates(date_str1, date_str2):
  """
  Compares two date strings and returns True if the first date is greater than the second date.

  Args:
    date_str1: The first date string.
    date_str2: The second date string.

  Returns:
    True if the first date is greater than the second date, False otherwise.
  """

  dt1 = datetime.datetime.strptime(date_str1, "%Y-%m-%d")
  dt2 = datetime.datetime.strptime(date_str2, "%Y-%m-%d")
  return dt1 > dt2

def add_one_day(dt):
  """
  Adds one day to the given datetime.

  Args:
    dt: The datetime to add one day to.

  Returns:
    The datetime one day after the given datetime.
  """

  return dt + datetime.timedelta(days=1)

if __name__ == "__main__":
  dt = datetime.datetime.now()
  new_dt = add_one_day(dt)
  x = f"{new_dt:%Y-%m-%d}"
  print(x)
  print(new_dt)


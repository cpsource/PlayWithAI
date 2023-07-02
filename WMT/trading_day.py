import datetime

def is_wall_street_trading_day(dt):
  """
  Returns True if the given datetime is a Wall Street trading day.

  Args:
    dt: The datetime to check.

  Returns:
    True if the datetime is a Wall Street trading day, False otherwise.
  """

  # Check if the datetime is a weekday.

  if dt.weekday() not in [0, 1, 2, 3, 4]:
    return False

  # Check if the datetime is a holiday.

  holidays = [
      datetime.datetime(2023, 1, 1),  # New Year's Day
      datetime.datetime(2023, 1, 20),  # Martin Luther King Jr. Day
      datetime.datetime(2023, 2, 20),  # Presidents' Day
      datetime.datetime(2023, 5, 29),  # Memorial Day
      datetime.datetime(2023, 7, 4),  # Independence Day
      datetime.datetime(2023, 9, 5),  # Labor Day
      datetime.datetime(2023, 11, 11),  # Veterans Day
      datetime.datetime(2023, 11, 24),  # Thanksgiving Day
      datetime.datetime(2023, 12, 25),  # Christmas Day
  ]
  for holiday in holidays:
    if dt == holiday:
      return False

  return True

if __name__ == "__main__":
  dt = datetime.datetime(2023, 6, 1)
  print(is_wall_street_trading_day(dt))


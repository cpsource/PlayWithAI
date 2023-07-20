import datetime
#from datetime import date

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

def is_wall_street_trading_day_str(date):
  cur_d = datetime.datetime.strptime(date, "%Y-%m-%d")

  if cur_d.year != 2023:
    print(f"oops, update the code in trading_day.py!")
    exit(0)
    
  tmp = is_wall_street_trading_day(cur_d)

  if 0:
    # lets cross-check
    k1 = cur_d.date()
    print(type(k1))
    print(dir(k1))
    if k1.is_working_day() and not k1.is_holiday():
      # market should be open
      if not tmp:
        print(f"disagreement #1 about markets being open")
      else:
        # market should be closed
        if tmp:
          print(f"disagreement #2 about markets being open")

  # return result
  return tmp

if __name__ == "__main__":
  dt = datetime.datetime(2023, 6, 1)
  print(is_wall_street_trading_day(dt))
  print(is_wall_street_trading_day_str("2023-01-01"))

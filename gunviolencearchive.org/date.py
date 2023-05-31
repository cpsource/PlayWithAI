# see details at https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

import datetime

def get_day_number(date):
  """
  Returns the day number from January 1, 2020, given a date of the form `YYYY-MM-DD`.

  Args:
    date: A string in the format `YYYY-MM-DD`.

  Returns:
    The day number from January 1, 2020.
  """

  parsed_date = datetime.datetime.strptime(date, '%B %d, %Y')
  return parsed_date.date().toordinal() - datetime.date(2020, 1, 1).toordinal() + 1

mydate="May 29, 2023"
print("get_day_number = ",get_day_number(mydate))

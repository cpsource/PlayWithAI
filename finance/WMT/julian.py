from datetime import datetime

# Get the current date
#current_date = datetime.now().date()
current_date = datetime.now()
print(dir(datetime),current_date)

# Calculate the Julian date
julian_date = current_date.toordinal() + 1721425.5

# Print the Julian date
print(julian_date)

#from datetime import datetime

# Get the current date and time
current_datetime = datetime.now()

# Calculate the Julian date
julian_date = current_datetime.toordinal() + (current_datetime - datetime.fromordinal(current_datetime.toordinal())).total_seconds() / (24 * 60 * 60)

# Print the Julian date
print(type(julian_date),julian_date)


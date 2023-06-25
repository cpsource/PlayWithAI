import datetime
import ephem

# Get the current date and time
now = datetime.datetime.now()

# Create a PyEphem object for the Moon
moon = pyephem.Moon()

# Calculate the phase of the Moon
phase = moon.phase(now)

# Print the phase of the Moon
print("The phase of the Moon is:", phase)

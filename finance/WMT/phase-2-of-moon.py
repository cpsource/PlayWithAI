import datetime
import ephem

# Get the current date and time
now = datetime.datetime.now()

# Create a PyEphem object for the Moon
moon = ephem.Moon()

moon.compute(now)

# Calculate the phase of the Moon
phase = moon.phase

# Print the phase of the Moon
print("The phase of the Moon is:", round(phase,2))

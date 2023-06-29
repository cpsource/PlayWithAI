import ephem

def get_moon_phase():
    # Compute the current phase of the moon
    moon = ephem.Moon()
    moon.compute(ephem.now())
    phase = moon.phase / 100  # Normalize phase value to range [0, 1]

    # Determine the phase name based on the phase value
    if phase < 0.03 or phase >= 0.97:
        phase_name = "New Moon"
    elif phase < 0.5:
        phase_name = "Waxing Crescent"
    elif phase < 0.53:
        phase_name = "First Quarter"
    elif phase < 0.97:
        phase_name = "Waxing Gibbous"
    else:
        phase_name = "Full Moon"

    return phase, phase_name

# Call the function to get the current moon phase
phase, current_phase = get_moon_phase()
print(f"Phase Percent: {phase},Current moon phase name:", current_phase)

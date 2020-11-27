import numpy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math

# Simulation parameters ...
freq = 100e6                # Hz
srate = 1e9                 # Hz (ADC sample rate)
repeat = 5                  # Repeat wave 5x
timestep = 1e-11            # Timestep for simulation (Not ADC sample rate) - at least 100x sample rate
adc_noise = 1.5             # Simulated ADC noise, normally distributed, in ADC counts
min_trig = 25               # Minimum and...
max_trig = 230              # ...maximum trigger levels to simulate 
# End

stime = (1 / freq) * repeat
theta_scale = (6.2831 / stime) * repeat
sample_interval = 1./srate

sample_step = int(sample_interval / timestep)
nsamples = stime / sample_interval

# Generate a real sine wave at given timestep.
sample_time = numpy.arange(0, stime, sample_interval)
real_time = numpy.arange(0, stime, timestep)
real_theta = (real_time * theta_scale) % 6.2831

real_sinewave = numpy.sin(real_theta, dtype=float)

# First sample to trigger on (to get some pretrigger)
first_samp = int(nsamples / (repeat * 2))

# Function to add noise to a variably-sized array
def add_noise(a, noiseAmt=0.0):
    return a + numpy.random.normal(0, noiseAmt, len(a))
    
# Function to sample an input wave at a given offset(0-1ns) and return the points as ADC values (8 bit)
def sample_wave(input_wave, offs):
    i = numpy.vectorize(numpy.int)
    return i(numpy.clip(add_noise((input_wave[int(offs * sample_step)::sample_step] * 127) + 127, noiseAmt=adc_noise), 0, 255))

# Function to trigger on a waveform passed in as an array and calculate the real offset and trigger point.
# The first instance of a rising edge is triggered on.  true_offset is passed only for debug purposes - not
# used to calculate trigger point.
def trigger_wave_rise(f, true_offset, level):
    # Find first rising edge.  Ignore first ~10 points (we want to trigger at centre-point of waveform.)
    last = f[0]
    for idx, pt in enumerate(f):
        if (idx > first_samp):
            if (pt > level) and (last <= level):
                break
        last = pt
    
    slopem4 = f[idx - 3] - f[idx - 4]
    slopem3 = f[idx - 2] - f[idx - 3]
    slopem2 = f[idx - 1] - f[idx - 2]
    slopem1 = f[idx - 0] - f[idx - 1]
    diff = level - f[idx]
    slopep1 = f[idx + 1] - f[idx + 0]
    slopep2 = f[idx + 2] - f[idx + 1]
    slopep3 = f[idx + 3] - f[idx + 2]
    slopep4 = f[idx + 4] - f[idx + 3]
    
    loc_slope1 = slopem1 + slopep1
    loc_slope2 = slopem2 - slopep2
    loc_slope3 = 0 # slopem3 + slopep3
    loc_slope4 = 0 # slopem4 - slopep4
    corr_slope = (loc_slope1 + (loc_slope2 * 0.25) + (loc_slope3 * 0.125) + (loc_slope4 * 0.0625))
    
    corr = -((diff / corr_slope) * 1.75)

    
    return (idx, corr)
    
t_level = min_trig  # Start at minimum
t_level_to_jitter = [None] * 255
t_level_list = list(range(255))

def refresh_jitter_graph():
    global fig, sub_ax
    
    sub_ax.clear()
    sub_ax.set_xlim(0, 255)
    sub_ax.set_ylim(0, 1000)
    sub_ax.set_title("Jitter v. Trigger Level")
    sub_ax.set_xlabel("Trigger Level")
    sub_ax.set_ylabel("Jitter (rms), ps")
    
    filt = [x for x in t_level_to_jitter if (x != None)]
    
    if len(filt) > 0:
        avg_error = sum(filt) / len(filt)
        print("AvgJitter = %.3f ps (based on %d samples)" % (avg_error, len(filt)))
        sub_ax.text(10, 10, "Overall average RMS jitter = %.1f ps" % avg_error, fontdict={'family' : 'monospace'})
    
    sub_ax.plot(t_level_list, t_level_to_jitter)

def iterate(i):
    global t_level, fig, main_ax
    errors_sq = []
    
    print("")
    print("** Iterate %d **" % i)
    print("")
    
    main_ax.clear()
    offs = 0
    
    for x in range(10):
        try:
            t = x * 0.05
            sampled = sample_wave(real_sinewave, t)
            (offs, corr) = trigger_wave_rise(sampled, t, level=t_level)
            errors_sq.append((t - corr) ** 2)
            main_ax.plot(sample_time - (offs * sample_interval) + (corr * sample_interval), sampled)
            print("X=%5d Offs=%5d Corr=%10.8f RealSub=%10.8f Err=%10.8f" % (x, offs, offs + corr, t, t - corr))
        except Exception as e:
            print("Trigger loss?", repr(e))

    main_ax.axhline(y=t_level, color="LightGray")
    main_ax.axvline(x=0, color="LightGray")
    
    main_ax.set_title("Waveform Preview")
    main_ax.set_xlabel("Time (ns)")
    main_ax.set_ylabel("8-bit Amplitude")
    main_ax.ticklabel_format(axis='x', style='sci', scilimits=(-9,-9))
    
    main_ax.text(0, 0,  "Trigger level   = %d" % t_level, fontdict={'family' : 'monospace'})

    if len(errors_sq) > 0:
        avg_error = math.sqrt(sum(errors_sq) / len(errors_sq))
        t_level_to_jitter[t_level] = avg_error * 1000
        main_ax.text(0, 20, "RMS jitter      = %.1f ps" % (avg_error * 1000), fontdict={'family' : 'monospace'})
    
    t_level += 1
    if (t_level > max_trig):
        t_level = min_trig
        
    refresh_jitter_graph()
        
    fig.canvas.draw()
    fig.canvas.flush_events()

fig, (main_ax, sub_ax) = plt.subplots(2)
ani = animation.FuncAnimation(fig, iterate, interval=1)

fig.set_size_inches(10, 10)
plt.show()
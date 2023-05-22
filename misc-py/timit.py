import time

def my_routine():
  for i in range(1000000):
    i * i

start_time = time.time()
my_routine()
end_time = time.time()

print(f"Elapsed time: {end_time - start_time}")

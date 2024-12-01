import time
import datetime

def perform_task():
    start_time = datetime.datetime.now()
    print(f"Task started at: {start_time}")
    
    while (datetime.datetime.now() - start_time).total_seconds() < 240:  # Run for 3 minutes (180 seconds)
        # Example calculation: Find prime numbers
        num = 2
        primes = []
        while num < 1000:  # Just an example range
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    break
            else:
                primes.append(num)
            num += 1

    end_time = datetime.datetime.now()
    print(f"Task completed at: {end_time}")

if __name__ == "__main__":
    perform_task()

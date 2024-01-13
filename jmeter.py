import subprocess
import time
import datetime
import signal
import sys
import random

def run_jmeter_test(test_plan, thread_count):
    print("running jmeter current thread count =", thread_count)
    # command = f"\"C:\\Users\\guanj\\OneDrive - University of Waterloo\\750 temp\\apache-jmeter-5.6.2\\bin\\jmeter\" -n -t {test_plan} -Jthread.count={thread_count} -DusePureIDs=true"
    command = f"\"/Users/zlc/Code/acmeair-driver/jmeter-5.6.2/bin/jmeter\" -n -t {test_plan} -Jthread.count={thread_count} -DusePureIDs=true"
    process = subprocess.Popen(command, shell=True)
    return process

def stop_jmeter_test(process):
    if process:
        process.terminate()
        print("JMeter test stopped.")

def adjust_thread_count(test_plan, new_thread_count):
    global current_test_process
    stop_jmeter_test(current_test_process)
    current_test_process = run_jmeter_test(test_plan, new_thread_count)

def signal_handler(sig, frame):
    print('Interrupt signal received. Stopping JMeter test...')
    stop_jmeter_test(current_test_process)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

duration = datetime.timedelta(hours=24)
end_time = datetime.datetime.now() + duration
initThreadCount = 50
current_test_process = run_jmeter_test('test.jmx', initThreadCount)

while True:
    if datetime.datetime.now() >= end_time:
        print("24 hours are over. Stopping JMeter test.")
        stop_jmeter_test(current_test_process)
        break

    try:
        time.sleep(300)  # Wait for 10 minutes
    except KeyboardInterrupt:
        # Optionally handle the KeyboardInterrupt exception if needed
        print('KeyboardInterrupt caught. Stopping JMeter test...')
        stop_jmeter_test(current_test_process)
        break
    random_change = random.randint(-40, 50) 
    random_ratio = random.uniform(0, 1)
    initThreadCount += random_change * random_ratio
    adjust_thread_count('test.jmx', initThreadCount)

print("Script execution completed. Exiting.")

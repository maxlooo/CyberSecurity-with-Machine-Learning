import time
import collections

# duration over which to look for anomalies
anomaly_window_seconds = 3600 

# threshold for simple anomaly detection
query_threshold = 8

# multiplier to determine how many times the average to set the threshold
avg_multiplier = 2

# global variable to store each user's recent history
recent_history = {}

# global variable to keep the average number of user queries in each hour
day_history = collections.deque(24*[query_threshold / avg_multiplier], 24)

def process_event(user, recent_history):
	'''call this function to update history when an event occurs'''
	cur_time = int(time.time())
	if user not in recent_history:
		recent_history[user] = []

	recent_history[user].append(cur_time)
	for x in recent_history[user]:
		if cur_time - x < anomaly_window_seconds:
			continue
	# print(x, " ", recent_history[user])
	return recent_history

def is_anomaly(user, recent_history):
	'''call this function whenever an event occurs for which you want
		to detect anomalies'''
	if user not in recent_history:
		return False
	if len(recent_history[user]) > query_threshold:
		return True
	else:
		return False

def hourly_threshold_update(recent_history):
	'''set the threshold for anomaly detection to be a multiple of the average queries per user in recent history'''
	cur_time = int(time.time())
	totalNumberOfQueries = 0
	if not recent_history:
		return query_threshold

	totalNumberOfQueries = sum([len(x) for x in recent_history.values()])
	print(recent_history.values())
	print("Number of Users: len(recent_history) = ", len(recent_history))
	print("totalNumberOfQueriesThisHour = ", totalNumberOfQueries)
	averageQueriesPerUserThisHour = float(totalNumberOfQueries) / len(recent_history)
	print("averageQueriesPerUserThisHour = ", averageQueriesPerUserThisHour)
	day_history.appendleft(averageQueriesPerUserThisHour)
	# print(day_history)
	averageOfHourlyAveragesThisDay = float(sum(day_history)) / len(day_history)
	print("Total average queries over length of a day: sum(day_history) = ", sum(day_history))
	print("Total hours over length of a day: len(day_history) = ", len(day_history))
	print("averageOfHourlyAveragesThisDay = ", averageOfHourlyAveragesThisDay)
	return (averageQueriesPerUserThisHour, averageOfHourlyAveragesThisDay)

if __name__ == '__main__':
	'''testing code'''
	for i in range(1012):
		process_event('a', recent_history)
	for i in range(6):
		process_event('b', recent_history)
	for i in range(3):
		process_event('c', recent_history)
		process_event('d', recent_history)
	
	currentAverageOfHourlyAveragesOfConnectionsLevel = hourly_threshold_update(recent_history)
	if currentAverageOfHourlyAveragesOfConnectionsLevel[1] > query_threshold:
		print("Average of Hourly Averages exceed hourly threshold of", query_threshold, ", Severe Anomaly = ", currentAverageOfHourlyAveragesOfConnectionsLevel[1])
	elif currentAverageOfHourlyAveragesOfConnectionsLevel[0] > query_threshold:
		print("Average Queries Per User This Hour exceed hourly threshold of", query_threshold, ", Anomaly = ", currentAverageOfHourlyAveragesOfConnectionsLevel[0])
	else: 
		print("Within hourly threshold of", query_threshold, ", no anomaly = ", currentAverageOfHourlyAveragesOfConnectionsLevel[0], currentAverageOfHourlyAveragesOfConnectionsLevel[1])
	print(day_history)
	print("a anomaly = ", is_anomaly('a', recent_history))
	print("b anomaly = ", is_anomaly('b', recent_history))
	print("c anomaly = ", is_anomaly('c', recent_history))
	print("d anomaly = ", is_anomaly('d', recent_history))

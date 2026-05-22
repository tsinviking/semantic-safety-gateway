import redis

client = redis.Redis(host='localhost', port=6379, decode_responses=True)

keys = client.keys('*')
print(f"Total keys in cache: {len(keys)}")
for key in keys:
    print(f"Key: {key}")
    # print(client.hgetall(key))
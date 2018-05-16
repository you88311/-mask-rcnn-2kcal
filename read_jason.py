
import json
with open("C:/Users/you88/Desktop/annotations/captions_train2014.json") as f:
    data = json.load(f)

print(type(data))
print(data)
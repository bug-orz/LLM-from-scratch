import json

idx=0

with open("./data/sample-1000.json") as fp:
    data=json.load(fp)

for i in range(int(len(data)/1000)):
    with open(f"./data/sample-{idx+2000}.json","w") as fp:
        json.dump(data[i*1000:(i+1)*1000],fp,indent=4)
    idx+=1
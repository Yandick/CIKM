import json

input_file = r"D:\SCUT\26_spring\CIKM\data\raw\amazon-food\Grocery_and_Gourmet_Food.jsonl"
with open(input_file, "r") as fin:
    for line in fin:
        data = json.loads(line)
        print(data)
        break
import os
import json

class Reporter:
    def __init__(self, experiment_path, filename):
        self.experiment_path = experiment_path
        self.file_name = filename
        self.output_file = os.path.join(experiment_path, self.file_name)
        self.data = {}

    #write the data in outputfile
    def __del__(self): 
        with open(self.output_file , 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    #save new data
    def report(self, key, data):
        self.data[key] = data

    #open file to read
    def load(self, filepath):
        self.data = json.load(open(filepath, 'r'))
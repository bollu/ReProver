import glob
import tqdm
import json
import pathlib

data = set()

for jpath in tqdm(glob.glob("dataset/*.json")):
    j = json.loads(open(jpath))
    record = dict()
    record["path"] = pathlib.Path(j["file_name"])
  
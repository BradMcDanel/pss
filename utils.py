import json

def load_jsonl(path):
    with open(path, 'r') as f:
        rows = [json.loads(line) for line in f]

    # convert to column format
    columns = {}
    for row in rows:
        for key, value in row.items():
            if key not in columns:
                columns[key] = []
            columns[key].append(value)

    return columns

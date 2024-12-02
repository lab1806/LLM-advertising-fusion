import json


def load_data(file_path='train.json',num_samples=2000):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data[:num_samples]


if __name__ == '__main__':

    #数据集切分
    dataset = load_data()
    with open('train_2000.json', 'w', encoding='utf-8') as file:
        for item in dataset:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
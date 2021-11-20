import json

def main():
    with open('../data/reviews_Baby_5.json', 'r') as f:
        data = json.load(f)
    print(data)
    # with open('../data/processed/data.json', 'w') as f:
    #     json.dump(data, f)
if __name__ == '__main__':
    main()
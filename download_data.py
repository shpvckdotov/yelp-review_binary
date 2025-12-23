from datasets import load_dataset
import pandas as pd



def main() -> None:
    dataset = load_dataset("yelp_polarity", cache_dir="/data")
    def save_dataset(dataset, split, path):
        df = pd.DataFrame(dataset[split])
        df.to_csv(path, index=False)

    save_dataset(dataset, 'train', 'data/yelp_train.csv')
    save_dataset(dataset, 'test', 'data/yelp_test.csv') 

if __name__ == "__main__":
    main()
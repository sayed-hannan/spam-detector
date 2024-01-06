import pandas as pd
import sys


class DataIngestion:
    def __init__(self, data_file: str):
        """
        Initialize the DataIngestion class with the path to the data file.

        :param data_file: str, path to the CSV data file.
        """
        self.data_file = data_file
        self.df = None

    def load_data(self) -> bool:
        """
        Load data from the CSV file into a DataFrame.

        :return: bool, True if data is successfully loaded, False otherwise.
        """
        try:
            self.df = pd.read_csv(self.data_file)
            return True
        except FileNotFoundError:
            return False

    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded DataFrame.

        :return: pd.DataFrame, the loaded data.
        """
        return self.df

    def ingest_data(self) -> pd.DataFrame:
        """
        Ingest data from the CSV file. This method loads data and provides it for further processing.

        :return: pd.DataFrame, the loaded data.
        :raises: FileNotFoundError if the data file is not found.
        """
        if self.load_data():
            return self.get_data()
        else:
            raise FileNotFoundError(f"Data file not found: {self.data_file}")


def main(data_file_path: str):
    data_ingestion = DataIngestion(data_file_path)
    try:
        df = data_ingestion.ingest_data()
        if df is not None:
            print("Data loaded successfully:")
            print(df.head())
        else:
            print("Failed to load data.")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_ingestion.py <data_file_path>")
        sys.exit(1)

    data_file_path = sys.argv[1]
    main(data_file_path)
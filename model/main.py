from cleandata import clean_data
from createmodel import create_model

def main():
    data = clean_data()
    create_model(data)
    
    print(data.head())
    print(data.describe())
    
    
    
if __name__ == "__main__":
        main()
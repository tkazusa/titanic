# coding=utf-8

# write code...
import os
from preprocess.preprocess2_usename import PreprocesserUsename


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")
ORG_CONCAT_DATA = os.path.join(ORG_DATA_DIR, "data.csv")

def main():
    list_col_fillna = ["Age", "Fare"]
    list_col_dummy = ["Sex", "Pclass", "Embarked"]
    list_col_drop = ["PassengerId", "Ticket", "Cabin", "Name"]

    prep = PreprocesserUsename()
    prep.fetch_origin_data(ORG_DATA_PATH=ORG_CONCAT_DATA)
    prep.fillna_median(list_col_fillna=list_col_fillna)
    prep.use_honorific()
    prep.dummy(list_col_dummy=list_col_dummy)
    prep.drop_column(list_col_drop=list_col_drop)
    prep.drop_na_samples()
    prep.save_train_data()
    prep.save_test_data()

if __name__ == "__main__":
    main()
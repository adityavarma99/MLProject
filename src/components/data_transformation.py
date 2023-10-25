
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

"""
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer    # pipeline for encoding and scaling
from sklearn.impute import SimpleImputer          # to handle missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocesssor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")  # ceate model and save pickel file, to save it we are creating preprocesssor_obj_file_path

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
## adding function inside the class of data transformation
    def get_data_transformer_object(self):                       # reason for creating get_data_transformer_object function is to create all my pickle file, which is resoponsile to converting categorical to numerical and scaling


        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            
            num_pipelines=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),                     # handling missing values by using median
                    ("scaler",StandardScaler())                                       # scaling the dataset
                ]
            )
            cat_pipelines=Pipeline(
                
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),              # handling missing values by using mode
                    ("one_hot_encoder",OneHotEncoder()),                              # applying one hot encoder
                    ("scaler",StandardScaler())                                       # scaling the dataset
                ]
            )

            logging.info("Categorical columns: {categorical_columns}")

            logging.info("Categorical columns: {numerical_columns}")

            # combining numerical pipeline with categorical pipeline together using column transformer
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipelines,numerical_columns),
                    ("cat_pipeline",cat_pipelines,categorical_columns)
                ]
            )

            return preprocessor

            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):    # in data transformation we are hadling cate and num values and creating pickle file

        try:
            train_df=pd.read_csv(train_path)   
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("obtainng preprocessing object")

            preprocessing_obj=self.get_data_transformer_object

            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
                )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # for saving the pickle file    # saved in file_path by dumping the pkl file from utils
            save_object(
                file_path=self.data_transformation_config.preprocesssor_obj_file_path,    
                obj=preprocessing_obj
            )
        
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesssor_obj_file_path,

            )
        
        except Exception as e:
            raise CustomException(e,sys)
        """




  

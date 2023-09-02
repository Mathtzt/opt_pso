import os
import pandas as pd

from datetime import datetime

class Utils:

    @staticmethod
    def create_folder(path, name, use_date = False):
        """
        Método responsável por criar a pasta no diretório passado como parâmetro.
        """
        if use_date:
            dt = datetime.now()
            day = dt.strftime("%d")
            mes = dt.strftime("%m")
            hour = dt.strftime("%H")
            mm = dt.strftime("%M")
            dirname_base = f"_{day}{mes}_{hour}{mm}"
            directory = name + dirname_base
        else:
            directory = name

        parent_dir = path

        full_path = os.path.join(parent_dir, directory)

        if os.path.isdir(full_path):
            return full_path
        else:
            os.mkdir(full_path)
            return full_path
        
    @staticmethod
    def save_experiment_as_csv(base_dir: str, dataframe: pd.DataFrame, filename: str):
        BASE_DIR = base_dir
        FILE_PATH = BASE_DIR + '/' + filename + '.csv'
        if not os.path.exists(FILE_PATH):
            dataframe.to_csv(FILE_PATH, index = False)
        else:
            df_loaded = pd.read_csv(FILE_PATH)
            dataframe_updated = pd.concat([df_loaded, dataframe], axis = 0)

            dataframe_updated.to_csv(FILE_PATH, index = False)

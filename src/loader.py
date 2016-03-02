import pandas as pd

_author_ = 'Stanley Hardt'

class Loader:
    def load_train_set(self, file_name):
        # Zaladowanie danych
        df = pd.read_table(file_name, sep=";", na_values=["NA", "n/a",'na', "nan", "NaN"], keep_default_na = False, low_memory=False)

        # Usuniecie wierszy ze nieinteresujacymi nas wartosciami w kolumnie res_name
        stop_list = ["DA","DC","DT", "DU", "DG", "DI","UNK", "UNX", "UNL",
                     "PR", "PD", "Y1", "EU", "N", "15P", "UQ", "PX4", "NAN", "n/a",'NaN']

        df = df[~df.res_name.isin(stop_list)]

        # Zapewnienie unikatowych wartosci res_name i pdb_code
        df.drop_duplicates(subset = ['res_name', 'pdb_code'], keep='first')

        # Usuniecie malo liczny grup res_name
        df_sizes = df.res_name.value_counts()
        accept_list = df_sizes[df_sizes > 4].index.values
        df = df[df.res_name.isin(accept_list)]

        filter_col = [col for col in list(df) if (col.startswith('part_00') or col == "res_name")]
        #dane do analizy
        df = df[filter_col]

        df = df.dropna( axis=0, how='any')


        # Usuniecie malo liczny grup res_name jeszcze raz
        df_sizes = df.res_name.value_counts()
        accept_list = df_sizes[df_sizes > 4].index.values
        df = df[df.res_name.isin(accept_list)]

        #podzial danych
        X = df.loc[:,'part_00_blob_electron_sum':'part_00_density_sqrt_E3']
        y = df.res_name
        return X,y

    def load_test_set(self, file_name):
        # Zaladowanie danych
        df = pd.read_table(file_name, sep=",", na_values=["NA", "n/a",'na', "nan", "NaN"], keep_default_na = False, low_memory=False)
        df = df.loc[:,'part_00_blob_electron_sum':'part_00_density_sqrt_E3']
        df = df.dropna( axis=0, how='any')
        return df

    def load_train_set_with_groups(self, file_name, groups_file_name):
        # Zaladowanie danych
        df = pd.read_table(file_name, sep=";", na_values=["NA", "n/a",'na', "nan", "NaN"], keep_default_na = False, low_memory=False)
        df_groups = pd.read_table(groups_file_name, sep=",", na_values=["NA", "n/a",'na', "nan", "NaN"], keep_default_na = False, low_memory=False)
        df['res_name'] = df_groups['res_name_group']

        # Zapewnienie unikatowych wartosci res_name i pdb_code
        df.drop_duplicates(subset = ['res_name', 'pdb_code'], keep='first')

        # Usuniecie malo liczny grup res_name
        df_sizes = df.res_name.value_counts()
        accept_list = df_sizes[df_sizes > 4].index.values
        df = df[df.res_name.isin(accept_list)]

        filter_col = [col for col in list(df) if (col.startswith('part_00') or col == "res_name")]
        #dane do analizy
        df = df[filter_col]

        df = df.dropna( axis=0, how='any')


        # Usuniecie malo liczny grup res_name jeszcze raz
        df_sizes = df.res_name.value_counts()
        accept_list = df_sizes[df_sizes > 4 ].index.values
        df = df[df.res_name.isin(accept_list)]

        #podzial danych
        X = df.loc[:,'part_00_blob_electron_sum':'part_00_density_sqrt_E3']
        y = df.res_name
        return X,y


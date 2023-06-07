# import packages
import pandas as pd
import numpy as np

def outlier_removal(df, numeric_cols):
    """
        Removes outliers from dataset

        Input arguments:
        df - dataframe to remove outliers from
        numeric_cols - list containing the names of the columns which are numeric

        Returns:
        Dataframe with outleirs removed

        """
    # get list of numeric columns to remove outliers from
    numeric_cols = list(set(numeric_cols) - set(["OPXVolume"]))
    numeric_cols = sorted(numeric_cols)

    df_no_outliers = df.copy()

    for p in numeric_cols:

        # remove missing values for each column
        df_no_outliers = df_no_outliers.dropna(subset = p)

        # Filter index values are that:
        # Above Upper bound
        max = df[p].mean() + 3*df[p].std()
        upper = df_no_outliers.index[df_no_outliers[p] >= max]

        # Below Lower bound
        min = df[p].mean() - 3*df[p].std()
        lower = df_no_outliers.index[df_no_outliers[p] < min]

        # join upper and lower bound values into one list
        bounds = np.hstack((lower, upper))
        bounds = list(bounds)

        # find index values that havent already been removed due to outliers in previous columns
        same_rows = set(list(bounds)) & set(list(df_no_outliers.index))

        # drop rows by index
        df_no_outliers.drop(index = list(same_rows), inplace = True)

    return df_no_outliers

def data_clean(fp, numeric_cols):
    """
        Cleans dataset

        Input arguments:
        fp - filepath to data
        numeric_cols - list containing the names of the columns which are numeric

        Returns:
        Cleaned dataset

    """
    
    df = pd.read_csv(fp)
    print(df.shape)
    
    # assume missing values for OPXVolume is because no lubricant was applied
    # NOTE in real project would verify this with client
    df["OPXVolume"] = df["OPXVolume"].fillna(0)

    # one hot encode location column
    df_clean = pd.get_dummies(df, columns = ["Location"])
    
    # if less than 10% of the data is missing or filled with non numeric values remove rows
    # else impute values with column mean
    for c in df_clean.columns.tolist():
        is_non_numeric = pd.to_numeric(df_clean[c], errors='coerce').isnull()
        char = df_clean[is_non_numeric][c].unique() 

        df_clean[c] = df_clean[c].replace(char, np.nan)
        df_clean[c] = pd.to_numeric(df_clean[c])

        if (df_clean[c].isnull().sum())/len(df_clean) < 0.1:
            df_clean.dropna(inplace = True)
        else:
            df_clean[c].fillna(np.mean(df_clean[c]))
    
    # ensure columns are of type numeric
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce', axis=1)

    # remove outliers
    print(df_clean.shape)
    df_final = outlier_removal(df_clean, numeric_cols)
    print(df_final.shape)

    # normalise numeric values to be between 1 and 0
    df_final[numeric_cols] = df_final[numeric_cols].apply(lambda x: (x-x.min())/ (x.max()-x.min()), axis=0)

    return df_final



def minority_balance_dataframe_by_multiple_categorical_variables(df, categorical_columns=None):
    """

    :param df: pandas.DataFrame
    :param categorical_columns: iterable of categorical columns names contained in {df}
    :return: balanced pandas.DataFrame
    """
    if categorical_columns is None or not all([c in df.columns for c in categorical_columns]):
        raise ValueError('Please provide one or more columns containing categorical variables')

    minority_class_combination_count = df.groupby(categorical_columns).apply(lambda x: x.shape[0]).min()
    df = df.groupby(categorical_columns).apply(
        lambda x: x.sample(minority_class_combination_count)
    ).drop(categorical_columns, axis=1).reset_index().set_index('level_1')

    df.sort_index(inplace=True)

    return df





"""Utils for ml."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
    r2_score
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import Bunch


def pythonic_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to pythonic style."""
    res = {}
    for col in df.columns:
        s = ''
        # Если в середине есть заглавная буква, добавляем знак подчеркивания
        for i in range(len(col)):
            if i > 0 and col[i].isupper() and col[i - 1].islower():
                s += '_' + col[i]
            else:
                s += col[i]

        s = s.strip().lower().replace(' ', '_')
        res[col] = s

    return df.rename(columns=res)


def regression_report(
    df, model_name, y_train, y_train_pred, y_test, y_test_pred, sort_by='R2'
):
    """Create report for regression models."""
    _df = df.copy()

    if len(_df) == 0:
        tuples = [
            ('R2', 'train'), ('R2', 'test'),
            ('MAE', 'train'), ('MAE', 'test'),
            ('RMSE', 'train'), ('RMSE', 'test'),
            ('MAPE', 'train'), ('MAPE', 'test'),
        ]
        index = pd.MultiIndex.from_tuples(tuples)
        _df = pd.DataFrame(index=index).T

    _df.loc[model_name] = {
        ('R2', 'train'): r2_score(y_train, y_train_pred),
        ('R2', 'test'): r2_score(y_test, y_test_pred),
        ('MAE', 'train'): mean_absolute_error(y_train, y_train_pred),
        ('MAE', 'test'): mean_absolute_error(y_test, y_test_pred),
        ('RMSE', 'train'): mean_squared_error(
            y_train, y_train_pred, squared=False
        ),
        ('RMSE', 'test'): mean_squared_error(
            y_test, y_test_pred, squared=False
        ),
        ('MAPE', 'train'): mean_absolute_percentage_error(
            y_train, y_train_pred
        ),
        ('MAPE', 'test'): mean_absolute_percentage_error(y_test, y_test_pred),
    }

    _df = _df.sort_values(by=(sort_by, 'test'))

    print(_df)

    return _df


def display_permutation_importance(
    result: Bunch,
    x_columns: list,
) -> pd.DataFrame:
    """View result of sklearn.inspection.permutation_importance function."""
    _res = pd.DataFrame(columns=['feature', 'mean', 'std'])
    for i in range(len(result.importances_mean)):
        _res.loc[i] = [
            x_columns[i],
            result.importances_mean[i],
            result.importances_std[i]
        ]
    _res = _res.sort_values(by='mean', ascending=False)
    return _res


class DisplayDfInPipe(BaseEstimator, TransformerMixin):
    """Print dataframe in pipeline."""

    def __init__(self, n: int = 10):
        self.n = n

    def fit(
        self: 'DisplayDfInPipe', x: pd.DataFrame, y: pd.DataFrame = None
    ) -> 'DisplayDfInPipe':
        """Fit."""
        return self

    def transform(
        self: 'DisplayDfInPipe', x: pd.DataFrame, y: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Transform."""
        print(x.head(self.n))

        return x


class OneHotEncoderMy(BaseEstimator, TransformerMixin):
    """OneHotEncoder with pandas support."""

    def __init__(self: 'OneHotEncoderMy', columns: list | str) -> None:
        if type(columns) == list:
            self.columns = columns
        elif type(columns) == str:
            self.columns = [columns]
        else:
            raise TypeError('Wrong type of parameter "columns"')

        self.encoders = {}
        for col in self.columns:
            self.encoders[col] = OneHotEncoder(
                sparse=False, dtype=np.uint8, handle_unknown='ignore'
            )

    def fit(
        self: 'OneHotEncoderMy',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> 'OneHotEncoderMy':
        """Fit."""
        self.columns = [col for col in self.columns if col in x.columns]
        for col in self.columns:
            self.encoders[col].fit(x[col].to_numpy().reshape(-1, 1))
        return self

    def transform(
        self: 'OneHotEncoderMy',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()
        for col in self.columns:
            feature_names = [f"{col}{x[2:]}" for x in
                             self.encoders[col].get_feature_names_out()]

            x_transformed = self.encoders[col].transform(
                df[col].to_numpy().reshape(-1, 1)
            )
            x_transformed_df = pd.DataFrame(
                data=x_transformed,
                columns=feature_names,
                index=df.index
            )
            df = pd.concat([df, x_transformed_df], axis=1).drop(columns=[col])
        return df


class OrdinalEncoderMy(BaseEstimator, TransformerMixin):
    """OrdinalEncoder with pandas support."""

    def __init__(self, columns: list | str):
        self.encoder = OrdinalEncoder(dtype=np.float64)
        if type(columns) == list:
            self.columns = columns
        elif type(columns) == str:
            self.columns = [columns]
        else:
            raise TypeError('Wrong type of parameter "columns"')

    def fit(
        self: 'OrdinalEncoderMy',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> 'OrdinalEncoderMy':
        """Fit."""
        # remove columns not exist in dataframe
        self.columns = [col for col in self.columns if col in x.columns]
        self.encoder.fit(x[self.columns])
        return self

    def transform(
        self: 'OrdinalEncoderMy',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()
        df[self.columns] = pd.DataFrame(
            data=self.encoder.transform(df[self.columns]),
            columns=df[self.columns].columns,
            index=df.index
        )
        return df


class StandardScalerMy(BaseEstimator, TransformerMixin):
    """StandardScaler with pandas support."""

    def __init__(
        self: 'StandardScalerMy',
        columns: list | str,
    ):
        if type(columns) == list:
            self.columns = columns
        elif type(columns) == str:
            self.columns = [columns]
        else:
            raise TypeError('Wrong type of parameter "columns"')

        self.scaler = StandardScaler()

    def fit(
        self: 'StandardScalerMy',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> 'StandardScalerMy':
        """Fit."""
        # remove columns not exist in dataframe
        self.columns = [col for col in self.columns if col in x.columns]
        self.scaler.fit(x[self.columns])
        return self

    def transform(
        self: 'StandardScalerMy',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df


class SimpleImputerMy(BaseEstimator, TransformerMixin):
    """SimpleImputer with pandas support."""

    def __init__(
        self: 'SimpleImputerMy',
        columns: list | str,
        strategy: str = 'mean',
        fill_value=None,
        missing_indicator=False
    ) -> None:
        if type(columns) == list:
            self.columns = columns
        elif type(columns) == str:
            self.columns = [columns]
        else:
            raise TypeError('Wrong type of parameter "columns"')

        self.missing_indicator = missing_indicator
        self.fill_value = fill_value
        self.strategy = strategy
        self.imputer = SimpleImputer(
            missing_values=np.nan, strategy=self.strategy,
            fill_value=self.fill_value
        )

    def fit(
        self: 'SimpleImputerMy',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> 'SimpleImputerMy':
        """Fit."""
        self.imputer.fit(x[self.columns])
        return self

    def transform(
        self: 'SimpleImputerMy',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()

        df[self.columns] = self.imputer.transform(df[self.columns])

        # добавляем признак с инфо о пропущенных значениях
        for col in self.columns:
            if self.missing_indicator:
                df[f'{col}_isna'] = df[col].isna()

        return df


class OutliersToNAN(BaseEstimator, TransformerMixin):
    """Replace outliers by NaN."""

    def __init__(
        self: 'OutliersToNAN',
        columns: list | str,
        low: float = None,
        high: float = None,
    ) -> None:
        if type(columns) == list:
            self.columns = columns
        elif type(columns) == str:
            self.columns = [columns]
        else:
            raise TypeError('Wrong type of parameter "columns"')

        self.low = low
        self.high = high

    def fit(
        self: 'OutliersToNAN',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> 'OutliersToNAN':
        """Fit."""
        # remove columns not exist in dataframe
        self.columns = [col for col in self.columns if col in x.columns]
        return self

    def transform(
        self: 'OutliersToNAN',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()
        for col in self.columns:
            if self.low is not None:
                df.loc[(df[col] < self.low), col] = np.NAN
            if self.high is not None:
                df.loc[(df[col] > self.high), col] = np.NAN
        return df


class IterativeImputerPd(BaseEstimator, TransformerMixin):
    """Iterative imputer."""

    def __init__(
        self: 'IterativeImputerPd',
        columns: list,
    ) -> None:
        self.columns = columns
        self.imputer = IterativeImputer()

    def fit(
        self: 'IterativeImputerPd',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> 'IterativeImputerPd':
        """Fit."""
        self.imputer.fit(x[self.columns])
        return self

    def transform(
        self: 'IterativeImputerPd',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()
        df[self.columns] = pd.DataFrame(
            data=self.imputer.transform(df[self.columns]),
            columns=df[self.columns].columns,
            index=df.index
        )
        return df


class SortInRow(BaseEstimator, TransformerMixin):
    """Sort values in row."""

    def __init__(
        self: 'SortInRow',
        columns: list,
        ascending: bool = True,
    ) -> None:
        self.ascending = ascending
        self.columns = columns

    def fit(
        self: 'SortInRow',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> 'SortInRow':
        """Fit."""
        return self

    def _sort(
        self: 'SortInRow',
        values: pd.Series,
    ) -> pd.Series:
        """Sort values in series."""
        return pd.Series(
            data=sorted(values.values, reverse=not self.ascending),
            index=values.index,
            name=values.name,
        )

    def transform(
        self: 'SortInRow',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Transform."""
        x[self.columns] = x[self.columns].apply(self._sort, axis=1)
        return x


class ToNumpyArray(BaseEstimator, TransformerMixin):
    """Convert dataframe to numpy.

    Some algorithms throw UserWarning, when get DataFrame."""

    def fit(
        self: 'ToNumpyArray',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> 'ToNumpyArray':
        """Fit."""
        return self

    def transform(
        self: 'ToNumpyArray',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> np.ndarray:
        """Transform."""
        return x.values


class NewFeatureFromCluster(BaseEstimator, TransformerMixin):
    """Generate new feature with KMeans clustering."""

    def __init__(
        self: 'NewFeatureFromCluster',
        new_feature_name: str,
        based_on: list,
        n_clusters: int,
    ) -> None:
        self.new_feature_name = new_feature_name
        self.based_on = based_on
        self.n_clusters = n_clusters

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
        )

    def fit(
        self: 'NewFeatureFromCluster',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> 'NewFeatureFromCluster':
        """Fit."""
        self.kmeans.fit(x[self.based_on])
        return self

    def transform(
        self: 'NewFeatureFromCluster',
        x: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Transform."""
        x[self.new_feature_name] = self.kmeans.predict(x[self.based_on])
        x[self.new_feature_name] = x[self.new_feature_name].astype('category')
        return x


if __name__ == '__main__':
    test = 'NewFeatureFromCluster'

    if test == 'SortInRow':
        df = pd.DataFrame(
            {
                'a': [0, 1, 2, 3],
                'b': [0, 2, 1, 3],
                'c': [0, 1, 2, 3],
                'd': [0, 3, 2, 3],
            }
        )
        print('before:')
        print(df)
        sort_in_row = SortInRow(['b', 'c'], ascending=False)
        df = sort_in_row.transform(df)
        print('after:')
        print(df)

    if test == 'NewFeatureFromCluster':
        df = pd.DataFrame(
            {
                'a': [0, 1, 2, 3],
                'b': [0, 2, 9, 8],
                'c': [0, 1, 10, 9],
                'd': [0, 3, 2, 3],
            }
        )
        print('before:')
        print(df)
        kmeans = NewFeatureFromCluster(
            new_feature_name='new',
            based_on=['b', 'c'],
            n_clusters=2,
        )
        print('after:')
        print(kmeans.fit_transform(df))

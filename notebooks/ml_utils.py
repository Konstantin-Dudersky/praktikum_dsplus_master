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
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler
)
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


class PipeOneHotEncoder(BaseEstimator, TransformerMixin):
    """OneHotEncoder with pandas support."""

    def __init__(
        self: 'PipeOneHotEncoder',
        columns: list | str,
        drop: str = None,
    ) -> None:
        self.feature_names_in_ = None
        if type(columns) == list:
            self.columns = columns
        elif type(columns) == str:
            self.columns = [columns]
        else:
            raise TypeError('Wrong type of parameter "columns"')

        self.drop = drop
        self.encoders = {}
        for col in self.columns:
            self.encoders[col] = OneHotEncoder(
                sparse=False,
                dtype=np.uint8,
                handle_unknown='ignore',
                drop=self.drop,
            )

    def fit(
        self: 'PipeOneHotEncoder',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> 'PipeOneHotEncoder':
        """Fit."""
        self.columns = [col for col in self.columns if col in x.columns]
        for col in self.columns:
            self.encoders[col].fit(x[col].to_numpy().reshape(-1, 1))
        return self

    def transform(
        self: 'PipeOneHotEncoder',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()
        for col in self.columns:
            feature_names = [
                f'{col}{x[2:]}' for x in
                self.encoders[col].get_feature_names_out()
            ]

            x_transformed = self.encoders[col].transform(
                df[col].to_numpy().reshape(-1, 1),
            )
            x_transformed_df = pd.DataFrame(
                data=x_transformed,
                columns=feature_names,
                index=df.index,
            )
            df = pd.concat([df, x_transformed_df], axis=1).drop(columns=[col])

        self.feature_names_in_ = df.columns
        return df

    def get_feature_names_out(
        self: 'PipeOneHotEncoder',
        input_features: list | str = None,
    ) -> list:
        """Get output feature names for transformation."""
        return self.feature_names_in_


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
            index=df.index,
        )
        return df


class PipeStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler with pandas support."""

    def __init__(
        self: 'PipeStandardScaler',
        columns: list = None,
    ) -> None:
        """Construct PipeStandardScaler.

        :param columns: Columns for scale. If None, scale all columns with
        float type.
        """
        self.feature_names_in_ = None
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(
        self: 'PipeStandardScaler',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> 'PipeStandardScaler':
        """Fit."""
        if self.columns is None:
            self.columns = x.select_dtypes(include=[float]).columns
        else:
            # remove columns not exist in dataframe
            self.columns = [col for col in self.columns if col in x.columns]
        self.scaler.fit(x[self.columns])
        return self

    def transform(
        self: 'PipeStandardScaler',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        self.feature_names_in_ = list(df.columns)
        return df

    def get_feature_names_out(
        self: 'PipeStandardScaler',
        input_features: list | str = None,
    ) -> list:
        """Get output feature names for transformation."""
        return self.feature_names_in_


class PipeSimpleImputer(BaseEstimator, TransformerMixin):
    """SimpleImputer with pandas support."""

    def __init__(
        self: 'PipeSimpleImputer',
        columns: list | str,
        strategy: str = 'mean',
        fill_value=None,
        missing_indicator: bool = False,
    ) -> None:
        self.feature_names_in_ = None
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
            fill_value=self.fill_value,
        )

    def fit(
        self: 'PipeSimpleImputer',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> 'PipeSimpleImputer':
        """Fit."""
        self.imputer.fit(x[self.columns])
        return self

    def transform(
        self: 'PipeSimpleImputer',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()

        df[self.columns] = pd.DataFrame(
            data=self.imputer.transform(df[self.columns]),
            columns=df[self.columns].columns,
            index=df[self.columns].index,
        ).astype(df[self.columns].dtypes.to_dict())

        # добавляем признак с инфо о пропущенных значениях
        for col in self.columns:
            if self.missing_indicator:
                df[f'{col}_isna'] = df[col].isna()
        self.feature_names_in_ = df.columns
        return df

    def get_feature_names_out(
        self: 'PipeSimpleImputer',
        input_features: list | str = None,
    ) -> list:
        """Get output feature names for transformation."""
        return self.feature_names_in_


class OutliersToNAN(BaseEstimator, TransformerMixin):
    """Replace outliers by NaN."""

    def __init__(
        self: 'OutliersToNAN',
        columns: list | str,
        low: float = None,
        high: float = None,
    ) -> None:
        self.feature_names_in_ = None
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
        y: pd.DataFrame = None,
    ) -> 'OutliersToNAN':
        """Fit."""
        # remove columns not exist in dataframe
        self.columns = [col for col in self.columns if col in x.columns]
        return self

    def transform(
        self: 'OutliersToNAN',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()
        for col in self.columns:
            if self.low is not None:
                df.loc[(df[col] < self.low), col] = np.NAN
            if self.high is not None:
                df.loc[(df[col] > self.high), col] = np.NAN
        self.feature_names_in_ = df.columns
        return df

    def get_feature_names_out(
        self: 'OutliersToNAN',
        input_features: list | str = None,
    ) -> list:
        """Get output feature names for transformation."""
        return self.feature_names_in_


class PipeIterativeImputer(BaseEstimator, TransformerMixin):
    """Iterative imputer."""

    def __init__(
        self: 'PipeIterativeImputer',
        columns: list,
    ) -> None:
        self.feature_names_in_ = None
        self.columns = columns
        self.imputer = IterativeImputer()

    def fit(
        self: 'PipeIterativeImputer',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> 'PipeIterativeImputer':
        """Fit."""
        self.imputer.fit(x[self.columns])
        return self

    def transform(
        self: 'PipeIterativeImputer',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        df = x.copy()
        df[self.columns] = pd.DataFrame(
            data=self.imputer.transform(df[self.columns]),
            columns=df[self.columns].columns,
            index=df.index,
        )
        self.feature_names_in_ = df.columns
        return df

    def get_feature_names_out(
        self: 'OutliersToNAN',
        input_features: list | str = None,
    ) -> list:
        """Get output feature names for transformation."""
        return self.feature_names_in_


class SortInRow(BaseEstimator, TransformerMixin):
    """Sort values in row."""

    def __init__(
        self: 'SortInRow',
        columns: list,
        ascending: bool = True,
    ) -> None:
        self.feature_names_in_ = None
        self.ascending = ascending
        self.columns = columns

    def fit(
        self: 'SortInRow',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
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
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        x[self.columns] = x[self.columns].apply(self._sort, axis=1)
        self.feature_names_in_ = x.columns
        return x

    def get_feature_names_out(
        self: 'SortInRow',
        input_features: list | str = None,
    ) -> list:
        """Get output feature names for transformation."""
        return self.feature_names_in_


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


class PipePolynomialFeatures(BaseEstimator, TransformerMixin):

    def __init__(
        self: 'PipePolynomialFeatures',
        columns: list,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
    ) -> None:
        self.feature_names_in_ = None
        self.columns = columns
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self.poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )

    def fit(
        self: 'PipePolynomialFeatures',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> 'PipePolynomialFeatures':
        """Fit."""
        self.poly.fit(x[self.columns])
        return self

    def transform(
        self: 'PipePolynomialFeatures',
        x: pd.DataFrame,
        y: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Transform."""
        new_cols = pd.DataFrame(
            data=self.poly.transform(x[self.columns]),
            index=x.index,
            columns=self.poly.get_feature_names_out(),
        )
        df_drop_old = x.drop(columns=self.columns)
        df = df_drop_old.join(new_cols)
        self.feature_names_in_ = df.columns
        return df

    def get_feature_names_out(
        self: 'PipePolynomialFeatures',
        input_features: list | str = None,
    ) -> list:
        """Get output feature names for transformation."""
        return self.feature_names_in_


if __name__ == '__main__':
    test = 'PolynomialFeatures_'

    if test == 'SortInRow':
        df = pd.DataFrame(
            {
                'a': [0, 1, 2, 3],
                'b': [0, 2, 1, 3],
                'c': [0, 1, 2, 3],
                'd': [0, 3, 2, 3],
            },
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

    if test == 'PolynomialFeatures_':
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
        poly = PipePolynomialFeatures(
            columns=['b', 'c'],
            degree=2,
            include_bias=True,
        )
        print('after:')
        print(poly.fit_transform(df))

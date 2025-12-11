import polars as pl
import polars.selectors as cs
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from path import target
from catboost import CatBoostClassifier, Pool
import yaml

with open("configs/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

from typing import Callable
from datetime import datetime
import mlflow


av_target = "is_test"

TransformFunction = Callable[[pl.DataFrame, str], pl.DataFrame]


class Transform:
    def __init__(self, name: str, column: str, info: str, func: TransformFunction):
        self.name: str = name
        self.column: str = column
        self.info: str = info
        self.func: TransformFunction = func

    def __str__(self) -> str:
        return f"Transform(name={self.name}, column={self.column}, info={self.info})"

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.func(df, self.column)


class DataTransformer:
    def __init__(self):
        self.transform_list: list[Transform] = []

    def add_transform(self, transform: Transform) -> None:
        self.transform_list.append(transform)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        for transform in self.transform_list:
            df = transform(df)
        return df

    def __str__(self) -> str:
        transform_names = [str(transform) for transform in self.transform_list]
        return f"DataTransformer(transforms={transform_names})"


class AdversarialValidation:
    def __init__(self, train: pl.DataFrame, test: pl.DataFrame, model_config: dict):
        self.X_train: pd.DataFrame = train.select(cs.exclude(av_target)).to_pandas()
        self.y_train: pd.Series = train.select(pl.col(av_target)).to_pandas()[av_target]
        self.X_test: pd.DataFrame = test.select(cs.exclude(av_target)).to_pandas()
        self.y_test: pd.Series = test.select(pl.col(av_target)).to_pandas()[av_target]

        mlflow.set_experiment("Adversarial Validation")
        self.av_target = av_target
        self.model = self._create_model(model_config)

    # TODO: Split into smaller functions
    def run(self, drop_list: list[str], cat_features: list[str], transform: DataTransformer | None = None) -> tuple[float, pd.DataFrame]: # fmt: skip
        """Runs adversarial validation and returns AUC and feature importance dataframe"""
        train_pool, test_pool = self._create_pools(drop_list, cat_features, transform)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with mlflow.start_run():
            self.model.fit(train_pool, eval_set=test_pool, use_best_model=True)

            auc = self.model.get_best_score()["validation"]["AUC"]
            ft_df = pd.DataFrame(
                {
                    "Feature": self.model.feature_names_,
                    "Importance": self.model.get_feature_importance(),
                }
            ).sort_values(by="Importance", ascending=False)
            print(f"Adversarial Validation AUC: {auc:.4f}")

            ft_df_path = f"logs/av_feat_importances_{timestamp}.csv"
            ft_df.to_csv(ft_df_path, index=False)
            plot_path = self._plot_importances(ft_df, timestamp)

            mlflow.log_param("cat_features", cat_features)
            mlflow.log_param("drop_list", drop_list)
            mlflow.log_params(model_config["catboost"])
            mlflow.log_metric("adversarial_validation_auc", float(auc))
            mlflow.log_artifact(ft_df_path)
            mlflow.log_artifact(plot_path)

            if transform is not None:
                mlflow.log_param("transforms", str(transform))

        return auc, ft_df

    def _create_model(self, model_config: dict) -> CatBoostClassifier:
        """Creates and returns a CatBoostClassifier model based on the provided configuration"""
        return CatBoostClassifier(
            iterations=model_config["catboost"]["iterations"],
            learning_rate=model_config["catboost"]["learning_rate"],
            depth=model_config["catboost"]["depth"],
            random_seed=model_config["catboost"]["random_seed"],
            eval_metric="AUC",
            logging_level="Silent",
        )

    def _create_pools(self, drop_list: list[str], cat_features: list[str], transform: DataTransformer | None = None) -> tuple[Pool, Pool]: # fmt: skip
        """Creates and returns CatBoost Pool objects for training and testing data"""
        if transform is not None:
            self.X_train = transform.transform(pl.from_pandas(self.X_train)).to_pandas()
            self.X_test = transform.transform(pl.from_pandas(self.X_test)).to_pandas()

        # Drop specified columns if they exist in the DataFrames
        cols_to_drop_train = [c for c in drop_list if c in self.X_train.columns]
        cols_to_drop_test = [c for c in drop_list if c in self.X_test.columns]

        X_train = self.X_train.drop(columns=cols_to_drop_train).copy()
        X_test = self.X_test.drop(columns=cols_to_drop_test).copy()

        # Filter cat_features to only those that exist in the POST-DROP DataFrame
        cat_features_filtered = [c for c in cat_features if c in X_train.columns]

        # Fill missing in categorical features and cast to string
        for c in cat_features_filtered:
            X_train[c] = X_train[c].fillna("<UNK>").astype(str)
            if c in X_test.columns:
                X_test[c] = X_test[c].fillna("<UNK>").astype(str)

        # Fill missing in numeric columns using train median
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        for c in num_cols:
            median_val = X_train[c].median()
            if pd.isna(median_val):
                median_val = 0  # fallback if entire column is NaN
            X_train[c] = X_train[c].fillna(median_val)
            if c in X_test.columns:
                X_test[c] = X_test[c].fillna(median_val)

        train_pool = Pool(
            data=X_train,
            label=self.y_train,
            cat_features=cat_features_filtered,
        )
        test_pool = Pool(
            data=X_test,
            label=self.y_test,
            cat_features=cat_features_filtered,
        )
        return train_pool, test_pool

    def _plot_importances(self, ft_df: pd.DataFrame, timestamp: str) -> str:
        """Plots top 20 feature importances"""
        save_path = f"logs/av_feat_importances_{timestamp}.png"
        plt.figure(figsize=(12, 8))
        sns.barplot(data=ft_df.head(20), x="Importance", y="Feature")
        plt.title("Top 20 Feature Importances in Adversarial Validation")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        return save_path


# TODO: Clean up this function
def create_adversarial_data(
    df_train: pl.DataFrame, df_test: pl.DataFrame, train_ratio: float = 0.8
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df_train = df_train.select(pl.all().exclude(target))
    df_train = df_train.select(pl.all().name.map(lambda n: n.replace("-", "_")))
    df_test = df_test.select(pl.all().name.map(lambda n: n.replace("-", "_")))

    with pl.StringCache():
        df_train = df_train.with_columns(pl.lit(0).alias(av_target))
        df_test = df_test.with_columns(pl.lit(1).alias(av_target))
        df_combined = pl.concat([df_train, df_test], how="vertical")

    cat_cols = df_combined.select(cs.string()).columns

    df_combined = df_combined.with_columns(
        [
            pl.col(cat_cols).fill_null("<UNK>"),
        ]
    )

    shuffled = df_combined.sample(fraction=1.0, seed=42, shuffle=True)
    train_ids = int(shuffled.height * train_ratio)
    train_shuffled = shuffled.slice(0, train_ids)
    test_shuffled = shuffled.slice(train_ids, None)

    assert train_shuffled.height + test_shuffled.height == df_combined.height
    return train_shuffled, test_shuffled, df_combined


# TODO: Refactor and fix plot_importances
def plot_importances(df: pl.DataFrame, considering_features: list[str]) -> None:
    for feat in considering_features:
        grouped_df = df.group_by(av_target).agg(pl.col(feat).unique())
        test = (
            grouped_df.filter(pl.col(av_target) == 0)
            .select(pl.col(feat).unique())
            .item()
            .to_list()
        )
        group_0_list = grouped_df.filter(pl.col(av_target) == 0)[feat].to_list()[0]
        group_1_list = grouped_df.filter(pl.col(av_target) == 1)[feat].to_list()[0]
        unique_to_0 = list(set(group_1_list) - set(group_0_list))
        unique_to_1 = list(set(group_0_list) - set(group_1_list))
        print(
            f"Feature: {feat:>5}, len(group_0_list): {len(group_0_list):>5}, len(group_1_list): {len(group_1_list):>5}, len(unique_to_0): {len(unique_to_0)}"
        )
        try:
            group_0_list.sort()
            group_1_list.sort()
            unique_to_0.sort()
        except TypeError:
            # Sorting failed, likely due to incompatible types in the list.
            pass
        print(f"\nFeature: {feat}")

        for target_val, values_list in grouped_df.iter_rows():
            print(f"Target {target_val} len({len(values_list)}): {values_list[:50]}")
        print(
            f"Features that only test have (len={len(unique_to_0)}) {unique_to_0[:50]}"
        )
        print(
            f"Features that only train have (len={len(unique_to_1)}) {unique_to_1[:50]}"
        )
        print("-" * 80)

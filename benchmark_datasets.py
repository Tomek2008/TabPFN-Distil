"""Curated suite of small, non-linear OpenML datasets for evaluating TabPFN and its distillations.

The suite contains 20 datasets (10 classification + 10 regression), every one with <= 1000 rows and
known non-linear structure (XOR / multiplicative interactions, multi-class geometry, signal / physics
data). This is exactly TabPFN's regime, and it is where a strong teacher / student pulls clearly ahead
of a linear baseline -- so distillation gains are easy to see.

Everything is fetched through ``sklearn.datasets.fetch_openml(data_id=...)`` (no extra dependency, with
built-in on-disk caching). ``OpenMLBenchmark`` loads any dataset into model-ready ``np.float32`` arrays
and runs a repeated train/test-split evaluation, reusing the project's conventions
(``StratifiedShuffleSplit`` / ``ShuffleSplit`` with ``random_state=0``, ``StandardScaler`` for students,
``accuracy_score`` for classification and ``r2_score`` / RMSE for regression).

Example
-------
>>> from benchmark_datasets import OpenMLBenchmark
>>> from tabpfn import TabPFNClassifier, TabPFNRegressor
>>> bench = OpenMLBenchmark()
>>> bench.list()                                          # registry as a DataFrame
>>> ds = bench.load("sonar")                              # -> LoadedDataset(X, y, ...)
>>> bench.evaluate(                                       # teacher across all classification sets
...     lambda task: TabPFNClassifier(), task="classification")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

Task = Literal["classification", "regression"]

RANDOM_STATE = 0


@dataclass(frozen=True)
class DatasetSpec:
    """A single OpenML dataset in the benchmark registry."""

    name: str
    data_id: int
    task: Task
    n_rows: int
    note: str = ""


# 10 small, non-linear classification datasets (all <= 1000 rows). data_id / row counts verified
# against OpenML via fetch_openml.
CLASSIFICATION_DATASETS: list[DatasetSpec] = [
    DatasetSpec("tic-tac-toe", 50, "classification", 958, "XOR-like win patterns, pure interaction"),
    DatasetSpec("monks-problems-2", 334, "classification", 601, "synthetic XOR / parity target"),
    DatasetSpec("sonar", 40, "classification", 208, "60 correlated sonar bands, non-linear boundary"),
    DatasetSpec("ionosphere", 59, "classification", 351, "radar signal, non-linear"),
    DatasetSpec("vehicle", 54, "classification", 846, "4-class silhouette geometry"),
    DatasetSpec("wdbc", 1510, "classification", 569, "breast cancer, non-linear feature interactions"),
    DatasetSpec("diabetes", 37, "classification", 768, "Pima diabetes, classic non-linear medical"),
    DatasetSpec("ilpd", 1480, "classification", 583, "Indian liver patient, non-linear medical"),
    DatasetSpec("balance-scale", 11, "classification", 625, "target is a product (distance x weight)"),
    DatasetSpec("blood-transfusion", 1464, "classification", 748, "non-linear recency / frequency"),
]

# 10 small, non-linear regression datasets (all <= 1000 rows).
REGRESSION_DATASETS: list[DatasetSpec] = [
    DatasetSpec("autoMpg", 196, "regression", 398, "non-linear mpg vs weight / horsepower"),
    DatasetSpec("machine_cpu", 230, "regression", 209, "non-linear CPU performance"),
    DatasetSpec("boston", 531, "regression", 506, "classic non-linear housing (known ethical caveat)"),
    DatasetSpec("bodyfat", 560, "regression", 252, "non-linear body measurements"),
    DatasetSpec("no2", 547, "regression", 500, "air-quality, non-linear"),
    DatasetSpec("pm10", 522, "regression", 500, "air-quality, non-linear"),
    DatasetSpec("sensory", 546, "regression", 576, "wine sensory scores"),
    DatasetSpec("cloud", 210, "regression", 108, "small non-linear"),
    DatasetSpec("autoPrice", 207, "regression", 159, "non-linear car pricing"),
    DatasetSpec("stock", 223, "regression", 950, "non-linear financial"),
]

BENCHMARK_DATASETS: list[DatasetSpec] = CLASSIFICATION_DATASETS + REGRESSION_DATASETS


@dataclass
class LoadedDataset:
    """A dataset loaded into model-ready arrays."""

    X: np.ndarray
    y: np.ndarray
    task: Task
    name: str
    data_id: int
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int | None:
        return int(len(np.unique(self.y))) if self.task == "classification" else None


class OpenMLBenchmark:
    """Load and evaluate the curated suite of small, non-linear OpenML datasets."""

    def __init__(self, task: Task | None = None, cache_dir: str | None = None) -> None:
        """Args:
        task: optionally restrict the suite to ``"classification"`` or ``"regression"``.
        cache_dir: ``data_home`` passed to ``fetch_openml`` (defaults to scikit-learn's cache).
        """
        if task is not None and task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification', 'regression' or None, got {task!r}")
        self.task = task
        self.cache_dir = cache_dir
        self.specs: list[DatasetSpec] = [
            s for s in BENCHMARK_DATASETS if task is None or s.task == task
        ]
        self._by_key: dict[str | int, DatasetSpec] = {}
        for s in self.specs:
            self._by_key[s.name] = s
            self._by_key[s.data_id] = s

    def list(self) -> pd.DataFrame:
        """Return the registry as a DataFrame (name, data_id, task, n_rows, note)."""
        return pd.DataFrame(
            [(s.name, s.data_id, s.task, s.n_rows, s.note) for s in self.specs],
            columns=["name", "data_id", "task", "n_rows", "note"],
        )

    def _spec(self, name_or_id: str | int | DatasetSpec) -> DatasetSpec:
        if isinstance(name_or_id, DatasetSpec):
            return name_or_id
        try:
            return self._by_key[name_or_id]
        except KeyError as exc:
            raise KeyError(
                f"{name_or_id!r} is not in this suite. Available: "
                f"{[s.name for s in self.specs]}"
            ) from exc

    def load(self, name_or_id: str | int | DatasetSpec) -> LoadedDataset:
        """Load a dataset by name or data_id into model-ready ``np.float32`` arrays.

        Categorical features are one-hot encoded and missing values imputed; the classification target
        is label-encoded to integers and the regression target cast to float.
        """
        spec = self._spec(name_or_id)
        bunch = fetch_openml(
            data_id=spec.data_id, as_frame=True, parser="auto", data_home=self.cache_dir
        )
        X_df: pd.DataFrame = bunch.data
        y_raw: pd.Series = bunch.target

        categorical = [c for c in X_df.columns if str(X_df[c].dtype) in ("category", "object", "bool")]
        numeric = [c for c in X_df.columns if c not in categorical]

        transformers = []
        if numeric:
            transformers.append(
                ("num", SimpleImputer(strategy="median"), numeric)
            )
        if categorical:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                )
            )
        pre = ColumnTransformer(transformers, remainder="drop")
        X = pre.fit_transform(X_df).astype(np.float32)
        feature_names = list(pre.get_feature_names_out())

        if spec.task == "classification":
            y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(np.int64)
        else:
            y = np.asarray(y_raw, dtype=np.float32)

        return LoadedDataset(
            X=X,
            y=y,
            task=spec.task,
            name=spec.name,
            data_id=spec.data_id,
            feature_names=feature_names,
        )

    def splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: Task,
        n_splits: int = 5,
        train_size: float | int = 0.6,
    ):
        """Yield ``(train_idx, test_idx)`` pairs.

        Stratified for classification (``StratifiedShuffleSplit``) and plain ``ShuffleSplit`` for
        regression, matching the project's existing convention with ``random_state=0``.
        """
        splitter_cls = StratifiedShuffleSplit if task == "classification" else ShuffleSplit
        splitter = splitter_cls(
            n_splits=n_splits, train_size=train_size, random_state=RANDOM_STATE
        )
        yield from splitter.split(X, y)

    def evaluate(
        self,
        estimator_factory: Callable[[Task], object],
        datasets: list[str] | None = None,
        task: Task | None = None,
        n_splits: int = 5,
        train_size: float | int = 0.6,
        scale: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Evaluate an estimator across datasets with repeated splits.

        Args:
            estimator_factory: callable ``task -> fresh estimator`` (e.g. ``lambda t: TabPFNClassifier()``
                for classification, or returning a student model). A fresh estimator is built per split.
            datasets: subset of dataset names to run (defaults to the whole suite / current task filter).
            task: optionally restrict to one task (in addition to any filter set in ``__init__``).
            n_splits: number of repeated shuffle splits.
            train_size: fraction or absolute number of training rows (rest is the test set).
            scale: standardize features with ``StandardScaler`` (helps student models; harmless to TabPFN).
            verbose: print per-dataset results as they complete.

        Returns:
            DataFrame with one row per dataset. Classification reports ``acc_mean``/``acc_std``;
            regression reports ``r2_mean``/``r2_std`` and ``rmse_mean``/``rmse_std``.
        """
        specs = self.specs
        if task is not None:
            specs = [s for s in specs if s.task == task]
        if datasets is not None:
            wanted = set(datasets)
            specs = [s for s in specs if s.name in wanted]
        if not specs:
            raise ValueError("No datasets selected. Check the `datasets`/`task` filters.")

        rows = []
        for spec in specs:
            ds = self.load(spec)
            primary, secondary = [], []  # accuracy, or (r2, rmse)
            for train_idx, test_idx in self.splits(
                ds.X, ds.y, ds.task, n_splits=n_splits, train_size=train_size
            ):
                X_tr, X_te = ds.X[train_idx], ds.X[test_idx]
                y_tr, y_te = ds.y[train_idx], ds.y[test_idx]
                if scale:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
                    X_te = scaler.transform(X_te).astype(np.float32)

                model = estimator_factory(ds.task)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)

                if ds.task == "classification":
                    primary.append(accuracy_score(y_te, pred))
                else:
                    primary.append(r2_score(y_te, pred))
                    secondary.append(np.sqrt(mean_squared_error(y_te, pred)))

            if spec.task == "classification":
                row = {
                    "name": spec.name,
                    "task": spec.task,
                    "n_rows": spec.n_rows,
                    "acc_mean": float(np.mean(primary)),
                    "acc_std": float(np.std(primary)),
                }
            else:
                row = {
                    "name": spec.name,
                    "task": spec.task,
                    "n_rows": spec.n_rows,
                    "r2_mean": float(np.mean(primary)),
                    "r2_std": float(np.std(primary)),
                    "rmse_mean": float(np.mean(secondary)),
                    "rmse_std": float(np.std(secondary)),
                }
            rows.append(row)
            if verbose:
                metric = (
                    f"acc={row['acc_mean']:.3f}+/-{row['acc_std']:.3f}"
                    if spec.task == "classification"
                    else f"r2={row['r2_mean']:.3f}+/-{row['r2_std']:.3f}"
                )
                print(f"[{spec.task:14s}] {spec.name:20s} {metric}")

        return pd.DataFrame(rows)


def main() -> None:
    """Light demo: print the registry and load one classification dataset."""
    bench = OpenMLBenchmark()
    print("Benchmark suite (20 datasets):")
    print(bench.list().to_string(index=False))

    ds = bench.load(CLASSIFICATION_DATASETS[0].name)
    print(f"\nLoaded {ds.name!r}: X={ds.X.shape} ({ds.X.dtype}), y={ds.y.shape}")
    print(f"classes: {ds.n_classes}, feature count: {len(ds.feature_names)}")


if __name__ == "__main__":
    main()

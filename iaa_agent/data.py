from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .models import CheckIn, DatasetCapabilities
from .utils import haversine_km, hour_bucket, time_bucket_label


REQUIRED_COLUMNS = {
    "user_id",
    "POI_id",
    "POI_catid",
    "POI_catid_code",
    "POI_catname",
    "latitude",
    "longitude",
    "timezone",
    "UTC_time",
    "local_time",
    "day_of_week",
    "norm_in_day_time",
    "trajectory_id",
}


@dataclass(frozen=True)
class QueryExample:
    traj_id: str
    context: pd.DataFrame
    target: pd.Series


class NYCDataRepository:
    def __init__(self, data_dir: str | Path = "datasets/NYC") -> None:
        self.data_dir = Path(data_dir)
        self.train = self._load_split("train")
        self.val = self._load_split("val")
        self.test = self._load_split("test")
        self.history = pd.concat([self.train, self.val], ignore_index=True)
        self._catalog = self._build_catalog(self.history)
        self._all_meta = self._build_catalog(pd.concat([self.history, self.test], ignore_index=True))
        self._history_by_user = {str(k): v.copy() for k, v in self.history.groupby("user_id", sort=False)}
        self._test_groups = {str(k): v.copy() for k, v in self.test.groupby("trajectory_id", sort=False)}
        self._global_category_transitions: dict[tuple[str, str], int] | None = None
        self._global_poi_transitions: dict[tuple[str, str], int] | None = None
        self._peer_vectors: dict[str, dict[str, float]] | None = None
        self._peer_cells: dict[str, set[str]] | None = None

    @property
    def capabilities(self) -> DatasetCapabilities:
        return DatasetCapabilities(
            notes=[
                "Foursquare NYC split has category, coordinates, timestamps, and trajectory_id.",
                "Reviews, images, opening hours, price, and ratings are unavailable in v0.",
            ]
        )

    @property
    def catalog(self) -> pd.DataFrame:
        return self._catalog.copy()

    @property
    def all_meta(self) -> pd.DataFrame:
        return self._all_meta.copy()

    def summary(self) -> dict:
        out: dict[str, dict] = {}
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            lengths = df.groupby("trajectory_id").size()
            out[name] = {
                "rows": int(len(df)),
                "users": int(df["user_id"].nunique()),
                "pois": int(df["POI_id"].nunique()),
                "categories": int(df["POI_catname"].nunique()),
                "trajectories": int(df["trajectory_id"].nunique()),
                "median_trajectory_length": float(lengths.median()),
                "min_utc_time": str(df["UTC_time"].min()),
                "max_utc_time": str(df["UTC_time"].max()),
            }
        out["dataset_capabilities"] = self.capabilities.model_dump()
        return out

    def iter_test_traj_ids(self) -> list[str]:
        return sorted(self._test_groups.keys(), key=_trajectory_sort_key)

    def get_query(self, traj_id: str) -> QueryExample:
        key = str(traj_id)
        if key not in self._test_groups:
            raise KeyError(f"Unknown test trajectory id: {traj_id}")
        group = self._test_groups[key].sort_values("UTC_time").reset_index(drop=True)
        if len(group) < 2:
            raise ValueError(f"Trajectory {traj_id} has fewer than 2 check-ins")
        return QueryExample(traj_id=key, context=group.iloc[:-1].copy(), target=group.iloc[-1].copy())

    def history_for_user(self, user_id: str | int, context: pd.DataFrame | None = None) -> pd.DataFrame:
        base = self._history_by_user.get(str(user_id), pd.DataFrame(columns=self.history.columns)).copy()
        if context is not None and len(context):
            visible = context.copy()
            return pd.concat([base, visible], ignore_index=True).sort_values("UTC_time").reset_index(drop=True)
        return base.sort_values("UTC_time").reset_index(drop=True)

    def runtime_catalog(self, context: pd.DataFrame | None = None) -> pd.DataFrame:
        if context is None or context.empty:
            return self.catalog
        visible = self._build_catalog(context)
        merged = pd.concat([self._catalog, visible], ignore_index=True)
        merged = merged.sort_values("visit_count", ascending=False).drop_duplicates("POI_id", keep="first")
        return merged.reset_index(drop=True)

    def poi_meta(self, poi_id: str, context: pd.DataFrame | None = None) -> dict:
        catalog = self.runtime_catalog(context)
        match = catalog[catalog["POI_id"] == poi_id]
        if match.empty:
            match = self._all_meta[self._all_meta["POI_id"] == poi_id]
        if match.empty:
            return {
                "POI_id": poi_id,
                "display_name": poi_id,
                "category": "Unknown",
                "latitude": 0.0,
                "longitude": 0.0,
                "visit_count": 0,
            }
        row = match.iloc[0]
        return {
            "POI_id": str(row["POI_id"]),
            "display_name": str(row["POI_id"]),
            "category": str(row["category"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "visit_count": int(row.get("visit_count", 0)),
        }

    def nearest_pois(
        self,
        latitude: float,
        longitude: float,
        limit: int = 50,
        context: pd.DataFrame | None = None,
        exclude: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        catalog = self.runtime_catalog(context)
        excluded = set(exclude or [])
        if excluded:
            catalog = catalog[~catalog["POI_id"].isin(excluded)].copy()
        if catalog.empty:
            return catalog.assign(distance_km=[])
        lat = catalog["latitude"].to_numpy(dtype=float)
        lon = catalog["longitude"].to_numpy(dtype=float)
        distances = np.array([haversine_km(latitude, longitude, la, lo) for la, lo in zip(lat, lon)])
        out = catalog.copy()
        out["distance_km"] = distances
        return out.sort_values("distance_km").head(limit).reset_index(drop=True)

    def rows_near_target_time(self, target_time: pd.Timestamp, minutes: int = 30) -> pd.DataFrame:
        target_minute = int(target_time.hour) * 60 + int(target_time.minute)
        minutes_of_day = self.history["hour"] * 60 + self.history["minute"]
        diff = (minutes_of_day - target_minute).abs()
        circ = np.minimum(diff, 1440 - diff)
        return self.history[circ <= minutes].copy()

    def global_category_transitions(self) -> dict[tuple[str, str], int]:
        if self._global_category_transitions is None:
            self._global_category_transitions = self._transition_counts("POI_catname")
        return self._global_category_transitions

    def global_poi_transitions(self) -> dict[tuple[str, str], int]:
        if self._global_poi_transitions is None:
            self._global_poi_transitions = self._transition_counts("POI_id")
        return self._global_poi_transitions

    def global_category_counts_at(self, target_hour_bucket: int, target_day: int) -> pd.DataFrame:
        df = self.history
        grouped = (
            df.assign(
                same_hour=(df["hour_bucket"] == target_hour_bucket).astype(int),
                same_day=(df["day_of_week"] == target_day).astype(int),
            )
            .groupby("POI_catname")
            .agg(total=("POI_catname", "size"), same_hour=("same_hour", "sum"), same_day=("same_day", "sum"))
            .reset_index()
        )
        grouped["score"] = grouped["same_hour"] * 2.0 + grouped["same_day"] + grouped["total"] * 0.05
        return grouped.sort_values("score", ascending=False)

    def user_peer_inputs(self) -> tuple[dict[str, dict[str, float]], dict[str, set[str]]]:
        if self._peer_vectors is not None and self._peer_cells is not None:
            return self._peer_vectors, self._peer_cells
        vectors: dict[str, dict[str, float]] = {}
        cells: dict[str, set[str]] = {}
        for user_id, rows in self.history.groupby("user_id", sort=False):
            vec: dict[str, float] = {}
            cellset: set[str] = set()
            for row in rows.itertuples(index=False):
                key = f"{row.POI_catname}|{row.hour_bucket}"
                vec[key] = vec.get(key, 0.0) + 1.0
                cellset.add(f"{round(float(row.latitude), 2)}:{round(float(row.longitude), 2)}")
            vectors[str(user_id)] = vec
            cells[str(user_id)] = cellset
        self._peer_vectors = vectors
        self._peer_cells = cells
        return vectors, cells

    def to_checkins(self, df: pd.DataFrame) -> list[CheckIn]:
        return [self._row_to_checkin(row) for _, row in df.iterrows()]

    def _load_split(self, split: str) -> pd.DataFrame:
        path = self.data_dir / f"NYC_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        df = df.copy()
        df["user_id"] = df["user_id"].astype(str)
        df["POI_id"] = df["POI_id"].astype(str)
        df["POI_catname"] = df["POI_catname"].fillna("Unknown").astype(str)
        df["trajectory_id"] = df["trajectory_id"].astype(str)
        df["UTC_time"] = pd.to_datetime(df["UTC_time"], utc=True)
        df["local_time"] = pd.to_datetime(df["local_time"], utc=True)
        df["hour"] = df["local_time"].dt.hour.astype(int)
        df["minute"] = df["local_time"].dt.minute.astype(int)
        df["hour_bucket"] = df["hour"].map(hour_bucket).astype(int)
        df["time_of_day_bucket"] = df["hour"].map(time_bucket_label)
        df["split"] = split
        return df.sort_values("UTC_time").reset_index(drop=True)

    def _build_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["POI_id", "category", "latitude", "longitude", "visit_count"])
        grouped = (
            df.groupby("POI_id")
            .agg(
                category=("POI_catname", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
                latitude=("latitude", "median"),
                longitude=("longitude", "median"),
                visit_count=("POI_id", "size"),
            )
            .reset_index()
        )
        grouped["display_name"] = grouped["POI_id"]
        return grouped

    def _transition_counts(self, column: str) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for _, group in self.history.groupby("trajectory_id", sort=False):
            ordered = group.sort_values("UTC_time")
            vals = [str(v) for v in ordered[column].tolist()]
            for a, b in zip(vals, vals[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    def _row_to_checkin(self, row: pd.Series) -> CheckIn:
        return CheckIn(
            user_id=str(row["user_id"]),
            poi_id=str(row["POI_id"]),
            category=str(row["POI_catname"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            utc_time=pd.Timestamp(row["UTC_time"]).isoformat(),
            local_time=pd.Timestamp(row["local_time"]).isoformat(),
            day_of_week=int(row["day_of_week"]),
            hour=int(row["hour"]),
            hour_bucket=int(row["hour_bucket"]),
            trajectory_id=str(row["trajectory_id"]),
        )


def _trajectory_sort_key(value: str) -> tuple[int, str]:
    try:
        return (int(value.split("_")[-1]), value)
    except Exception:
        return (10**9, value)


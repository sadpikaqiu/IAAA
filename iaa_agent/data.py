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
    mode: str = "trajectory"
    history: pd.DataFrame | None = None
    target_index: int | None = None


class NYCDataRepository:
    def __init__(self, data_dir: str | Path = "datasets/NYC") -> None:
        self.data_dir = Path(data_dir)
        self.train = self._load_split("train")
        self.val = self._load_split("val")
        self.test = self._load_split("test")
        self._build_poi_index()
        self.train = self._attach_poi_idx(self.train)
        self.val = self._attach_poi_idx(self.val)
        self.test = self._attach_poi_idx(self.test)
        self.all_events = pd.concat([self.train, self.val, self.test], ignore_index=True).sort_values("UTC_time").reset_index(drop=True)
        self.history = pd.concat([self.train, self.val], ignore_index=True)
        self._catalog = self._build_catalog(self.history)
        self._all_meta = self._build_catalog(pd.concat([self.history, self.test], ignore_index=True))
        self._history_by_user = {str(k): v.copy() for k, v in self.history.groupby("user_id", sort=False)}
        self._test_groups = {str(k): v.copy() for k, v in self.test.groupby("trajectory_id", sort=False)}
        self._global_category_transitions: dict[tuple[str, str], int] | None = None
        self._global_poi_transitions: dict[tuple[str, str], int] | None = None
        self._peer_vectors: dict[str, dict[str, float]] | None = None
        self._peer_cells: dict[str, set[str]] | None = None
        self._active_history_mode = "global_train_val"

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
        out["poi_id_mapping"] = {
            "count": len(self._poi_id_to_idx),
            "format": "P000001-style stable IDs sorted by original Foursquare POI_id",
        }
        out["active_history_mode"] = self._active_history_mode
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
        return QueryExample(traj_id=key, context=group.iloc[:-1].copy(), target=group.iloc[-1].copy(), mode="trajectory")

    def use_user_chronological_split(self, train_ratio: float = 0.8) -> None:
        """Use each user's first train_ratio events as the active global history."""
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        parts = []
        for _, rows in self.all_events.groupby("user_id", sort=False):
            ordered = rows.sort_values("UTC_time").reset_index(drop=True)
            cutoff = max(1, int(len(ordered) * train_ratio))
            if len(ordered) >= 2:
                cutoff = min(cutoff, len(ordered) - 1)
            parts.append(ordered.iloc[:cutoff].copy())
        self.history = pd.concat(parts, ignore_index=True).sort_values("UTC_time").reset_index(drop=True) if parts else self.history.iloc[:0].copy()
        self._catalog = self._build_catalog(self.history)
        self._history_by_user = {str(k): v.copy() for k, v in self.history.groupby("user_id", sort=False)}
        self._reset_history_caches()
        self._active_history_mode = f"user_chronological_{train_ratio:.3f}"

    def iter_user_test_events(self, train_ratio: float = 0.8, min_context: int = 1) -> list[tuple[str, int]]:
        keys: list[tuple[str, int]] = []
        for user_id, rows in self.all_events.groupby("user_id", sort=False):
            ordered = rows.sort_values("UTC_time").reset_index(drop=True)
            if len(ordered) <= min_context:
                continue
            cutoff = max(min_context, int(len(ordered) * train_ratio))
            cutoff = min(cutoff, len(ordered) - 1)
            for idx in range(cutoff, len(ordered)):
                if idx >= min_context:
                    keys.append((str(user_id), int(idx)))
        keys.sort(key=lambda x: (x[0], x[1]))
        return keys

    def iter_session_test_keys(
        self,
        train_ratio: float = 0.8,
        min_context: int = 1,
        user_id: str | int | None = None,
    ) -> list[tuple[str, str]]:
        keys: list[tuple[pd.Timestamp, str, str]] = []
        if user_id is None:
            user_groups = self.all_events.groupby("user_id", sort=False)
        else:
            user_key = str(user_id)
            rows = self.all_events[self.all_events["user_id"] == user_key]
            if rows.empty:
                raise KeyError(f"Unknown user id: {user_id}")
            user_groups = [(user_key, rows)]
        for current_user_id, rows in user_groups:
            ordered = rows.sort_values("UTC_time").reset_index(drop=True)
            if len(ordered) <= min_context:
                continue
            cutoff = self._user_cutoff_index(ordered, train_ratio)
            for trajectory_id, session in ordered.groupby("trajectory_id", sort=False):
                session = session.sort_values("UTC_time")
                if len(session) <= min_context:
                    continue
                target = session.iloc[-1]
                target_index = int(session.index[-1])
                if target_index < cutoff:
                    continue
                keys.append((pd.Timestamp(target["UTC_time"]), str(current_user_id), str(trajectory_id)))
        keys.sort(key=lambda x: (x[0], x[1], _trajectory_sort_key(x[2])))
        return [(user_id, trajectory_id) for _, user_id, trajectory_id in keys]

    def get_session_query(
        self,
        user_id: str | int,
        trajectory_id: str | int,
        train_ratio: float = 0.8,
        min_context: int = 1,
    ) -> QueryExample:
        user_key = str(user_id)
        trajectory_key = str(trajectory_id)
        rows = self.all_events[self.all_events["user_id"] == user_key].sort_values("UTC_time").reset_index(drop=True)
        if rows.empty:
            raise KeyError(f"Unknown user id: {user_id}")
        cutoff = self._user_cutoff_index(rows, train_ratio)
        session = rows[rows["trajectory_id"] == trajectory_key].sort_values("UTC_time")
        if session.empty:
            raise KeyError(f"Unknown trajectory id {trajectory_id} for user {user_id}")
        if len(session) <= min_context:
            raise ValueError(f"Trajectory {trajectory_id} has fewer than {min_context + 1} check-ins")
        target_index = int(session.index[-1])
        if target_index < cutoff:
            raise ValueError(f"Trajectory {trajectory_id} target is before the held-out split cutoff {cutoff}")
        history = rows.iloc[:cutoff].copy()
        context = session.iloc[:-1].copy().reset_index(drop=True)
        target = session.iloc[-1].copy()
        return QueryExample(
            traj_id=f"session_{trajectory_key}",
            context=context,
            target=target,
            mode="session_split",
            history=history,
            target_index=target_index,
        )

    def get_user_query(
        self,
        user_id: str | int,
        target_index: int,
        train_ratio: float = 0.8,
        context_size: int = 5,
        require_test_index: bool = True,
    ) -> QueryExample:
        user_key = str(user_id)
        rows = self.all_events[self.all_events["user_id"] == user_key].sort_values("UTC_time").reset_index(drop=True)
        if rows.empty:
            raise KeyError(f"Unknown user id: {user_id}")
        if target_index < 1 or target_index >= len(rows):
            raise ValueError(f"target_index must be in [1, {len(rows) - 1}] for user {user_id}")
        cutoff = max(1, int(len(rows) * train_ratio))
        cutoff = min(cutoff, len(rows) - 1)
        if require_test_index and target_index < cutoff:
            raise ValueError(
                f"target_index {target_index} is before the user split cutoff {cutoff}; "
                "choose an index in the held-out tail or set require_test_index=False"
            )
        start = max(0, target_index - context_size)
        context = rows.iloc[start:target_index].copy()
        history = rows.iloc[:cutoff].copy()
        query_id = f"user_{user_key}_idx_{target_index}"
        return QueryExample(
            traj_id=query_id,
            context=context,
            target=rows.iloc[target_index].copy(),
            mode="user_timeline",
            history=history,
            target_index=target_index,
        )

    def user_timeline_info(self, user_id: str | int, train_ratio: float = 0.8) -> dict:
        user_key = str(user_id)
        rows = self.all_events[self.all_events["user_id"] == user_key].sort_values("UTC_time").reset_index(drop=True)
        if rows.empty:
            raise KeyError(f"Unknown user id: {user_id}")
        cutoff = max(1, int(len(rows) * train_ratio))
        cutoff = min(cutoff, len(rows) - 1)
        first_test = rows.iloc[cutoff]
        last_test = rows.iloc[-1]
        return {
            "user_id": user_key,
            "num_checkins": int(len(rows)),
            "train_ratio": train_ratio,
            "train_cutoff_index": int(cutoff),
            "valid_target_index_start": int(cutoff),
            "valid_target_index_end": int(len(rows) - 1),
            "first_test_time": pd.Timestamp(first_test["local_time"]).isoformat(),
            "last_test_time": pd.Timestamp(last_test["local_time"]).isoformat(),
            "first_test_poi_idx": str(first_test["POI_idx"]),
            "last_test_poi_idx": str(last_test["POI_idx"]),
        }

    def history_for_user(
        self,
        user_id: str | int,
        context: pd.DataFrame | None = None,
        base_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        base = base_history.copy() if base_history is not None else self._history_by_user.get(str(user_id), pd.DataFrame(columns=self.history.columns)).copy()
        if context is not None and len(context):
            visible = context.copy()
            merged = pd.concat([base, visible], ignore_index=True)
            merged = merged.drop_duplicates(["user_id", "POI_id", "UTC_time", "trajectory_id"], keep="first")
            return merged.sort_values("UTC_time").reset_index(drop=True)
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
                "POI_idx": self.poi_idx(poi_id),
                "display_name": self.poi_idx(poi_id),
                "category": "Unknown",
                "latitude": 0.0,
                "longitude": 0.0,
                "visit_count": 0,
            }
        row = match.iloc[0]
        return {
            "POI_id": str(row["POI_id"]),
            "POI_idx": str(row.get("POI_idx", self.poi_idx(str(row["POI_id"])))),
            "display_name": str(row.get("POI_idx", self.poi_idx(str(row["POI_id"])))),
            "category": str(row["category"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "visit_count": int(row.get("visit_count", 0)),
        }

    def poi_idx(self, poi_id: str) -> str:
        return self._poi_id_to_idx.get(str(poi_id), "P000000")

    def original_poi_id(self, poi_idx: str) -> str | None:
        return self._poi_idx_to_id.get(str(poi_idx))

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
            return pd.DataFrame(columns=["POI_id", "POI_idx", "category", "latitude", "longitude", "visit_count"])
        grouped = (
            df.groupby("POI_id")
            .agg(
                POI_idx=("POI_idx", "first"),
                category=("POI_catname", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
                latitude=("latitude", "median"),
                longitude=("longitude", "median"),
                visit_count=("POI_id", "size"),
            )
            .reset_index()
        )
        grouped["display_name"] = grouped["POI_idx"]
        return grouped

    def _build_poi_index(self) -> None:
        all_rows = pd.concat([self.train, self.val, self.test], ignore_index=True)
        poi_ids = sorted(str(x) for x in all_rows["POI_id"].dropna().unique())
        self._poi_id_to_idx = {poi_id: f"P{i + 1:06d}" for i, poi_id in enumerate(poi_ids)}
        self._poi_idx_to_id = {idx: poi_id for poi_id, idx in self._poi_id_to_idx.items()}

    def _attach_poi_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["POI_idx"] = out["POI_id"].map(self._poi_id_to_idx).fillna("P000000")
        return out

    def _user_cutoff_index(self, rows: pd.DataFrame, train_ratio: float) -> int:
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        cutoff = max(1, int(len(rows) * train_ratio))
        if len(rows) >= 2:
            cutoff = min(cutoff, len(rows) - 1)
        return cutoff

    def _reset_history_caches(self) -> None:
        self._global_category_transitions = None
        self._global_poi_transitions = None
        self._peer_vectors = None
        self._peer_cells = None

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
            poi_idx=str(row.get("POI_idx", self.poi_idx(str(row["POI_id"])))),
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

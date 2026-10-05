#!/usr/bin/env python3
"""
SmartRecruit - Face Activity Synthetic Dataset Generator
=========================================================
Run:  python generate_dataset.py

Creates (next to this script):
    face_activity_dataset.csv
    train.csv / validation.csv / test.csv

IMPORTANT: This dataset is SYNTHETIC. High accuracy on it does NOT imply the
same accuracy on real webcam data. Validate on real sessions before trusting it.

Only features your OpenCV + MediaPipe app already produces are used:
    face_presence_percent, maximum_faces_detected, longest_face_absence_sec,
    multiple_face_duration_sec, looking_away_events,
    looking_center_percent, looking_left_percent, looking_right_percent
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 10_000
OUT_DIR = Path(__file__).resolve().parent

CLASS_PROPORTIONS = {
    "NORMAL": 0.25,
    "NO_FACE": 0.15,
    "MULTIPLE_FACES": 0.15,
    "LOOKING_LEFT": 0.15,
    "LOOKING_RIGHT": 0.15,
    "LOOKING_AWAY": 0.15,
}
LABELS = list(CLASS_PROPORTIONS)

FEATURES = [
    "face_presence_percent",
    "maximum_faces_detected",
    "longest_face_absence_sec",
    "multiple_face_duration_sec",
    "looking_away_events",
    "looking_center_percent",
    "looking_left_percent",
    "looking_right_percent",
]
TARGET = "activity_label"
COLUMNS = ["session_id"] + FEATURES + [TARGET]

GAZE_SUM_TOLERANCE = 0.011  # rounding tolerance (2-decimal values)


class DatasetValidationError(Exception):
    """Raised when generated data fails validation."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def make_gaze(left, right):
    """Return (center, left, right) rounded to 2 decimals and summing to 100."""
    left = float(np.clip(left, 0.0, 100.0))
    right = float(np.clip(right, 0.0, 100.0))
    total = left + right
    if total > 100.0:
        left, right = left * 100.0 / total, right * 100.0 / total
    left, right = round(left, 2), round(right, 2)
    center = round(100.0 - left - right, 2)
    if center < 0:  # rounding edge case
        center = 0.0
        right = round(100.0 - left, 2)
    return center + 0.0, left + 0.0, right + 0.0  # +0.0 turns -0.0 into 0.0


def normal_gaze(rng):
    """Mostly-center gaze with a little left/right (natural glancing)."""
    left = rng.beta(1.5, 12) * 100
    right = rng.beta(1.5, 12) * 100
    if rng.random() < 0.08:  # occasional notable glance to one side
        if rng.random() < 0.5:
            left = rng.uniform(15, 30)
        else:
            right = rng.uniform(15, 30)
    if left + right > 55:
        scale = 55 / (left + right)
        left, right = left * scale, right * scale
    return left, right


def small_absence(rng, duration):
    """Brief tracking dropouts. Returns (presence_percent, longest_absence_sec)."""
    r = rng.random()
    if r < 0.45:
        absent_total = 0.0
    elif r < 0.90:
        absent_total = rng.exponential(2.0)
    else:  # occasional longer dip (overlaps with weak NO_FACE sessions)
        absent_total = rng.uniform(5, 25)
    absent_total = min(absent_total, duration * 0.3)
    presence = 100.0 - absent_total / duration * 100.0
    longest = absent_total * rng.uniform(0.4, 1.0) if absent_total > 0 else 0.0
    return presence, longest


def single_face_or_blip(rng):
    """Mostly exactly 1 face; rarely a very brief 2-face blip."""
    if rng.random() < 0.02:
        return 2, rng.uniform(0.2, 2.0)
    return 1, 0.0


def row(presence, max_faces, longest, mfd, events, center, left, right):
    return {
        "face_presence_percent": round(float(np.clip(presence, 0, 100)), 2),
        "maximum_faces_detected": int(max_faces),
        "longest_face_absence_sec": round(max(float(longest), 0.0), 2),
        "multiple_face_duration_sec": round(max(float(mfd), 0.0), 2),
        "looking_away_events": int(max(events, 0)),
        "looking_center_percent": center,
        "looking_left_percent": left,
        "looking_right_percent": right,
    }


# --------------------------------------------------------------------------
# Class generators (each returns one feature dict)
# --------------------------------------------------------------------------
def gen_normal(rng):
    duration = rng.uniform(120, 1800)
    presence, longest = small_absence(rng, duration)
    max_faces, mfd = single_face_or_blip(rng)
    left, right = normal_gaze(rng)
    center, left, right = make_gaze(left, right)
    events = rng.poisson(0.8 + (left + right) / 12)
    return row(presence, max_faces, longest, mfd, events, center, left, right)


def gen_no_face(rng):
    duration = rng.uniform(120, 1800)

    # Sessions where no face was ever detected (presence = 0)
    if rng.random() < 0.12:
        center, left, right = 100.0, 0.0, 0.0  # gaze undefined -> neutral
        return row(0.0, 0, duration, 0.0, 0, center, left, right)

    # Partial absence; some are mild (overlap with NORMAL / looking classes)
    if rng.random() < 0.65:
        presence = rng.uniform(5, 65)
    else:
        presence = rng.uniform(65, 90)
    absent_total = duration * (1 - presence / 100)
    longest = float(np.clip(absent_total * rng.beta(2, 2.5), 0.5, absent_total))
    max_faces, mfd = single_face_or_blip(rng)

    if rng.random() < 0.7:
        left, right = normal_gaze(rng)
    else:  # gaze is noisy when tracking is unreliable
        l, r, _ = rng.dirichlet([1.2, 1.2, 1.2]) * 100
        left, right = l, r
    center, left, right = make_gaze(left, right)
    events = rng.poisson(1.2 + (left + right) / 25)
    return row(presence, max_faces, longest, mfd, events, center, left, right)


def gen_multiple_faces(rng):
    duration = rng.uniform(120, 1800)
    presence, longest = small_absence(rng, duration)
    if rng.random() < 0.15:
        presence = rng.uniform(60, 80)  # face lost more often
        absent_total = duration * (1 - presence / 100)
        longest = absent_total * rng.uniform(0.1, 0.6)
    max_faces = int(rng.choice([2, 3, 4], p=[0.78, 0.17, 0.05]))

    present_time = duration * presence / 100
    if rng.random() < 0.15:
        mfd = rng.uniform(0.5, 3.0)  # brief second face (overlaps NORMAL blips)
    else:
        mfd = present_time * rng.uniform(0.01, 0.35)
    mfd = float(np.clip(mfd, 0.5, max(present_time, 0.5)))

    left, right = normal_gaze(rng)
    center, left, right = make_gaze(left * 1.2, right * 1.2)
    events = rng.poisson(1.5 + (left + right) / 12)
    return row(presence, max_faces, longest, mfd, events, center, left, right)


def _gen_looking_side(rng, dominant):
    duration = rng.uniform(120, 1800)
    presence, longest = small_absence(rng, duration)
    max_faces, mfd = single_face_or_blip(rng)

    if rng.random() < 0.10:  # weak dominance
        main = rng.uniform(25, 38)
    else:
        main = float(np.clip(rng.normal(52, 15), 22, 92))
    other = rng.beta(1.4, 9) * 100
    if rng.random() < 0.08:  # looks both ways but one side still wins
        other = rng.uniform(15, 32)
    other = min(other, max(main - 3, 0), 100 - main)

    left, right = (main, other) if dominant == "left" else (other, main)
    center, left, right = make_gaze(left, right)
    events = rng.poisson(1.5 + main / 18)
    return row(presence, max_faces, longest, mfd, events, center, left, right)


def gen_looking_left(rng):
    return _gen_looking_side(rng, "left")


def gen_looking_right(rng):
    return _gen_looking_side(rng, "right")


def gen_looking_away(rng):
    duration = rng.uniform(120, 1800)
    presence, longest = small_absence(rng, duration)
    max_faces, mfd = single_face_or_blip(rng)

    off_center = float(np.clip(rng.normal(70, 14), 38, 98))
    frac_left = rng.beta(2.2, 2.2)  # both sides contribute; sometimes skewed
    center, left, right = make_gaze(off_center * frac_left, off_center * (1 - frac_left))
    events = rng.poisson(3 + off_center / 12)
    if events == 0 and rng.random() < 0.9:
        events = 1
    return row(presence, max_faces, longest, mfd, events, center, left, right)


GENERATORS = {
    "NORMAL": gen_normal,
    "NO_FACE": gen_no_face,
    "MULTIPLE_FACES": gen_multiple_faces,
    "LOOKING_LEFT": gen_looking_left,
    "LOOKING_RIGHT": gen_looking_right,
    "LOOKING_AWAY": gen_looking_away,
}


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------
def class_counts(n_total):
    counts = {k: int(round(n_total * v)) for k, v in CLASS_PROPORTIONS.items()}
    diff = n_total - sum(counts.values())
    counts["NORMAL"] += diff
    return counts


def generate_dataset(n_total=N_SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    counts = class_counts(n_total)
    seen = set()
    records = []

    for label, n in counts.items():
        made = 0
        attempts = 0
        while made < n:
            attempts += 1
            if attempts > n * 50:
                raise DatasetValidationError(
                    f"Could not generate {n} unique rows for {label}."
                )
            rec = GENERATORS[label](rng)
            key = tuple(rec[f] for f in FEATURES)
            if key in seen:  # no identical duplicate rows (across ALL classes)
                continue
            seen.add(key)
            rec[TARGET] = label
            records.append(rec)
            made += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    df.insert(0, "session_id", [f"S{i + 1:05d}" for i in range(len(df))])
    return df[COLUMNS]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_dataset(df, n_expected=N_SAMPLES):
    """Return list of PASS messages, or raise DatasetValidationError."""
    errors, passed = [], []

    def check(ok, pass_msg, fail_msg):
        (passed if ok else errors).append(pass_msg if ok else fail_msg)

    check(list(df.columns) == COLUMNS, "Columns match specification.",
          f"Unexpected columns: {list(df.columns)}")
    check(len(df) == n_expected, f"Row count = {len(df)}.",
          f"Row count {len(df)} != {n_expected}.")

    n_missing = int(df.isnull().sum().sum())
    check(n_missing == 0, "No missing values.", f"{n_missing} missing values found.")
    if n_missing:
        raise DatasetValidationError("Validation FAILED:\n - " + "\n - ".join(errors))

    n_dup_rows = int(df.duplicated(subset=FEATURES + [TARGET]).sum())
    n_dup_feat = int(df.duplicated(subset=FEATURES).sum())
    check(n_dup_rows == 0 and n_dup_feat == 0, "No duplicate rows.",
          f"{max(n_dup_rows, n_dup_feat)} duplicate rows found.")
    n_dup_ids = int(df["session_id"].duplicated().sum())
    check(n_dup_ids == 0, "No duplicate session IDs.", f"{n_dup_ids} duplicate session IDs.")

    pct_cols = ["face_presence_percent", "looking_center_percent",
                "looking_left_percent", "looking_right_percent"]
    bad_pct = int(((df[pct_cols] < 0) | (df[pct_cols] > 100)).any(axis=1).sum())
    check(bad_pct == 0, "All percentages within [0, 100].",
          f"{bad_pct} rows with percentages outside [0, 100].")

    dur_cols = ["longest_face_absence_sec", "multiple_face_duration_sec"]
    bad_dur = int((df[dur_cols] < 0).any(axis=1).sum())
    check(bad_dur == 0, "All durations >= 0.", f"{bad_dur} rows with negative durations.")

    for c in ["maximum_faces_detected", "looking_away_events"]:
        ok = (df[c] >= 0).all() and (df[c] == df[c].round()).all()
        check(bool(ok), f"{c} is a non-negative integer.",
              f"{c} contains negative or non-integer values.")

    bad_lbl = int((~df[TARGET].isin(LABELS)).sum())
    check(bad_lbl == 0, "All labels valid.", f"{bad_lbl} rows with invalid labels.")

    gaze_sum = (df["looking_center_percent"] + df["looking_left_percent"]
                + df["looking_right_percent"])
    bad_sum = int(((gaze_sum - 100).abs() > GAZE_SUM_TOLERANCE).sum())
    check(bad_sum == 0, "Gaze percentages sum to 100 (max abs deviation "
          f"{(gaze_sum - 100).abs().max():.4f}).",
          f"{bad_sum} rows where gaze percentages do not sum to 100.")

    expected = class_counts(n_expected)
    actual = df[TARGET].value_counts().to_dict()
    check(all(actual.get(k, 0) == v for k, v in expected.items()),
          "Class distribution matches target.",
          f"Class counts {actual} != expected {expected}.")

    # Internal logic checks
    multi = df[TARGET] == "MULTIPLE_FACES"
    check(bool(((df.loc[multi, "maximum_faces_detected"] >= 2)
                & (df.loc[multi, "multiple_face_duration_sec"] > 0)).all()),
          "MULTIPLE_FACES rows have faces >= 2 and duration > 0.",
          "Some MULTIPLE_FACES rows violate faces >= 2 / duration > 0.")
    nf = df[TARGET] == "NO_FACE"
    check(bool((df.loc[nf, "longest_face_absence_sec"] > 0).all()),
          "NO_FACE rows have longest absence > 0.",
          "Some NO_FACE rows have longest absence = 0.")
    mf_consistent = ((df["maximum_faces_detected"] >= 2)
                     == (df["multiple_face_duration_sec"] > 0)).all()
    check(bool(mf_consistent),
          "multiple_face_duration > 0 only when max faces >= 2.",
          "Inconsistency between maximum_faces_detected and multiple_face_duration_sec.")
    zero_faces = df["maximum_faces_detected"] == 0
    check(bool((df.loc[zero_faces, "face_presence_percent"] == 0).all()),
          "max faces = 0 implies face presence = 0.",
          "Rows with 0 faces but non-zero face presence.")

    if errors:
        raise DatasetValidationError("Validation FAILED:\n - " + "\n - ".join(errors))
    return passed


def validate_splits(train, val, test, full):
    errors = []
    if len(train) + len(val) + len(test) != len(full):
        errors.append("Split sizes do not add up to the full dataset.")
    ids = [set(d["session_id"]) for d in (train, val, test)]
    if (ids[0] & ids[1]) or (ids[0] & ids[2]) or (ids[1] & ids[2]):
        errors.append("Splits overlap (shared session IDs).")
    for name, d in (("train", train), ("validation", val), ("test", test)):
        if set(d[TARGET].unique()) != set(LABELS):
            errors.append(f"{name} split is missing one or more classes.")
    if errors:
        raise DatasetValidationError("Split validation FAILED:\n - " + "\n - ".join(errors))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Generating synthetic face-activity dataset ...")
    df = generate_dataset()

    print("\nValidating dataset ...")
    results = validate_dataset(df)  # raises on failure -> nothing saved
    for msg in results:
        print(f"  [PASS] {msg}")

    # Stratified 70 / 15 / 15 split (reproducible)
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df[TARGET], random_state=SEED)
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, stratify=train_val[TARGET], random_state=SEED)
    validate_splits(train, val, test, df)
    print("  [PASS] Splits are disjoint, complete and contain all classes.")

    df.to_csv(OUT_DIR / "face_activity_dataset.csv", index=False)
    train.sort_values("session_id").to_csv(OUT_DIR / "train.csv", index=False)
    val.sort_values("session_id").to_csv(OUT_DIR / "validation.csv", index=False)
    test.sort_values("session_id").to_csv(OUT_DIR / "test.csv", index=False)

    # ---- Report ----
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Samples  : {len(df)}")
    print(f"Features : {len(FEATURES)}  ->  {FEATURES}")
    print("\nClass distribution (full dataset):")
    dist = df[TARGET].value_counts().reindex(LABELS)
    for lbl, n in dist.items():
        print(f"  {lbl:<15}{n:>6}  ({n / len(df) * 100:5.1f}%)")

    print("\nSplit sizes:")
    for name, d in (("train", train), ("validation", val), ("test", test)):
        print(f"  {name:<11}{len(d):>6} rows  ({len(d) / len(df) * 100:4.1f}%)")
    print("\nClass distribution per split:")
    split_dist = pd.DataFrame({
        "train": train[TARGET].value_counts(),
        "validation": val[TARGET].value_counts(),
        "test": test[TARGET].value_counts(),
    }).reindex(LABELS)
    print(split_dist.to_string())

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nBasic statistics:")
    print(df[FEATURES].describe().round(2).T.to_string())
    print("\nPer-class feature means:")
    print(df.groupby(TARGET)[FEATURES].mean().reindex(LABELS).round(2).T.to_string())

    print("\nFiles written to:", OUT_DIR)
    for f in ["face_activity_dataset.csv", "train.csv", "validation.csv", "test.csv"]:
        print("  -", f)

    print("\n" + "!" * 60)
    print("WARNING: This dataset is SYNTHETIC.")
    print("High accuracy here does NOT mean the same accuracy on real webcam")
    print("data. Validate on real sessions (lighting, cameras, backgrounds,")
    print("face size/position, glasses, different users, head poses).")
    print("!" * 60)


if __name__ == "__main__":
    main()
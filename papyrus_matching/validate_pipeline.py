#!/usr/bin/env python3

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


RECALL_KS = (1, 5, 10, 15, 20, 30, 40, 50, 100, 200, 500, 1000)


@dataclass
class PairAggregate:
    fragment_a: str
    fragment_b: str
    retrieval_score: float | None
    offsets_b_on_a_yx: np.ndarray
    translation_scores: np.ndarray


def as_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def normalized_percentile_rank(rank: int | None, num_candidates: int) -> float:
    if rank is None:
        return 0.0
    if num_candidates <= 1:
        return 1.0
    percentile = (num_candidates - rank) / (num_candidates - 1)
    return float(np.clip(percentile, 0.0, 1.0))


def metric_summary(results: list[dict], include_npr: bool = False) -> dict:
    summary = {f"R@{k}": mean([item[f"recall_at_{k}"] for item in results]) for k in RECALL_KS}
    summary["MRR"] = mean([item["mrr"] for item in results])
    if include_npr:
        summary["NPR"] = mean([item["normalized_percentile_rank"] for item in results])
    return summary


def recall_result_fields(recall_at) -> dict:
    return {f"recall_at_{k}": recall_at(k) for k in RECALL_KS}


def unordered_pair_key(fragment_a: str, fragment_b: str) -> tuple[str, str]:
    return tuple(sorted((fragment_a, fragment_b)))


def ground_truth_pair_lookup(pairs: list[dict]) -> dict[tuple[str, str], dict]:
    return {unordered_pair_key(pair["fragment_a"], pair["fragment_b"]): pair for pair in pairs}


def load_ground_truth(path: Path) -> tuple[dict[str, dict], dict[str, dict[str, np.ndarray]], list[dict]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    fragments = {fragment["id"]: fragment for fragment in payload["fragments"]}
    directed_offsets: dict[str, dict[str, np.ndarray]] = {fragment_id: {} for fragment_id in fragments}

    for pair in payload["pairs"]:
        fragment_a = pair["fragment_a"]
        fragment_b = pair["fragment_b"]
        offset_b_on_a = np.asarray(pair["offset_b_on_a_yx"], dtype=float)

        # Directed evaluation asks where the query should move onto the candidate.
        directed_offsets[fragment_a][fragment_b] = -offset_b_on_a
        directed_offsets[fragment_b][fragment_a] = offset_b_on_a

    return fragments, directed_offsets, payload["pairs"]


def parse_hdf5_name(path: Path) -> tuple[str, str] | None:
    parts = path.stem.split("__")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def aggregate_hdf5(path: Path, fragment_a: str, fragment_b: str) -> PairAggregate:
    with h5py.File(path, "r") as handle:
        scores = np.asarray(handle["scores"][:], dtype=float)
        translation_ids = np.asarray(handle["translation_ids"][:])
        t_relatives = np.asarray(handle["t_relatives"][:], dtype=float)

    if scores.size == 0 or translation_ids.size == 0 or t_relatives.size == 0:
        return PairAggregate(
            fragment_a=fragment_a,
            fragment_b=fragment_b,
            retrieval_score=None,
            offsets_b_on_a_yx=np.empty((0, 2), dtype=float),
            translation_scores=np.empty((0,), dtype=float),
        )

    order = np.argsort(translation_ids, kind="stable")
    sorted_ids = translation_ids[order]
    sorted_scores = scores[order]
    sorted_offsets = t_relatives[order]

    unique_ids, starts = np.unique(sorted_ids, return_index=True)
    del unique_ids

    translation_scores = np.maximum.reduceat(sorted_scores, starts)
    offsets = sorted_offsets[starts]

    return PairAggregate(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        retrieval_score=as_float_or_none(float(np.nanmax(scores))),
        offsets_b_on_a_yx=offsets,
        translation_scores=translation_scores,
    )


class ScoreStore:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.paths: dict[frozenset[str], tuple[Path, str, str]] = {}
        self.cache: dict[frozenset[str], PairAggregate | None] = {}
        self.warnings: set[str] = set()
        self._index_files()

    def _index_files(self) -> None:
        if not self.results_dir.exists():
            self.warnings.add(f"Results directory does not exist: {self.results_dir}")
            return

        for path in sorted(self.results_dir.glob("*.hdf5")):
            parsed = parse_hdf5_name(path)
            if parsed is None:
                self.warnings.add(f"Skipping malformed HDF5 filename: {path.name}")
                continue
            fragment_a, fragment_b = parsed
            self.paths[frozenset((fragment_a, fragment_b))] = (path, fragment_a, fragment_b)

    def get(self, fragment_a: str, fragment_b: str) -> PairAggregate | None:
        key = frozenset((fragment_a, fragment_b))
        if key in self.cache:
            return self.cache[key]

        entry = self.paths.get(key)
        if entry is None:
            self.warnings.add(f"Missing HDF5 for pair: {fragment_a} / {fragment_b}")
            self.cache[key] = None
            return None

        path, stored_a, stored_b = entry
        try:
            aggregate = aggregate_hdf5(path, stored_a, stored_b)
        except Exception as exc:
            self.warnings.add(f"Failed reading {path.name}: {exc}")
            aggregate = None

        self.cache[key] = aggregate
        return aggregate


def retrieval_score(store: ScoreStore, query: str, candidate: str) -> float | None:
    pair = store.get(query, candidate)
    if pair is None:
        return None
    return pair.retrieval_score


def directed_position_candidates(store: ScoreStore, query: str, candidate: str) -> list[dict]:
    pair = store.get(query, candidate)
    if pair is None or pair.translation_scores.size == 0:
        return []

    if pair.fragment_a == candidate and pair.fragment_b == query:
        directed_offsets = pair.offsets_b_on_a_yx
    elif pair.fragment_a == query and pair.fragment_b == candidate:
        directed_offsets = -pair.offsets_b_on_a_yx
    else:
        store.warnings.add(
            f"HDF5 orientation mismatch for requested pair {query} / {candidate}: "
            f"{pair.fragment_a} / {pair.fragment_b}"
        )
        return []

    return [
        {
            "candidate": candidate,
            "score": float(score),
            "offset_yx": offset.astype(float),
            "translation_index": int(translation_index),
        }
        for translation_index, (score, offset) in enumerate(zip(pair.translation_scores, directed_offsets))
    ]


def serialize_position_item(
    item: dict,
    rank: int,
    is_gt_neighbor: bool,
    is_hit: bool,
    offset_error: float | None,
) -> dict:
    return {
        "rank": int(rank),
        "candidate": item["candidate"],
        "score": float(item["score"]),
        "offset_yx": [float(v) for v in item["offset_yx"]],
        "translation_index": int(item["translation_index"]),
        "is_gt_neighbor": bool(is_gt_neighbor),
        "offset_error_px": as_float_or_none(offset_error),
        "is_hit": bool(is_hit),
    }


def serialize_global_retrieval_item(item: dict, rank: int, is_relevant: bool) -> dict:
    return {
        "rank": int(rank),
        "fragment_a": item["fragment_a"],
        "fragment_b": item["fragment_b"],
        "score": as_float_or_none(item["score"]),
        "is_relevant": bool(is_relevant),
        "is_missing": item["score"] is None,
    }


def serialize_global_position_item(
    item: dict,
    rank: int,
    is_gt_pair: bool,
    is_hit: bool,
    offset_error: float | None,
) -> dict:
    return {
        "rank": int(rank),
        "fragment_a": item["fragment_a"],
        "fragment_b": item["fragment_b"],
        "score": float(item["score"]),
        "offset_b_on_a_yx": [float(v) for v in item["offset_b_on_a_yx"]],
        "translation_index": int(item["translation_index"]),
        "is_gt_pair": bool(is_gt_pair),
        "offset_error_px": as_float_or_none(offset_error),
        "is_hit": bool(is_hit),
    }


def rank_retrieval(
    store: ScoreStore,
    fragments: dict[str, dict],
    query: str,
    relevant: set[str],
    diagnostic_top_k: int,
) -> dict:
    ranked = []
    for candidate in fragments:
        if candidate == query:
            continue
        score = retrieval_score(store, query, candidate)
        ranked.append({"candidate": candidate, "score": score})

    ranked.sort(key=lambda item: (item["score"] is not None, item["score"] if item["score"] is not None else -np.inf), reverse=True)

    def is_hit(item: dict) -> bool:
        return item["score"] is not None and item["candidate"] in relevant

    def recall_at(k: int) -> float:
        top_candidates = {item["candidate"] for item in ranked[:k] if is_hit(item)}
        return len(top_candidates & relevant) / len(relevant) if relevant else 0.0

    first_rank = None
    first_relevant = None
    for rank, item in enumerate(ranked, start=1):
        if is_hit(item):
            first_rank = rank
            first_relevant = {
                "rank": int(rank),
                "candidate": item["candidate"],
                "score": as_float_or_none(item["score"]),
                "is_relevant": True,
                "is_missing": False,
            }
            break

    diagnostics = [
        {
            "rank": int(rank),
            "candidate": item["candidate"],
            "score": as_float_or_none(item["score"]),
            "is_relevant": item["candidate"] in relevant,
            "is_missing": item["score"] is None,
        }
        for rank, item in enumerate(ranked[:diagnostic_top_k], start=1)
    ]

    return {
        **recall_result_fields(recall_at),
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        "first_relevant_rank": first_rank,
        "first_relevant": first_relevant,
        "top": diagnostics,
    }


def rank_global_retrieval(
    store: ScoreStore,
    fragments: dict[str, dict],
    pairs: list[dict],
    diagnostic_top_k: int,
) -> dict:
    gt_pairs = ground_truth_pair_lookup(pairs)
    relevant_keys = set(gt_pairs)

    ranked = []
    for fragment_a, fragment_b in itertools.combinations(fragments, 2):
        score = retrieval_score(store, fragment_a, fragment_b)
        ranked.append(
            {
                "fragment_a": fragment_a,
                "fragment_b": fragment_b,
                "score": score,
                "pair_key": unordered_pair_key(fragment_a, fragment_b),
            }
        )

    ranked.sort(
        key=lambda item: (
            item["score"] is not None,
            item["score"] if item["score"] is not None else -np.inf,
        ),
        reverse=True,
    )

    def is_hit(item: dict) -> bool:
        return item["score"] is not None and item["pair_key"] in relevant_keys

    def recall_at(k: int) -> float:
        top_pairs = {item["pair_key"] for item in ranked[:k] if is_hit(item)}
        return len(top_pairs & relevant_keys) / len(relevant_keys) if relevant_keys else 0.0

    first_rank = None
    first_relevant = None
    rank_1_record = None
    for rank, item in enumerate(ranked, start=1):
        hit = is_hit(item)
        if rank == 1:
            rank_1_record = serialize_global_retrieval_item(
                item=item,
                rank=rank,
                is_relevant=item["pair_key"] in relevant_keys,
            )
        if hit:
            first_rank = rank
            first_relevant = serialize_global_retrieval_item(
                item=item,
                rank=rank,
                is_relevant=True,
            )
            break

    diagnostics = [
        serialize_global_retrieval_item(
            item=item,
            rank=rank,
            is_relevant=item["pair_key"] in relevant_keys,
        )
        for rank, item in enumerate(ranked[:diagnostic_top_k], start=1)
    ]

    return {
        **recall_result_fields(recall_at),
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        "first_relevant_rank": first_rank,
        "num_pair_candidates": len(ranked),
        "num_relevant_pairs": len(relevant_keys),
        "rank_1": rank_1_record,
        "first_relevant": first_relevant,
        "top": diagnostics,
    }


def rank_positioning(
    store: ScoreStore,
    fragments: dict[str, dict],
    query: str,
    gt_offsets: dict[str, np.ndarray],
    position_tolerance: float,
    diagnostic_top_k: int,
) -> dict:
    candidates = []
    for candidate in fragments:
        if candidate == query:
            continue
        candidates.extend(directed_position_candidates(store, query, candidate))

    candidates.sort(key=lambda item: item["score"], reverse=True)
    relevant = set(gt_offsets)

    def is_hit(item: dict) -> tuple[bool, float | None]:
        target_offset = gt_offsets.get(item["candidate"])
        if target_offset is None:
            return False, None
        error = float(np.linalg.norm(item["offset_yx"] - target_offset))
        return error <= position_tolerance, error

    def recall_at(k: int) -> float:
        hit_fragments = set()
        for item in candidates[:k]:
            hit, _ = is_hit(item)
            if hit:
                hit_fragments.add(item["candidate"])
        return len(hit_fragments & relevant) / len(relevant) if relevant else 0.0

    first_rank = None
    first_hit_record = None
    rank_1_record = None
    for rank, item in enumerate(candidates, start=1):
        hit, error = is_hit(item)
        if rank == 1:
            rank_1_record = serialize_position_item(
                item=item,
                rank=rank,
                is_gt_neighbor=item["candidate"] in relevant,
                is_hit=hit,
                offset_error=error,
            )
        if hit:
            first_rank = rank
            first_hit_record = serialize_position_item(
                item=item,
                rank=rank,
                is_gt_neighbor=True,
                is_hit=True,
                offset_error=error,
            )
            break

    diagnostics = []
    for rank, item in enumerate(candidates[:diagnostic_top_k], start=1):
        hit, error = is_hit(item)
        diagnostics.append(
            serialize_position_item(
                item=item,
                rank=rank,
                is_gt_neighbor=item["candidate"] in relevant,
                is_hit=hit,
                offset_error=error,
            )
        )

    return {
        **recall_result_fields(recall_at),
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        "first_hit_rank": first_rank,
        "normalized_percentile_rank": normalized_percentile_rank(first_rank, len(candidates)),
        "num_position_candidates": len(candidates),
        "rank_1": rank_1_record,
        "first_hit": first_hit_record,
        "top": diagnostics,
    }


def global_position_candidates(store: ScoreStore, fragments: dict[str, dict]) -> list[dict]:
    candidates = []
    for fragment_a, fragment_b in itertools.combinations(fragments, 2):
        pair = store.get(fragment_a, fragment_b)
        if pair is None or pair.translation_scores.size == 0:
            continue

        for translation_index, (score, offset) in enumerate(
            zip(pair.translation_scores, pair.offsets_b_on_a_yx)
        ):
            candidates.append(
                {
                    "fragment_a": pair.fragment_a,
                    "fragment_b": pair.fragment_b,
                    "score": float(score),
                    "offset_b_on_a_yx": offset.astype(float),
                    "translation_index": int(translation_index),
                    "pair_key": unordered_pair_key(pair.fragment_a, pair.fragment_b),
                }
            )
    return candidates


def gt_offset_for_orientation(gt_pair: dict, fragment_a: str, fragment_b: str) -> np.ndarray | None:
    offset_b_on_a = np.asarray(gt_pair["offset_b_on_a_yx"], dtype=float)

    if gt_pair["fragment_a"] == fragment_a and gt_pair["fragment_b"] == fragment_b:
        return offset_b_on_a
    if gt_pair["fragment_a"] == fragment_b and gt_pair["fragment_b"] == fragment_a:
        return -offset_b_on_a
    return None


def rank_global_positioning(
    store: ScoreStore,
    fragments: dict[str, dict],
    pairs: list[dict],
    position_tolerance: float,
    diagnostic_top_k: int,
) -> dict:
    gt_pairs = ground_truth_pair_lookup(pairs)
    relevant_keys = set(gt_pairs)
    candidates = global_position_candidates(store, fragments)
    candidates.sort(key=lambda item: item["score"], reverse=True)

    def is_hit(item: dict) -> tuple[bool, float | None]:
        gt_pair = gt_pairs.get(item["pair_key"])
        if gt_pair is None:
            return False, None

        target_offset = gt_offset_for_orientation(gt_pair, item["fragment_a"], item["fragment_b"])
        if target_offset is None:
            return False, None

        error = float(np.linalg.norm(item["offset_b_on_a_yx"] - target_offset))
        return error <= position_tolerance, error

    def recall_at(k: int) -> float:
        hit_pairs = set()
        for item in candidates[:k]:
            hit, _ = is_hit(item)
            if hit:
                hit_pairs.add(item["pair_key"])
        return len(hit_pairs & relevant_keys) / len(relevant_keys) if relevant_keys else 0.0

    first_rank = None
    first_hit_record = None
    rank_1_record = None
    for rank, item in enumerate(candidates, start=1):
        hit, error = is_hit(item)
        if rank == 1:
            rank_1_record = serialize_global_position_item(
                item=item,
                rank=rank,
                is_gt_pair=item["pair_key"] in relevant_keys,
                is_hit=hit,
                offset_error=error,
            )
        if hit:
            first_rank = rank
            first_hit_record = serialize_global_position_item(
                item=item,
                rank=rank,
                is_gt_pair=True,
                is_hit=True,
                offset_error=error,
            )
            break

    diagnostics = []
    for rank, item in enumerate(candidates[:diagnostic_top_k], start=1):
        hit, error = is_hit(item)
        diagnostics.append(
            serialize_global_position_item(
                item=item,
                rank=rank,
                is_gt_pair=item["pair_key"] in relevant_keys,
                is_hit=hit,
                offset_error=error,
            )
        )

    return {
        **recall_result_fields(recall_at),
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        "first_hit_rank": first_rank,
        "normalized_percentile_rank": normalized_percentile_rank(first_rank, len(candidates)),
        "num_position_candidates": len(candidates),
        "num_relevant_pairs": len(relevant_keys),
        "rank_1": rank_1_record,
        "first_hit": first_hit_record,
        "top": diagnostics,
    }


def rank_positioning_given_pair(
    store: ScoreStore,
    query: str,
    candidate: str,
    target_offset: np.ndarray,
    position_tolerance: float,
    diagnostic_top_k: int,
) -> dict:
    candidates = directed_position_candidates(store, query, candidate)
    candidates.sort(key=lambda item: item["score"], reverse=True)

    def is_hit(item: dict) -> tuple[bool, float]:
        error = float(np.linalg.norm(item["offset_yx"] - target_offset))
        return error <= position_tolerance, error

    def recall_at(k: int) -> float:
        for item in candidates[:k]:
            hit, _ = is_hit(item)
            if hit:
                return 1.0
        return 0.0

    first_rank = None
    first_hit_record = None
    rank_1_record = None
    for rank, item in enumerate(candidates, start=1):
        hit, error = is_hit(item)
        if rank == 1:
            rank_1_record = serialize_position_item(
                item=item,
                rank=rank,
                is_gt_neighbor=True,
                is_hit=hit,
                offset_error=error,
            )
        if hit:
            first_rank = rank
            first_hit_record = serialize_position_item(
                item=item,
                rank=rank,
                is_gt_neighbor=True,
                is_hit=True,
                offset_error=error,
            )
            break

    diagnostics = []
    for rank, item in enumerate(candidates[:diagnostic_top_k], start=1):
        hit, error = is_hit(item)
        diagnostics.append(
            serialize_position_item(
                item=item,
                rank=rank,
                is_gt_neighbor=True,
                is_hit=hit,
                offset_error=error,
            )
        )

    return {
        "query": query,
        "candidate": candidate,
        "gt_offset_yx": [float(v) for v in target_offset],
        **recall_result_fields(recall_at),
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        "first_hit_rank": first_rank,
        "normalized_percentile_rank": normalized_percentile_rank(first_rank, len(candidates)),
        "num_position_candidates": len(candidates),
        "rank_1": rank_1_record,
        "first_hit": first_hit_record,
        "top": diagnostics,
    }


def evaluate(
    gt_json: Path,
    results_dir: Path,
    position_tolerance: float,
    diagnostic_top_k: int,
) -> dict:
    fragments, directed_offsets, pairs = load_ground_truth(gt_json)
    store = ScoreStore(results_dir)

    query_ids = [fragment_id for fragment_id in fragments if directed_offsets.get(fragment_id)]
    retrieval_results = []
    positioning_results = []
    positioning_given_pair_results = []
    query_diagnostics = []

    for query in query_ids:
        relevant = set(directed_offsets[query])
        retrieval = rank_retrieval(
            store=store,
            fragments=fragments,
            query=query,
            relevant=relevant,
            diagnostic_top_k=diagnostic_top_k,
        )
        positioning = rank_positioning(
            store=store,
            fragments=fragments,
            query=query,
            gt_offsets=directed_offsets[query],
            position_tolerance=position_tolerance,
            diagnostic_top_k=diagnostic_top_k,
        )
        positioning_given_pair = [
            rank_positioning_given_pair(
                store=store,
                query=query,
                candidate=candidate,
                target_offset=target_offset,
                position_tolerance=position_tolerance,
                diagnostic_top_k=diagnostic_top_k,
            )
            for candidate, target_offset in sorted(directed_offsets[query].items())
        ]

        retrieval_results.append(retrieval)
        positioning_results.append(positioning)
        positioning_given_pair_results.extend(positioning_given_pair)
        query_diagnostics.append(
            {
                "query": query,
                "gt_neighbors": sorted(relevant),
                "retrieval": retrieval,
                "positioning": positioning,
                "positioning_given_pair": positioning_given_pair,
            }
        )

    global_retrieval = rank_global_retrieval(
        store=store,
        fragments=fragments,
        pairs=pairs,
        diagnostic_top_k=diagnostic_top_k,
    )
    global_positioning = rank_global_positioning(
        store=store,
        fragments=fragments,
        pairs=pairs,
        position_tolerance=position_tolerance,
        diagnostic_top_k=diagnostic_top_k,
    )

    return {
        "summary": {
            "gt_json": gt_json.as_posix(),
            "results_dir": results_dir.as_posix(),
            "position_tolerance": float(position_tolerance),
            "num_fragments": len(fragments),
            "num_gt_pairs": len(pairs),
            "num_directed_gt_pairs": len(positioning_given_pair_results),
            "num_queries": len(query_ids),
            "num_global_pair_candidates": global_retrieval["num_pair_candidates"],
            "num_global_position_candidates": global_positioning["num_position_candidates"],
            "retrieval": metric_summary(retrieval_results),
            "positioning": metric_summary(positioning_results, include_npr=True),
            "positioning_given_pair": metric_summary(positioning_given_pair_results, include_npr=True),
            "global_retrieval": metric_summary([global_retrieval]),
            "global_positioning": metric_summary([global_positioning], include_npr=True),
            "num_warnings": len(store.warnings),
        },
        "global": {
            "retrieval": global_retrieval,
            "positioning": global_positioning,
        },
        "queries": query_diagnostics,
        "warnings": sorted(store.warnings),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate validation retrieval and positioning metrics from HDF5 scores.")
    parser.add_argument("--gt-json", type=Path, required=True, help="Ground-truth JSON from prepare_validation_set.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing raw precompute HDF5 files.")
    parser.add_argument("--position-tolerance", type=float, default=25.0, help="Position tolerance in pixels.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write metrics and diagnostics JSON.")
    parser.add_argument("--diagnostic-top-k", type=int, default=50, help="Number of top candidates to keep per query.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output = evaluate(
        gt_json=args.gt_json,
        results_dir=args.results_dir,
        position_tolerance=args.position_tolerance,
        diagnostic_top_k=args.diagnostic_top_k,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    retrieval = output["summary"]["retrieval"]
    positioning = output["summary"]["positioning"]
    positioning_given_pair = output["summary"]["positioning_given_pair"]
    global_retrieval = output["summary"]["global_retrieval"]
    global_positioning = output["summary"]["global_positioning"]
    print("[INFO] Retrieval:", retrieval)
    print("[INFO] Positioning:", positioning)
    print("[INFO] Positioning given pair:", positioning_given_pair)
    print("[INFO] Global retrieval:", global_retrieval)
    print("[INFO] Global positioning:", global_positioning)
    print(f"[INFO] Saved validation metrics to {args.output_json}")
    if output["warnings"]:
        print(f"[WARNING] {len(output['warnings'])} warning(s); see output JSON.")


if __name__ == "__main__":
    main()

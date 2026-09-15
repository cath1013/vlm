# Audit of the map-prior trajectory heuristic (`rule`)

Date: 2026-08-29

## Decision

**Do not use `rule` as a primary or established scientific baseline.** Keep it only as a
secondary, exploratory ablation under the name **map-prior constant-speed heuristic**, and
only when the road network comes from CARLA OpenDRIVE (or another map that is independent
of the evaluated scenario's future).

The implementation is deterministic, reproducible, and does not read collision labels in
`rule_predict()` itself. Its basic mechanics are covered by tests, and its recorded test-set
trajectory error is slightly better than constant velocity. However:

1. its probability constants and thresholds have no cited or independently validated basis;
2. the returned values are heuristic priors, not calibrated probabilities;
3. several code comments show that rules were revised after inspecting dataset-wide behavior,
   so the archived test result is not a pristine evaluation of a frozen baseline;
4. if an external map is unavailable, the pipeline synthesizes a map from the complete
   scenario trajectories, which exposes future route geometry;
5. the strongest published-looking route-selection number is calculated only after excluding
   unmatched, single-candidate, and ambiguous examples.

Until these issues are addressed, use **constant velocity** as the primary transparent
trajectory baseline. Report `rule` only to characterize the legacy pipeline.

## What was audited

- Runtime prediction: [`traffic_llm/prediction.py`](../traffic_llm/prediction.py)
- Candidate generation and priors: [`traffic_llm/roadmap.py`](../traffic_llm/roadmap.py)
- Map loading and fallback: [`traffic_llm/da_runner.py`](../traffic_llm/da_runner.py)
- Trajectory-derived map construction: [`traffic_llm/roadgen.py`](../traffic_llm/roadgen.py)
- Dataset and labels: [`examples/make_predict_dataset.py`](../examples/make_predict_dataset.py)
- Recorded RankNet comparison: `out/predict_model/train_report_rank.json`
- Recorded coordinate comparison: `out/predict_model/train_report_waypoints.json`
- Relevant tests: `tests/test_pipeline.py` and `tests/test_deepaccident.py`
- Provenance discussion: [`meeting.txt`](../meeting.txt) and the restored Claude session history

The audit distinguishes the deterministic runtime algorithm from the process that originally
authored it. The meeting notes say Claude produced the rule-based method and that its rationale
was not clear to the speaker. The repository does not contain a design citation or a human
validation record for the priors.

## Exact runtime algorithm

For each actor, the pipeline builds a `PredictContext` from the current actor state and road
network. `rule_predict()` then follows these branches:

| Condition | Output |
|---|---|
| Speed is unknown | One point at the current position; prediction deferred |
| Speed is below 0.5 m/s | One point at the current position; "remain stopped" |
| No road candidate exists | Constant-velocity straight-line extrapolation |
| Road candidates exist | Follow every map candidate at the actor's current constant speed and attach the map prior |

The path geometry is sampled at one-second intervals. It assumes the current speed remains
constant; acceleration, braking intent, turn signals, interaction with other actors, and traffic
signals do not affect the predicted coordinates.

If multiple candidates become geometrically identical at the output resolution, they are merged
and their prior mass is summed. Candidates removed for invalid geometry cause the remaining
priors to be renormalized.

### Candidate prior rules

When a reachable junction exists, exits are classified from their heading change:

| Heading change | Maneuver | Initial weight |
|---:|---|---:|
| less than 30 degrees | Straight | 0.60 |
| 30 to 150 degrees, clockwise | Right | 0.20 |
| 30 to 150 degrees, counter-clockwise | Left | 0.20 |
| greater than 150 degrees | U-turn | 0.02 |

Lane position then changes turn weights:

- innermost lane and left turn: multiply by 2.0;
- other lanes and left turn: multiply by 0.4;
- outermost lane and right turn: multiply by 2.0;
- other lanes and right turn: multiply by 0.4.

Weights are normalized across maneuver types and divided among exits with the same maneuver
label. These constants are hard-coded in `RoadNetwork.downstream_paths()`. No citation,
calibration experiment, likelihood fit, or expert sign-off was found for them. They should
therefore be called **priors/weights**, not probabilities in a statistical sense.

### Other consequential constants

| Constant | Meaning | Evidence status |
|---:|---|---|
| 0.5 m/s | Below this, predict that the actor remains stopped | Uncited |
| 5 m | If no more than this distance remains after the current road, do not branch at a junction | Uncited |
| 5 m | Spatial sampling interval along the map path | Implementation choice |
| 0.5 m | Prepend the actor position if the map path starts farther away | Implementation tolerance |
| 20 degrees | Road-generation heading grouping tolerance | Uncited, dataset-informed |
| 2 m | Road-generation same-corridor lateral tolerance | Uncited, dataset-informed |
| 12 m | Road-generation junction snapping radius | Uncited, dataset-informed |

The last three affect only the trajectory-synthesized map fallback, not a correctly loaded
OpenDRIVE map.

## Leakage assessment

### Runtime rule: no direct label leakage

`rule_predict()` receives only:

- the current `ActorState`;
- the prediction horizon;
- candidate paths from the road network.

It does not import or inspect collision truth, accident labels, future positions, or the
window's grading record. With an independently supplied OpenDRIVE map, the rule is causal with
respect to the information visible in its function boundary.

### Critical fallback risk: future-derived road map

When no GeoJSON or OpenDRIVE map is supplied, `DeepAccidentRunner.build_network()` calls
`collect_tracks()`, which reads agent poses and ground-truth object tracks over the complete
scenario. `synthesize_road_network()` then infers lanes, roads, and junctions from those tracks.

This does not directly reveal that a collision occurs, but it reveals future route geometry and
which roads/lanes actors eventually traverse. That is inappropriate for a strict online
anticipation evaluation. The fallback is useful for visualization or dataset exploration, but
it must not be used for a publishable causal baseline.

Enforced safeguard for future experiments:

```text
published evaluation => OpenDRIVE map required
missing town map      => fail the scenario; never silently synthesize it
```

`examples/run_deepaccident.py` enforces this for window, evaluation-record, and standalone
payload generation. Exploratory work can opt in explicitly with `--allow-trajectory-map`; the
manifest then records both the source and that it was built from scenario trajectories.
`examples/make_predict_dataset.py` applies the same rule to RankNet/WaypointNet dataset
generation and refuses partial town-map coverage unless the exploratory override is explicit.

### Post-hoc development contamination

Several comments document fixes motivated by dataset-wide measurements, such as reversed route
rates, candidate probability sums, and lane-offset behavior. Those fixes may be correct software
repairs, but they mean the heuristic evolved after observing the dataset that is also called the
test set. There is no versioned record showing that its constants were frozen before test-set
inspection.

This is not direct per-example label leakage. It does mean the existing test figures should be
described as retrospective engineering measurements, not an untouched held-out benchmark.

## What the existing tests establish

The targeted audit suite executed 76 tests successfully. It covers:

- straight and junction candidate generation;
- lane-dependent left/right weighting;
- heading convention for left and right turns;
- path alignment with the actor's lane rather than the road centerline;
- stopped and no-map fallbacks;
- probability-mass preservation and duplicate merging;
- OpenDRIVE parsing and lane counts;
- pluggable predictor compatibility with RankNet and WaypointNet.

Result:

```text
76 passed
```

These are implementation and invariant tests. They show that the code behaves as written; they
do not validate that the chosen weights represent real driver behavior or that the resulting
probabilities are calibrated.

## Existing quantitative evidence

### Comprehensive coordinate evaluation

The WaypointNet training report evaluates all 23,453 test samples and contains a rule trajectory
baseline using the highest-prior candidate:

| Method | ADE | FDE |
|---|---:|---:|
| Constant velocity | 4.133 m | 7.423 m |
| `rule` top-prior path | 4.007 m | 6.996 m |
| WaypointNet | 3.010 m | 5.250 m |

Relative to constant velocity, `rule` improves ADE by 0.126 m (about 3.0%) and FDE by 0.427 m
(about 5.8%). This is a small but real recorded advantage, not enough to compensate for the
provenance and leakage concerns.

On the 4,975 samples whose true trajectory lies outside the candidate set, `rule` performs very
poorly:

| Subset | Rule ADE | Constant-velocity ADE | WaypointNet ADE |
|---|---:|---:|---:|
| Off-candidate test samples | 8.262 m | 7.370 m | 4.782 m |

This is expected: the rule cannot choose a trajectory that the map candidate generator did not
produce.

### Filtered route-selection evaluation

The RankNet report gives `rule` 60.95% test top-1 accuracy and RankNet 68.79%. That comparison
uses 3,211 test records after filtering the source data:

- 32,481 unmatched records excluded;
- 85,446 single-candidate records excluded;
- 34,080 ambiguous records excluded.

The filtered result is useful for comparing two candidate rankers on the task they share. It is
not an overall trajectory success rate and should not be quoted without the conditioning.

### Missing validation

No evidence was found for:

- probability calibration (ECE, Brier score, reliability diagram);
- robustness across prior/threshold choices;
- an independent expert review of the traffic assumptions;
- a frozen development/test protocol for the heuristic;
- downstream accident-anticipation results over a representative validation set;
- a comparison against recognized trajectory baselines beyond constant velocity.

## Failure modes

1. **Braking and acceleration:** paths use current constant speed, so an accelerating or braking
   actor is systematically misplaced.
2. **Stopped actors:** every actor below 0.5 m/s is predicted to remain stopped for the horizon,
   even at a traffic light that may turn green.
3. **Interaction blindness:** no leading vehicle, right-of-way, collision avoidance, or yielding
   behavior affects the trajectory.
4. **Uncalibrated multimodality:** a 0.6 prior does not mean a 60% empirical probability.
5. **Candidate-set ceiling:** lane changes, road departures, unusual evasive motion, and missing
   map connections cannot be predicted.
6. **Lane-index sensitivity:** incorrect localization or lane indexing changes turn weights by a
   factor of five (2.0 versus 0.4).
7. **Map fallback leakage:** full-scenario trajectories can determine the available road geometry.
8. **Road-end truncation:** the rule may return fewer than five future steps, reducing downstream
   temporal coverage.

## Conditions for retaining `rule`

Retain it only if all of the following are adopted:

1. Rename it in papers and tables to **map-prior constant-speed heuristic**.
2. Require CARLA OpenDRIVE for every evaluated scenario and record the map source in each
   manifest.
3. Report constant velocity beside it; do not let `rule` replace the transparent baseline.
4. Treat weights as heuristic priors and do not make probability-calibration claims.
5. Put it in an ablation/appendix, not the headline DeepAccident comparison.
6. Report the comprehensive ADE/FDE numbers and candidate coverage, not only filtered top-1.
7. Freeze the implementation before the next validation run.

If enforcing independent maps is impractical, discard `rule` from the evaluation rather than
using trajectory-synthesized maps.

## Recommended experiment lineup

Primary evaluation:

```text
constant velocity       transparent non-learned trajectory baseline
WaypointNet             primary learned trajectory predictor
DeepAccident comparator external accident-prediction comparison
```

Secondary ablation:

```text
map-prior heuristic (`rule`)
RankNet candidate ranker
WaypointNet coordinate regressor
```

This preserves the useful historical comparison without asking readers to trust an unaudited
Claude-generated heuristic as the scientific reference point.

# Adaptive Source Gate Sweep

Raw input: `llada_moe_planning_revision_constraint_span_multisource_v1_raw.jsonl`

## Findings

- Score-maximal plateau: gap min `6`, quality floor `0.25`, repair `0.474107`, `58` generations, added `plan_002,plan_006`.
- Efficiency-maximal plateau: gap min `10`, quality floor `0.25`, repair `0.472768`, `57` generations, gain/extra generation `0.025794`, added `plan_002`.
- Efficiency mode loses `0.001339` mean task score versus score-max and spends fewer generations.
- Named `score_max` mode (`gap=6`, `quality=0.25`) is on the same operating plateau.
- Named `efficiency` mode (`gap=10`, `quality=0.25`) is on the same operating plateau.

## Score-Sorted Grid

| Gap Min | Quality Floor | Generations | Repair | Delta vs Evolved | Extra Budget | Gain/Extra Gen | W/T/L | Added Tasks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 6 | 0.25 | 58 | 0.474107 | 0.030357 | 1.25 | 0.024286 | 7/1/0 | plan_002,plan_006 |
| 10 | 0.25 | 57 | 0.472768 | 0.029018 | 1.12 | 0.025794 | 7/1/0 | plan_002 |

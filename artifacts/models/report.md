# Phase 2 model report

Trained 2026-10-13 on seasons 2018-2025 (regular season + playoffs), validated on the last season with models fit on the earlier seasons. Lookup tables for the comparison were rebuilt from 2021 to the season before validation, so no validation play touches either competitor. Win-probability filter (2%-98%) applied except to the late-game models.

## Summary

| Model | Val rows | Log loss: model | situation-only | table | constant | Beats table | Beats constant | Calibration (deciles ≥500 within 3 pts) | C | Team features |
|---|---|---|---|---|---|---|---|---|---|---|
| play_call | 33571 | 0.5592 | 0.5604 | 0.5652 | 0.6730 | yes | yes | yes | 1.0 | yes |
| fourth_down | 3997 | 0.2901 | 0.2925 | 0.4227 | 1.0396 | yes | yes | yes | 1.0 | yes |
| air_yards | 16556 | 1.7110 | 1.7129 | 1.7180 | 1.8039 | yes | yes | yes | 1.0 | yes |
| dropback_outcome | 18816 | 0.4449 | 0.4475 | 0.4484 | 0.4513 | yes | yes | yes | 10.0 | yes |
| throw_outcome | 16556 | 0.6861 | 0.6881 | 0.6901 | 0.7345 | yes | yes | yes | 0.1 | yes |
| yac | 10628 | 1.8578 | 1.8582 | 1.8693 | 2.0449 | yes | yes | yes | 10.0 | yes |
| rush_yards | 12191 | 2.6607 | 2.6613 | 2.6702 | 2.7376 | yes | yes | yes | 0.1 | yes |
| fumble_lost | 25080 | 0.0431 |  | 0.0435 | 0.0479 | yes | yes | yes | 10.0 | no |
| fg_make | 1067 | 0.3593 |  | 0.3615 | 0.4148 | yes | yes | yes | 1.0 | no |
| pat_choice | 1460 | 0.1611 |  | 0.2357 | 0.3005 | yes | yes | yes | 10.0 | no |
| clock_runoff | 37519 | 1.3477 | 1.3514 | 1.5177 | 2.0570 | yes | yes | yes | 10.0 | yes |
| presnap_penalty | 36491 | 0.2738 |  | 0.2740 | 0.2762 | yes | yes | yes | 0.1 | no |

Spec note: the spec's single `pass_outcome` model (5 classes) is implemented as `dropback_outcome` (throw / sack / scramble) followed by `throw_outcome` (complete / incomplete / interception given a throw), because sacks and scrambles have no air yards to condition on. The two multiply to the five-class distribution.

## play_call

Rows: 270393 total, 33571 validation. Features: 31. Best C = 1.0.

Log loss on validation: model 0.5592, Phase 1 table 0.5652, constant 0.6730.

Calibration (P(class 1)):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 3358 | 0.044 | 0.063 | -0.019 |
| 2 | 3357 | 0.109 | 0.106 | +0.003 |
| 3 | 3357 | 0.212 | 0.206 | +0.006 |
| 4 | 3357 | 0.298 | 0.284 | +0.014 |
| 5 | 3357 | 0.382 | 0.365 | +0.017 |
| 6 | 3357 | 0.460 | 0.483 | -0.023 |
| 7 | 3357 | 0.514 | 0.521 | -0.007 |
| 8 | 3357 | 0.562 | 0.552 | +0.010 |
| 9 | 3357 | 0.627 | 0.640 | -0.013 |
| 10 | 3357 | 0.789 | 0.774 | +0.015 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| log_ydstogo | run | -1.309 |
| down_3 | run | -1.080 |
| yardline | run | -0.983 |
| down_1 | run | +0.848 |
| yardline_sq | run | +0.822 |

Sanity read: passing should rise with ydstogo, trailing_late and down_3; fall with leading_late

## fourth_down

Rows: 31448 total, 3997 validation. Features: 24. Best C = 1.0.

Log loss on validation: model 0.2901, Phase 1 table 0.4227, constant 1.0396.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 400 | 0.539 | 0.565 | -0.026 |
| 2 | 400 | 0.681 | 0.728 | -0.047 |
| 3 | 400 | 0.790 | 0.772 | +0.017 |
| 4 | 400 | 0.874 | 0.892 | -0.019 |
| 5 | 400 | 0.927 | 0.945 | -0.018 |
| 6 | 400 | 0.959 | 0.965 | -0.006 |
| 7 | 400 | 0.977 | 0.975 | +0.002 |
| 8 | 399 | 0.988 | 0.985 | +0.003 |
| 9 | 399 | 0.995 | 0.997 | -0.003 |
| 10 | 399 | 0.998 | 0.997 | +0.001 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| yardline_sq | go | +5.077 |
| yardline_sq | field_goal | -4.614 |
| fg_distance_sq | field_goal | -3.427 |
| yardline | punt | +3.235 |
| fg_distance | punt | +3.235 |

Sanity read: go rate should fall with ydstogo; punt should rise with yardline (own territory); FG with short fg_distance

## air_yards

Rows: 135319 total, 16556 validation. Features: 57. Best C = 1.0.

Log loss on validation: model 1.7110, Phase 1 table 1.7180, constant 1.8039.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1656 | 0.218 | 0.224 | -0.006 |
| 2 | 1656 | 0.237 | 0.233 | +0.004 |
| 3 | 1656 | 0.244 | 0.251 | -0.007 |
| 4 | 1656 | 0.250 | 0.262 | -0.012 |
| 5 | 1656 | 0.256 | 0.246 | +0.009 |
| 6 | 1656 | 0.263 | 0.272 | -0.009 |
| 7 | 1655 | 0.272 | 0.282 | -0.009 |
| 8 | 1655 | 0.286 | 0.284 | +0.002 |
| 9 | 1655 | 0.311 | 0.303 | +0.008 |
| 10 | 1655 | 0.560 | 0.566 | -0.006 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| i_d3_logyd | 0 | +0.731 |
| down_3 | 0 | -0.668 |
| red_zone | 5 | -0.651 |
| red_zone | 6 | -0.610 |
| i_d3_logyd | 2 | -0.557 |

Sanity read: deep bins rise with ydstogo and trailing_late; fall near the goal line (yardline small)

## dropback_outcome

Rows: 152045 total, 18816 validation. Features: 24. Best C = 10.0.

Log loss on validation: model 0.4449, Phase 1 table 0.4484, constant 0.4513.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1882 | 0.813 | 0.810 | +0.003 |
| 2 | 1882 | 0.851 | 0.846 | +0.005 |
| 3 | 1882 | 0.870 | 0.860 | +0.010 |
| 4 | 1882 | 0.885 | 0.880 | +0.005 |
| 5 | 1882 | 0.896 | 0.872 | +0.024 |
| 6 | 1882 | 0.904 | 0.887 | +0.017 |
| 7 | 1881 | 0.911 | 0.896 | +0.015 |
| 8 | 1881 | 0.917 | 0.914 | +0.003 |
| 9 | 1881 | 0.924 | 0.916 | +0.008 |
| 10 | 1881 | 0.935 | 0.918 | +0.017 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| yardline | throw | -0.316 |
| yardline | sack | +0.306 |
| yardline_sq | throw | +0.285 |
| yardline_sq | sack | -0.239 |
| log_ydstogo | sack | +0.167 |

Sanity read: sacks and scrambles should rise with ydstogo / down_3 (long-developing plays)

## throw_outcome

Rows: 135319 total, 16556 validation. Features: 45. Best C = 0.1.

Log loss on validation: model 0.6861, Phase 1 table 0.6901, constant 0.7345.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1656 | 0.503 | 0.527 | -0.024 |
| 2 | 1656 | 0.545 | 0.550 | -0.005 |
| 3 | 1656 | 0.582 | 0.595 | -0.012 |
| 4 | 1656 | 0.623 | 0.643 | -0.019 |
| 5 | 1656 | 0.667 | 0.678 | -0.010 |
| 6 | 1656 | 0.706 | 0.713 | -0.007 |
| 7 | 1655 | 0.736 | 0.720 | +0.016 |
| 8 | 1655 | 0.764 | 0.761 | +0.003 |
| 9 | 1655 | 0.793 | 0.780 | +0.013 |
| 10 | 1655 | 0.832 | 0.816 | +0.016 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| air_le-1 | complete | +0.233 |
| yardline | complete | +0.231 |
| air_yards_sq | complete | -0.226 |
| yardline | incomplete | -0.215 |
| air_yards_c | incomplete | -0.209 |

Sanity read: completions fall and interceptions rise with air_yards_c

## yac

Rows: 87728 total, 10628 validation. Features: 73. Best C = 10.0.

Log loss on validation: model 1.8578, Phase 1 table 1.8693, constant 2.0449.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1063 | 0.171 | 0.196 | -0.025 |
| 2 | 1063 | 0.188 | 0.207 | -0.019 |
| 3 | 1063 | 0.210 | 0.224 | -0.014 |
| 4 | 1063 | 0.228 | 0.230 | -0.001 |
| 5 | 1063 | 0.249 | 0.252 | -0.003 |
| 6 | 1063 | 0.277 | 0.268 | +0.009 |
| 7 | 1063 | 0.308 | 0.283 | +0.024 |
| 8 | 1063 | 0.341 | 0.339 | +0.002 |
| 9 | 1062 | 0.387 | 0.397 | -0.010 |
| 10 | 1062 | 0.696 | 0.687 | +0.009 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| log_room | 1 | +4.897 |
| yardline | 1 | -3.156 |
| log_room | 0 | -2.246 |
| room_1 | 1 | +2.028 |
| room_le5 | 1 | +2.000 |

Sanity read: long YAC bins fall with air_yards_c (screens/short passes get YAC) and near the goal line

## rush_yards

Rows: 95754 total, 12191 validation. Features: 56. Best C = 0.1.

Log loss on validation: model 2.6607, Phase 1 table 2.6702, constant 2.7376.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1220 | 0.111 | 0.116 | -0.005 |
| 2 | 1219 | 0.118 | 0.142 | -0.024 |
| 3 | 1219 | 0.121 | 0.119 | +0.002 |
| 4 | 1219 | 0.123 | 0.123 | +0.000 |
| 5 | 1219 | 0.126 | 0.118 | +0.008 |
| 6 | 1219 | 0.129 | 0.113 | +0.016 |
| 7 | 1219 | 0.135 | 0.119 | +0.016 |
| 8 | 1219 | 0.149 | 0.135 | +0.015 |
| 9 | 1219 | 0.185 | 0.190 | -0.005 |
| 10 | 1219 | 0.348 | 0.343 | +0.005 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| log_ydstogo | 2 | -0.956 |
| yl_21_30 | 40 | -0.648 |
| log_ydstogo | 1 | +0.577 |
| red_zone | 25 | -0.553 |
| yl_1 | 2 | -0.514 |

Sanity read: long gains fall with red_zone/inside_5 and with short-yardage situations

## fumble_lost

Rows: 200202 total, 25080 validation. Features: 5. Best C = 10.0.

Log loss on validation: model 0.0431, Phase 1 table 0.0435, constant 0.0479.

Calibration (P(class 1)):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 2508 | 0.003 | 0.004 | -0.001 |
| 2 | 2508 | 0.004 | 0.004 | +0.000 |
| 3 | 2508 | 0.005 | 0.005 | +0.001 |
| 4 | 2508 | 0.006 | 0.002 | +0.004 |
| 5 | 2508 | 0.006 | 0.001 | +0.005 |
| 6 | 2508 | 0.007 | 0.003 | +0.004 |
| 7 | 2508 | 0.007 | 0.005 | +0.002 |
| 8 | 2508 | 0.008 | 0.004 | +0.004 |
| 9 | 2508 | 0.008 | 0.015 | -0.007 |
| 10 | 2508 | 0.038 | 0.038 | -0.001 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| yards_gained_c | yes | -0.472 |
| pt_sack | yes | +0.338 |
| pt_run | yes | -0.117 |
| pt_scramble | yes | -0.032 |
| pt_completion | yes | -0.015 |

Sanity read: sacks should carry the highest fumble-lost rate; completions the lowest

## fg_make

Rows: 7988 total, 1067 validation. Features: 6. Best C = 1.0.

Log loss on validation: model 0.3593, Phase 1 table 0.3615, constant 0.4148.

Calibration (P(class 1)):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 107 | 0.610 | 0.636 | -0.025 |
| 2 | 107 | 0.702 | 0.710 | -0.009 |
| 3 | 107 | 0.763 | 0.766 | -0.003 |
| 4 | 107 | 0.822 | 0.813 | +0.009 |
| 5 | 107 | 0.870 | 0.897 | -0.027 |
| 6 | 107 | 0.911 | 0.869 | +0.041 |
| 7 | 107 | 0.942 | 0.925 | +0.016 |
| 8 | 106 | 0.961 | 0.962 | -0.001 |
| 9 | 106 | 0.976 | 0.981 | -0.005 |
| 10 | 106 | 0.987 | 0.991 | -0.003 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| kick_distance | made | -1.504 |
| kick_distance_cu | made | +0.689 |
| kick_distance_sq | made | -0.397 |
| era | made | +0.136 |
| wind_c | made | -0.119 |

Sanity read: make probability falls with kick_distance and wind, rises with roof_dome

## pat_choice

Rows: 11482 total, 1460 validation. Features: 35. Best C = 10.0.

Log loss on validation: model 0.1611, Phase 1 table 0.2357, constant 0.3005.

Calibration (P(class 1)):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 146 | 0.002 | 0.000 | +0.002 |
| 2 | 146 | 0.004 | 0.000 | +0.004 |
| 3 | 146 | 0.006 | 0.014 | -0.008 |
| 4 | 146 | 0.009 | 0.014 | -0.005 |
| 5 | 146 | 0.013 | 0.021 | -0.007 |
| 6 | 146 | 0.024 | 0.027 | -0.004 |
| 7 | 146 | 0.037 | 0.034 | +0.003 |
| 8 | 146 | 0.064 | 0.021 | +0.043 |
| 9 | 146 | 0.169 | 0.171 | -0.002 |
| 10 | 146 | 0.627 | 0.589 | +0.038 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| second_half | two | +1.047 |
| sd_clip | two | -0.717 |
| sd_3 | two | -0.616 |
| sd_-14 | two | -0.542 |
| half_secs | two | -0.540 |

Sanity read: two-point tries peak at specific deficits after the TD (-2, -5, -10, +1, +4, +5) and late in the game

## clock_runoff

Rows: 301400 total, 37519 validation. Features: 36. Best C = 10.0.

Log loss on validation: model 1.3477, Phase 1 table 1.5177, constant 2.0570.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 3752 | 0.245 | 0.257 | -0.012 |
| 2 | 3752 | 0.327 | 0.334 | -0.007 |
| 3 | 3752 | 0.362 | 0.360 | +0.002 |
| 4 | 3752 | 0.388 | 0.378 | +0.010 |
| 5 | 3752 | 0.410 | 0.412 | -0.001 |
| 6 | 3752 | 0.435 | 0.427 | +0.008 |
| 7 | 3752 | 0.493 | 0.484 | +0.010 |
| 8 | 3752 | 0.579 | 0.572 | +0.007 |
| 9 | 3752 | 0.638 | 0.612 | +0.025 |
| 10 | 3751 | 0.802 | 0.804 | -0.001 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| clock_stopper | 8 | -2.520 |
| clock_stopper | 0 | +2.481 |
| clock_stopper | 7 | -2.263 |
| clock_stopper | 1 | +2.077 |
| clock_stopper | 9 | -1.463 |

Sanity read: long runoffs with leading_late; short with clock_stopper, under_2min_half and trailing_late

## presnap_penalty

Rows: 291662 total, 36491 validation. Features: 30. Best C = 0.1.

Log loss on validation: model 0.2738, Phase 1 table 0.2740, constant 0.2762.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 3650 | 0.906 | 0.902 | +0.004 |
| 2 | 3649 | 0.923 | 0.919 | +0.004 |
| 3 | 3649 | 0.934 | 0.933 | +0.002 |
| 4 | 3649 | 0.938 | 0.943 | -0.005 |
| 5 | 3649 | 0.941 | 0.937 | +0.005 |
| 6 | 3649 | 0.944 | 0.945 | -0.001 |
| 7 | 3649 | 0.946 | 0.942 | +0.004 |
| 8 | 3649 | 0.948 | 0.945 | +0.003 |
| 9 | 3649 | 0.952 | 0.953 | -0.001 |
| 10 | 3649 | 0.959 | 0.954 | +0.005 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| d2_logyd | none | -0.246 |
| d2_logyd | defense | +0.210 |
| down_3 | defense | +0.210 |
| log_ydstogo | defense | +0.172 |
| down_2 | none | +0.167 |

Sanity read: penalties rise with ydstogo (obvious passing downs) and are rarer on 4th down

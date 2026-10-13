# Phase 2 model report

Trained 2026-10-13 on seasons 2018-2025 (regular season + playoffs), validated on the last season with models fit on the earlier seasons. Lookup tables for the comparison were rebuilt from 2021 to the season before validation, so no validation play touches either competitor. Win-probability filter (2%-98%) applied except to the late-game models.

## Summary

| Model | Val rows | Log loss: model | table | constant | Beats table | Beats constant | Calibration (deciles ≥500 within 3 pts) | C |
|---|---|---|---|---|---|---|---|---|
| play_call | 33571 | 0.5604 | 0.5652 | 0.6730 | yes | yes | NO | 1.0 |
| fourth_down | 3997 | 0.2925 | 0.4227 | 1.0396 | yes | yes | yes | 1.0 |
| air_yards | 16556 | 1.7129 | 1.7180 | 1.8039 | yes | yes | yes | 0.1 |
| dropback_outcome | 18816 | 0.4475 | 0.4484 | 0.4513 | yes | yes | yes | 10.0 |
| throw_outcome | 16556 | 0.6881 | 0.6901 | 0.7345 | yes | yes | yes | 0.01 |
| yac | 10628 | 1.8582 | 1.8693 | 2.0449 | yes | yes | yes | 10.0 |
| rush_yards | 12191 | 2.6613 | 2.6702 | 2.7376 | yes | yes | yes | 0.1 |
| fumble_lost | 25080 | 0.0431 | 0.0435 | 0.0479 | yes | yes | yes | 10.0 |
| fg_make | 1067 | 0.3593 | 0.3615 | 0.4148 | yes | yes | yes | 1.0 |
| pat_choice | 1460 | 0.1611 | 0.2357 | 0.3005 | yes | yes | yes | 10.0 |
| clock_runoff | 37519 | 1.3514 | 1.5177 | 2.0570 | yes | yes | yes | 1.0 |
| presnap_penalty | 36491 | 0.2738 | 0.2740 | 0.2762 | yes | yes | yes | 0.1 |

Spec note: the spec's single `pass_outcome` model (5 classes) is implemented as `dropback_outcome` (throw / sack / scramble) followed by `throw_outcome` (complete / incomplete / interception given a throw), because sacks and scrambles have no air yards to condition on. The two multiply to the five-class distribution.

## play_call

Rows: 270393 total, 33571 validation. Features: 30. Best C = 1.0.

Log loss on validation: model 0.5604, Phase 1 table 0.5652, constant 0.6730.

Calibration (P(class 1)):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 3358 | 0.046 | 0.065 | -0.019 |
| 2 | 3357 | 0.110 | 0.107 | +0.004 |
| 3 | 3357 | 0.218 | 0.202 | +0.015 |
| 4 | 3357 | 0.300 | 0.299 | +0.000 |
| 5 | 3357 | 0.385 | 0.353 | +0.032 |
| 6 | 3357 | 0.469 | 0.484 | -0.015 |
| 7 | 3357 | 0.515 | 0.528 | -0.013 |
| 8 | 3357 | 0.553 | 0.547 | +0.006 |
| 9 | 3357 | 0.617 | 0.632 | -0.015 |
| 10 | 3357 | 0.784 | 0.777 | +0.007 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| log_ydstogo | run | -1.294 |
| down_3 | run | -1.070 |
| yardline | run | -0.966 |
| down_1 | run | +0.838 |
| yardline_sq | run | +0.811 |

Sanity read: passing should rise with ydstogo, trailing_late and down_3; fall with leading_late

## fourth_down

Rows: 31448 total, 3997 validation. Features: 23. Best C = 1.0.

Log loss on validation: model 0.2925, Phase 1 table 0.4227, constant 1.0396.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 400 | 0.540 | 0.522 | +0.018 |
| 2 | 400 | 0.680 | 0.730 | -0.050 |
| 3 | 400 | 0.788 | 0.795 | -0.007 |
| 4 | 400 | 0.872 | 0.863 | +0.009 |
| 5 | 400 | 0.925 | 0.943 | -0.017 |
| 6 | 400 | 0.958 | 0.975 | -0.017 |
| 7 | 400 | 0.977 | 0.980 | -0.003 |
| 8 | 399 | 0.988 | 0.980 | +0.008 |
| 9 | 399 | 0.995 | 0.997 | -0.003 |
| 10 | 399 | 0.998 | 0.995 | +0.003 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| yardline_sq | go | +5.037 |
| yardline_sq | field_goal | -4.455 |
| fg_distance_sq | field_goal | -3.329 |
| yardline | punt | +3.302 |
| fg_distance | punt | +3.302 |

Sanity read: go rate should fall with ydstogo; punt should rise with yardline (own territory); FG with short fg_distance

## air_yards

Rows: 135319 total, 16556 validation. Features: 53. Best C = 0.1.

Log loss on validation: model 1.7129, Phase 1 table 1.7180, constant 1.8039.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1656 | 0.220 | 0.237 | -0.017 |
| 2 | 1656 | 0.240 | 0.228 | +0.012 |
| 3 | 1656 | 0.245 | 0.256 | -0.011 |
| 4 | 1656 | 0.250 | 0.242 | +0.008 |
| 5 | 1656 | 0.255 | 0.243 | +0.011 |
| 6 | 1656 | 0.260 | 0.266 | -0.005 |
| 7 | 1655 | 0.270 | 0.259 | +0.011 |
| 8 | 1655 | 0.285 | 0.285 | -0.000 |
| 9 | 1655 | 0.310 | 0.305 | +0.005 |
| 10 | 1655 | 0.558 | 0.566 | -0.007 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| i_d3_logyd | 0 | +0.631 |
| down_3 | 0 | -0.615 |
| red_zone | 5 | -0.558 |
| i_d3_logyd | 2 | -0.546 |
| red_zone | 6 | -0.524 |

Sanity read: deep bins rise with ydstogo and trailing_late; fall near the goal line (yardline small)

## dropback_outcome

Rows: 152045 total, 18816 validation. Features: 20. Best C = 10.0.

Log loss on validation: model 0.4475, Phase 1 table 0.4484, constant 0.4513.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1882 | 0.831 | 0.830 | +0.000 |
| 2 | 1882 | 0.849 | 0.840 | +0.008 |
| 3 | 1882 | 0.872 | 0.872 | -0.000 |
| 4 | 1882 | 0.894 | 0.878 | +0.017 |
| 5 | 1882 | 0.900 | 0.882 | +0.019 |
| 6 | 1882 | 0.904 | 0.885 | +0.019 |
| 7 | 1881 | 0.907 | 0.907 | +0.000 |
| 8 | 1881 | 0.910 | 0.892 | +0.018 |
| 9 | 1881 | 0.915 | 0.897 | +0.017 |
| 10 | 1881 | 0.924 | 0.915 | +0.009 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| yardline | throw | -0.323 |
| yardline | sack | +0.315 |
| yardline_sq | throw | +0.289 |
| yardline_sq | sack | -0.243 |
| log_ydstogo | sack | +0.173 |

Sanity read: sacks and scrambles should rise with ydstogo / down_3 (long-developing plays)

## throw_outcome

Rows: 135319 total, 16556 validation. Features: 39. Best C = 0.01.

Log loss on validation: model 0.6881, Phase 1 table 0.6901, constant 0.7345.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1656 | 0.504 | 0.505 | -0.001 |
| 2 | 1656 | 0.546 | 0.549 | -0.003 |
| 3 | 1656 | 0.582 | 0.603 | -0.022 |
| 4 | 1656 | 0.619 | 0.641 | -0.022 |
| 5 | 1656 | 0.669 | 0.696 | -0.027 |
| 6 | 1656 | 0.708 | 0.718 | -0.010 |
| 7 | 1655 | 0.735 | 0.705 | +0.030 |
| 8 | 1655 | 0.764 | 0.772 | -0.008 |
| 9 | 1655 | 0.790 | 0.763 | +0.026 |
| 10 | 1655 | 0.829 | 0.807 | +0.022 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| air_le-1 | complete | +0.216 |
| air_yards_sq | complete | -0.199 |
| air_yards_c | incomplete | -0.196 |
| yardline | complete | +0.189 |
| yardline | incomplete | -0.189 |

Sanity read: completions fall and interceptions rise with air_yards_c

## yac

Rows: 87728 total, 10628 validation. Features: 70. Best C = 10.0.

Log loss on validation: model 1.8582, Phase 1 table 1.8693, constant 2.0449.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1063 | 0.171 | 0.195 | -0.024 |
| 2 | 1063 | 0.188 | 0.200 | -0.012 |
| 3 | 1063 | 0.210 | 0.226 | -0.016 |
| 4 | 1063 | 0.228 | 0.246 | -0.018 |
| 5 | 1063 | 0.249 | 0.242 | +0.008 |
| 6 | 1063 | 0.278 | 0.269 | +0.009 |
| 7 | 1063 | 0.308 | 0.280 | +0.027 |
| 8 | 1063 | 0.341 | 0.332 | +0.009 |
| 9 | 1062 | 0.387 | 0.401 | -0.015 |
| 10 | 1062 | 0.694 | 0.691 | +0.003 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| log_room | 1 | +4.952 |
| yardline | 1 | -3.187 |
| log_room | 0 | -2.241 |
| room_1 | 1 | +1.990 |
| room_le5 | 1 | +1.956 |

Sanity read: long YAC bins fall with air_yards_c (screens/short passes get YAC) and near the goal line

## rush_yards

Rows: 95754 total, 12191 validation. Features: 53. Best C = 0.1.

Log loss on validation: model 2.6613, Phase 1 table 2.6702, constant 2.7376.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 1220 | 0.112 | 0.110 | +0.002 |
| 2 | 1219 | 0.118 | 0.126 | -0.007 |
| 3 | 1219 | 0.121 | 0.130 | -0.010 |
| 4 | 1219 | 0.123 | 0.138 | -0.015 |
| 5 | 1219 | 0.125 | 0.117 | +0.008 |
| 6 | 1219 | 0.128 | 0.117 | +0.011 |
| 7 | 1219 | 0.134 | 0.119 | +0.015 |
| 8 | 1219 | 0.148 | 0.140 | +0.008 |
| 9 | 1219 | 0.184 | 0.185 | -0.002 |
| 10 | 1219 | 0.347 | 0.343 | +0.004 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| log_ydstogo | 2 | -0.953 |
| yl_21_30 | 40 | -0.652 |
| log_ydstogo | 1 | +0.575 |
| red_zone | 25 | -0.555 |
| yl_1 | 2 | -0.515 |

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

Rows: 301400 total, 37519 validation. Features: 35. Best C = 1.0.

Log loss on validation: model 1.3514, Phase 1 table 1.5177, constant 2.0570.

Calibration (probability of the modal class):

| Decile | n | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| 1 | 3752 | 0.244 | 0.262 | -0.018 |
| 2 | 3752 | 0.326 | 0.336 | -0.010 |
| 3 | 3752 | 0.360 | 0.365 | -0.005 |
| 4 | 3752 | 0.385 | 0.390 | -0.005 |
| 5 | 3752 | 0.407 | 0.410 | -0.002 |
| 6 | 3752 | 0.432 | 0.416 | +0.016 |
| 7 | 3752 | 0.488 | 0.476 | +0.012 |
| 8 | 3752 | 0.579 | 0.575 | +0.004 |
| 9 | 3752 | 0.638 | 0.611 | +0.026 |
| 10 | 3751 | 0.803 | 0.804 | -0.001 |

Largest standardized coefficients:

| Feature | Class | Coef |
|---|---|---|
| clock_stopper | 8 | -2.468 |
| clock_stopper | 0 | +2.452 |
| clock_stopper | 7 | -2.231 |
| clock_stopper | 1 | +2.067 |
| clock_stopper | 9 | -1.426 |

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

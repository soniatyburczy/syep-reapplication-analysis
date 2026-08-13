# Descriptive tables


## Cohort table, by index year

| Index year   |   N at risk |   N returned |   Return rate | 95% CI         |   Unique people |   Mean age |   % age 16-17 |   % age 21+ |   Median hours |   % at cap |   % zero hours |   % no show |   N worksites |   % at real worksite |   % cluster known |   % provider known |
|:-------------|------------:|-------------:|--------------:|:---------------|----------------:|-----------:|--------------:|------------:|---------------:|-----------:|---------------:|------------:|--------------:|---------------------:|------------------:|-------------------:|
| 2022         |       41757 |        23207 |         0.556 | [0.551, 0.561] |           41757 |      18.32 |         0.423 |       0.16  |          137   |      0.287 |          0.183 |       0.048 |          2658 |                0.565 |                 1 |              0.622 |
| 2023         |       43652 |        26004 |         0.596 | [0.591, 0.600] |           43652 |      18.35 |         0.419 |       0.167 |          136.2 |      0.236 |          0.179 |       0.065 |          4875 |                0.897 |                 1 |              1     |
| 2024         |       45008 |        28254 |         0.628 | [0.623, 0.632] |           45008 |      18.28 |         0.435 |       0.157 |          142   |      0.351 |          0.172 |       0.057 |          5214 |                0.904 |                 1 |              1     |
| Pooled       |      130417 |        77465 |         0.594 | [0.591, 0.597] |           97761 |      18.31 |         0.426 |       0.162 |          139   |      0.292 |          0.178 |       0.058 |          8441 |                0.793 |                 1 |              0.879 |


## Service option at t vs t+1, returners only

| service_option   |   Ladders for Leaders |   Older Youth |   Younger Youth |
|:-----------------|----------------------:|--------------:|----------------:|
| Older Youth      |                  6298 |         71162 |               5 |


## Risk set share and return rate by worksite status, by year

| Index year   | Worksite status   |      N |   % of risk set |   N returned |   Return rate |   ci_lo |   ci_hi | thin   |
|:-------------|:------------------|-------:|----------------:|-------------:|--------------:|--------:|--------:|:-------|
| 2022         | active            |  23599 |           0.565 |        14123 |         0.598 |   0.592 |   0.605 | False  |
| 2022         | exited            |     10 |           0     |            6 |         0.6   |   0.313 |   0.832 | True   |
| 2022         | no_show           |   1202 |           0.029 |          367 |         0.305 |   0.28  |   0.332 | False  |
| 2022         | test              |      7 |           0     |            2 |         0.286 |   0.082 |   0.641 | True   |
| 2022         | unplaced          |    275 |           0.007 |           85 |         0.309 |   0.257 |   0.366 | False  |
| 2022         | nan               |  16664 |           0.399 |         8624 |         0.518 |   0.51  |   0.525 | False  |
| 2023         | active            |  39167 |           0.897 |        24487 |         0.625 |   0.62  |   0.63  | False  |
| 2023         | exited            |      2 |           0     |            2 |         1     |   0.342 |   1     | True   |
| 2023         | no_show           |   2816 |           0.065 |          973 |         0.346 |   0.328 |   0.363 | False  |
| 2023         | test              |     30 |           0.001 |           16 |         0.533 |   0.361 |   0.698 | False  |
| 2023         | unplaced          |   1023 |           0.023 |          325 |         0.318 |   0.29  |   0.347 | False  |
| 2023         | nan               |    614 |           0.014 |          201 |         0.327 |   0.291 |   0.365 | False  |
| 2024         | active            |  40707 |           0.904 |        26673 |         0.655 |   0.651 |   0.66  | False  |
| 2024         | no_show           |   2478 |           0.055 |          934 |         0.377 |   0.358 |   0.396 | False  |
| 2024         | test              |     26 |           0.001 |           16 |         0.615 |   0.425 |   0.776 | True   |
| 2024         | unplaced          |    387 |           0.009 |          133 |         0.344 |   0.298 |   0.392 | False  |
| 2024         | nan               |   1410 |           0.031 |          498 |         0.353 |   0.329 |   0.379 | False  |
| Pooled       | active            | 103473 |           0.793 |        65283 |         0.631 |   0.628 |   0.634 | False  |
| Pooled       | exited            |     12 |           0     |            8 |         0.667 |   0.391 |   0.862 | True   |
| Pooled       | no_show           |   6496 |           0.05  |         2274 |         0.35  |   0.339 |   0.362 | False  |
| Pooled       | test              |     63 |           0     |           34 |         0.54  |   0.418 |   0.657 | False  |
| Pooled       | unplaced          |   1685 |           0.013 |          543 |         0.322 |   0.3   |   0.345 | False  |
| Pooled       | nan               |  18688 |           0.143 |         9323 |         0.499 |   0.492 |   0.506 | False  |


## Return rate by borough

| borough       |     n |   returned |   rate |   ci_lo |   ci_hi | thin   |
|:--------------|------:|-----------:|-------:|--------:|--------:|:-------|
| Brooklyn      | 57895 |      36360 |  0.628 |   0.624 |   0.632 | False  |
| Staten Island |  5179 |       2984 |  0.576 |   0.563 |   0.59  | False  |
| Manhattan     | 11402 |       6482 |  0.568 |   0.559 |   0.578 | False  |
| Bronx         | 27362 |      15491 |  0.566 |   0.56  |   0.572 | False  |
| Queens        | 28579 |      16148 |  0.565 |   0.559 |   0.571 | False  |


## Return rate by age_on_start

|   age_on_start |     n |   returned |   rate |   ci_lo |   ci_hi | thin   |
|---------------:|------:|-----------:|-------:|--------:|--------:|:-------|
|             16 | 27581 |      18774 |  0.681 |   0.675 |   0.686 | False  |
|             17 | 27966 |      17037 |  0.609 |   0.603 |   0.615 | False  |
|             18 | 22456 |      13178 |  0.587 |   0.58  |   0.593 | False  |
|             19 | 18106 |      10511 |  0.581 |   0.573 |   0.588 | False  |
|             20 | 13243 |       7175 |  0.542 |   0.533 |   0.55  | False  |
|             21 |  9512 |       4940 |  0.519 |   0.509 |   0.529 | False  |
|             23 |  4666 |       2365 |  0.507 |   0.493 |   0.521 | False  |
|             22 |  6887 |       3485 |  0.506 |   0.494 |   0.518 | False  |


## Return rate by hours_band

| hours_band   |     n |   returned |   rate |   ci_lo |   ci_hi | thin   |
|:-------------|------:|-----------:|-------:|--------:|--------:|:-------|
| 150 (cap)    | 38054 |      27314 |  0.718 |   0.713 |   0.722 | False  |
| 126-149      | 38902 |      25514 |  0.656 |   0.651 |   0.661 | False  |
| 76-125       | 20559 |      12667 |  0.616 |   0.609 |   0.623 | False  |
| 26-75        |  6241 |       3401 |  0.545 |   0.533 |   0.557 | False  |
| 1-25         |  3451 |       1326 |  0.384 |   0.368 |   0.401 | False  |
| 0            | 23210 |       7243 |  0.312 |   0.306 |   0.318 | False  |


## Return rate by provider

| provider                       |     n |   returned |   rate |   ci_lo |   ci_hi | thin   |
|:-------------------------------|------:|-----------:|-------:|--------:|--------:|:-------|
| Council of Jewish Organization | 19054 |      13732 |  0.721 |   0.714 |   0.727 | False  |
| Woodycrest Center For Human De |  1005 |        705 |  0.701 |   0.672 |   0.729 | False  |
| El Barrio's Operation Fightbac |  2034 |       1383 |  0.68  |   0.659 |   0.7   | False  |
| Infinity Educational Programs  |  4028 |       2703 |  0.671 |   0.656 |   0.685 | False  |
| Kips Bay                       |  1210 |        790 |  0.653 |   0.626 |   0.679 | False  |
| Hellenic American Neighborhood |  4835 |       3089 |  0.639 |   0.625 |   0.652 | False  |
| Simpson Street Development Ass |  1948 |       1227 |  0.63  |   0.608 |   0.651 | False  |
| Sesame Flyers International, I |   855 |        538 |  0.629 |   0.596 |   0.661 | False  |
| Brooklyn Neighborhood Improvem |  1841 |       1154 |  0.627 |   0.604 |   0.649 | False  |
| Research Foundation - Medgar E |  1514 |        948 |  0.626 |   0.601 |   0.65  | False  |
| NY Center for Interper- NYCID  |   709 |        438 |  0.618 |   0.581 |   0.653 | False  |
| Catholic Charities Neighborhoo |  2912 |       1780 |  0.611 |   0.593 |   0.629 | False  |
| Community Counseling & Mediati |  2368 |       1440 |  0.608 |   0.588 |   0.628 | False  |
| Chinese American Planning Coun |  6711 |       4004 |  0.597 |   0.585 |   0.608 | False  |
| Edith and Carl Marks Jewish Co |  2122 |       1254 |  0.591 |   0.57  |   0.612 | False  |
| RiseBoro Community Partnership |  1489 |        876 |  0.588 |   0.563 |   0.613 | False  |
| United Activities Unlimited    |  2387 |       1397 |  0.585 |   0.565 |   0.605 | False  |
| Center for Family Life         |  2372 |       1385 |  0.584 |   0.564 |   0.604 | False  |
| CAMBA                          |  1265 |        735 |  0.581 |   0.554 |   0.608 | False  |
| CCCS                           |  3042 |       1744 |  0.573 |   0.556 |   0.591 | False  |
| YM-YWHA                        |  1667 |        954 |  0.572 |   0.548 |   0.596 | False  |
| St. Nicks Alliance Corporation |  1598 |        911 |  0.57  |   0.546 |   0.594 | False  |
| Commonpoint                    |  4854 |       2752 |  0.567 |   0.553 |   0.581 | False  |
| Roads                          |   624 |        351 |  0.562 |   0.523 |   0.601 | False  |
| Chinatown Manpower Project, In |  2228 |       1250 |  0.561 |   0.54  |   0.582 | False  |
| Rockaway Development & Revital |  1556 |        872 |  0.56  |   0.536 |   0.585 | False  |
| Inwood Community Services, Inc |  1879 |       1053 |  0.56  |   0.538 |   0.583 | False  |
| Research Foundation - LaGuardi |  2827 |       1581 |  0.559 |   0.541 |   0.577 | False  |
| Aspira of New York             |  1718 |        958 |  0.558 |   0.534 |   0.581 | False  |
| Madison Square                 |   464 |        259 |  0.558 |   0.513 |   0.603 | False  |
| Henry Street Settlement, Inc.  |  2740 |       1522 |  0.555 |   0.537 |   0.574 | False  |
| Bridge Street                  |  2640 |       1457 |  0.552 |   0.533 |   0.571 | False  |
| The Children's Aid Society     |  2025 |       1118 |  0.552 |   0.53  |   0.574 | False  |
| Queens Community House, Inc.   |  1822 |        987 |  0.542 |   0.519 |   0.564 | False  |
| Community Association of Progr |  2126 |       1150 |  0.541 |   0.52  |   0.562 | False  |
| BronxWorks, Inc                |  1191 |        641 |  0.538 |   0.51  |   0.566 | False  |
| Mosholu Montefiore Community C |  4065 |       2184 |  0.537 |   0.522 |   0.553 | False  |
| Cypress Hills Local Developmen |  1643 |        883 |  0.537 |   0.513 |   0.561 | False  |
| Children's Arts & Science Work |  1149 |        614 |  0.534 |   0.505 |   0.563 | False  |
| Police Athletic League         |  2265 |       1210 |  0.534 |   0.514 |   0.555 | False  |
| Greater Ridgewood Youth Counci |  2806 |       1488 |  0.53  |   0.512 |   0.549 | False  |
| SOBRO                          |   955 |        503 |  0.527 |   0.495 |   0.558 | False  |
| nan                            | 15792 |       8321 |  0.527 |   0.519 |   0.535 | False  |
| Italian American Civil Rights  |  3868 |       2031 |  0.525 |   0.509 |   0.541 | False  |
| Phipps Neighborhoods           |   637 |        316 |  0.496 |   0.457 |   0.535 | False  |
| BCS                            |   919 |        453 |  0.493 |   0.461 |   0.525 | False  |
| Boys & Girls Club Queens       |   658 |        324 |  0.492 |   0.454 |   0.531 | False  |


## Return rate by is_no_show

|   is_no_show |      n |   returned |   rate |   ci_lo |   ci_hi | thin   |
|-------------:|-------:|-----------:|-------:|--------:|--------:|:-------|
|            0 | 105233 |      65868 |  0.626 |   0.623 |   0.629 | False  |
|          nan |  18688 |       9323 |  0.499 |   0.492 |   0.506 | False  |
|            1 |   6496 |       2274 |  0.35  |   0.339 |   0.362 | False  |

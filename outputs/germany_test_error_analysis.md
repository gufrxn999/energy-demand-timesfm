# Germany test error analysis

The Germany adapter lowered daily-window MAE on 81.8% of the 274 test days versus weekly seasonal naive, 85.8% versus zero-shot TimesFM, and 74.8% versus the unchanged PJME adapter. This indicates its aggregate gain is not confined to a handful of days.

The Germany adapter's median daily MAE was 784.0 MW and its 90th-percentile daily MAE was 1,771.0 MW. Its five highest-error days contributed 7.9% of its summed daily MAE. Highest-error days were:

| UTC date | Daily MAE (MW) |
|---|---:|
| 2020-01-09 | 5,026.6 |
| 2020-06-11 | 4,874.1 |
| 2020-01-02 | 4,384.7 |

The Germany adapter's worst forecast lead was hour 13, with 1,344.7 MW MAE. The largest daily regressions versus weekly naive were 2020-05-04 (+2,934.7 MW), 2020-04-14 (+1,338.7 MW), 2020-06-22 (+970.6 MW), 2020-05-12 (+902.3 MW), and 2020-03-29 (+835.1 MW). It still improves overall despite these days.

The test evaluation was performed once after checkpoint selection; this error review is descriptive and did not change the model.

# Termination reasons

2,576 stopped trials with a stated reason.

## Rule-based categories

| reason     |   share |
|:-----------|--------:|
| enrollment |   0.309 |
| other      |   0.28  |
| funding    |   0.096 |
| business   |   0.088 |
| covid      |   0.058 |
| logistics  |   0.056 |
| efficacy   |   0.054 |
| safety     |   0.031 |
| regulatory |   0.03  |

## Topics (TF-IDF + k-means)

|   cluster |   size | top_terms                                                                                                           |
|----------:|-------:|:--------------------------------------------------------------------------------------------------------------------|
|         0 |    904 | pi, participants, lack, left, investigator, institution, enrolled, subjects                                         |
|         3 |    482 | study, terminated, study terminated, sponsor, withdrawn, early, stopped, decided                                    |
|         4 |    242 | trial, safety, development, clinical, did, concerns, study, primary                                                 |
|         7 |    179 | recruitment, slow recruitment, slow, low recruitment, low, insufficient recruitment, insufficient, lack recruitment |
|         2 |    164 | enrollment, low enrollment, lack enrollment, low, slow enrollment, lack, slow, poor enrollment                      |
|         1 |    141 | funding, lack funding, lack, insufficient funding, funding issues, issues, insufficient, loss funding               |
|         6 |    136 | decision, sponsor decision, sponsor, business decision, business, safety, decision terminate, study                 |
|         8 |    132 | accrual, slow accrual, slow, low accrual, low, poor accrual, poor, lack accrual                                     |
|         9 |    108 | patients, patients enrolled, unable, recruit, enrolled, study, recruit patients, number                             |
|         5 |     88 | covid, covid 19, 19, pandemic, 19 pandemic, study, 19 related, enrollment                                           |

## Topic x category

|   topic |   business |   covid |   efficacy |   enrollment |   funding |   logistics |   other |   regulatory |   safety |
|--------:|-----------:|--------:|-----------:|-------------:|----------:|------------:|--------:|-------------:|---------:|
|       0 |         60 |      32 |         52 |          152 |        62 |         118 |     365 |           40 |       23 |
|       1 |          0 |       0 |          0 |            4 |       137 |           0 |       0 |            0 |        0 |
|       2 |          0 |       6 |          3 |          154 |         0 |           0 |       0 |            0 |        1 |
|       3 |         28 |      14 |         25 |           95 |        41 |          21 |     217 |           20 |       21 |
|       4 |         34 |       2 |         50 |           22 |         4 |           3 |      86 |           12 |       29 |
|       5 |          0 |      88 |          0 |            0 |         0 |           0 |       0 |            0 |        0 |
|       6 |        104 |       0 |          1 |            9 |         2 |           0 |      19 |            0 |        1 |
|       7 |          0 |       5 |          3 |          171 |         0 |           0 |       0 |            0 |        0 |
|       8 |          0 |       0 |          3 |          127 |         0 |           0 |       0 |            0 |        2 |
|       9 |          0 |       3 |          1 |           61 |         1 |           2 |      34 |            4 |        2 |

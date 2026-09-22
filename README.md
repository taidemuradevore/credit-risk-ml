# Predicting loan defaults

Can you tell from a loan application whether someone will pay the loan back? This project tries to answer that using about 32,000 consumer loans, then turns the predictions into an approve-or-decline rule based on what each kind of mistake costs.

The data is the [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) from Kaggle: 32,581 loans with the borrower's age, income, home ownership, employment length, loan amount and purpose, credit history, and whether the loan defaulted. About 22% did.

## What I did

I removed 165 exact duplicate rows and blanked out a handful of impossible values, like ages over 100 and 123 years of employment. Missing values stayed in until modelling, since anything used to fill them has to be learned from the training data only.

I kept 80% of the loans for training and held back 20% for testing, with the same default rate in both. There's no date column, so I couldn't train on older loans and test on newer ones, and a random split was the best option.

Loan grade and interest rate are set by the lender after it has already assessed the borrower, so a model that uses them is partly copying the lender's existing model. I left them out and measured separately how much they'd add.

For models, I used logistic regression as a simple baseline, then gradient boosted trees, which build many small decision trees that each correct the mistakes of the ones before. The tree model's settings were picked using the training data only.

Giving defaults extra weight during training is the usual fix for imbalanced data. I tried it, but it didn't improve the scores and it pushed the predicted probabilities too high, so I left it off and handled the imbalance through the costs instead.

## Results

Both scores are measured on the held-back test loans. AUC is how often the model ranks a loan that defaulted above one that didn't, where 0.5 is a coin flip. PR-AUC is how well it picks out defaults without flagging good loans, and random guessing scores about 0.22 here. I don't report accuracy, because calling every loan safe would already be 78% accurate.

| Model | AUC | PR-AUC |
|---|---|---|
| Logistic regression (baseline) | 0.82 | 0.65 |
| Gradient boosted trees | 0.90 | 0.82 |
| Gradient boosted trees + lender's grade and rate | 0.95 | 0.91 |

Adding the lender's two columns raises PR-AUC by 0.09. That's a big jump, and most of it comes from the lender's own assessment of the borrower rather than anything in the application.

The strongest signals are home ownership (renters default more than owners), a previous default on file, how big the loan is relative to income, income itself, and what the loan is for. A blank employment history is also a warning sign: those applicants defaulted 31.5% of the time, against 21.5% for everyone else.

## Choosing a cutoff

The model gives each applicant a probability of default. The usual cutoff of 50% assumes that approving a loan that defaults costs the same as turning away someone who would have paid. It doesn't. I assumed a default loses 45% of the loan amount and a turned-away customer loses 5% in profit, which puts the cutoff at 10%.

| Rule | Applicants declined | Total cost on test loans |
|---|---|---|
| Decline above 10% risk | 48.5% | $1.41M |
| Decline above 50% risk | 13.4% | $2.61M |
| Approve everyone | 0% | $6.85M |

The 10% cutoff costs $1.2M less than 50%, and it comes out ahead under every combination of loss and profit assumptions I tried. The downside is that it declines nearly half of applicants, and that share ranges from 19% to 86% depending on the assumptions, especially the profit margin.

## Limitations

- The loss and profit figures are my assumptions. The data has nothing on recoveries or loan terms, and a real lender would use its own numbers, probably different for each type of loan.
- With no dates, I can't check how the model holds up on future applicants, which is what a lender actually cares about.
- Costs are based on the original loan amount, which overstates the loss on loans that default after being mostly paid off.
- 740 rows list more years of employment than seems possible for the person's age. I kept them, since most are only slightly off.
- Declining half of applicants may not be acceptable to a real business, whatever the numbers say.

## Running it

Python 3.12.

```
pip install -r requirements.txt
jupyter lab
```

Run `01_eda_and_cleaning.ipynb` first. It writes `credit_risk_clean.csv`, which `02_modelling.ipynb` reads. The modelling notebook also saves the test predictions to `preds.npz`.

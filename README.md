# Phishing URL Checker

A small Python tool that scores URLs for phishing risk using lexical features and an
interpretable logistic regression model (scikit-learn). It never fetches the URL.

![Demo output](checker-demo.png)

## How it works
- `URLFeatureExtractor` takes hostname features: length, dots, hyphens, IP host, subdomain
  depth, credential keywords (login, verify, secure, account, update...), Shannon entropy,
  risky TLD, digit ratio, and known URL shorteners.
- `PhishingClassifier` wraps `StandardScaler` + `LogisticRegression`, reports real test
  metrics, shows feature weights, and explains each score by its top per-feature contributions.
- Rule flags on the full URL (`@` in URL, no HTTPS, keyword in path) are printed as extra
  reasons. They don't feed into the model (see Limitations for why).

## Run
```bash
pip install -r requirements.txt
# dataset (see below) -> data/PhiUSIIL_Phishing_URL_Dataset.csv
python train.py                      # trains, writes model.joblib + metrics.json
python checker.py "hxxp://paypal-account-verify[.]secure-login[.]tk/signin"
python checker.py --weights
```
Defanged input (`hxxp`, `[.]`) is accepted.

## Dataset
PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository, CC BY 4.0):
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
It has 235,795 URLs: 134,850 legitimate and 100,945 phishing.
Prasad, A. & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL
detection framework. Computers & Security.

## Results (80/20 stratified split, seed 42, 47,159 test URLs)
| Metric | Value |
|---|---|
| Accuracy | 79.3% |
| Precision (phishing) | 85.6% |
| Recall (phishing) | 62.1% |

Confusion matrix (rows = actual, cols = predicted):

|  | pred legit | pred phish |
|---|---|---|
| legit | 24,865 | 2,105 |
| phish | 7,658 | 12,531 |

![Feature weights](features.png)

## Limitations
- In this dataset every legitimate URL is an `https://` homepage with no path. My first
  version also used full-URL length, path length and HTTPS, and it scored 99.6%. It wasn't
  detecting phishing, it was learning "has a path or uses http". I limited the model to
  hostname features, so 79% is the more honest number.
- It's a simple lexical model. It can't see page content, certificates, domain age or
  reputation, and it misses phishing on clean-looking or compromised domains. Recall is 62%.
- It's a learning project, not a production security control.

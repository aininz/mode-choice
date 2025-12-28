# PyTorch Implementation of Nested Logit Mode Choice Model with Heterogenous Features + Dynamic Pricing Tools

Live on [Streamlit](https://mode-choice-zmx3ayg3h2forofenappzjp.streamlit.app/).

A nested logit discrete choice model is implemented in PyTorch and deployed as a Streamlit app (containerized with Docker). The public **ModeCanada** dataset is used to model travel mode choice across **train, car, bus, air** using a **two-level nest structure (Land vs. Air)** where **Land = {train, car, bus}** and **Air = {air}**. Heterogeneous effects introduced via **income** and **urban** features. The project is framed as a hypothetical travel-agency e-ticketing case study, where **train recall** is prioritized to reduce missed public transport demand signals.

## Dataset

The dataset is based on the `ModeCanada` example commonly used in discrete choice modeling:

- Source documentation: ModeCanada dataset (mlogit)  
  https://rdrr.io/rforge/mlogit/man/ModeCanada.html

Each observation represents a choice occasion with:
- **Alternative-specific attributes** (vary by mode): `cost`, `ivt` (in-vehicle time), `ovt` (out-of-vehicle time), `freq` (frequency)
- **Individual-specific attributes** (vary by user): `income`, `urban` (indicator)

## Model overview

A **nested logit** model is used to relax the IIA assumption by allowing correlated unobserved utility within a nest (substitution patterns are captured more realistically than in plain multinomial logit).

Utilities are computed as:

- `U_ni = V_ni + ε_ni`
- `P(i|n) = P(i|m,n) * P(m|n)` (nested logit factorization)
- `V_ni` is formed from base alternative features and engineered features, with heterogeneity introduced, including via alternative-specific covariates (e.g., `income_train`) and interactions (e.g., `urban_x_ovt`).

Choice probabilities are decomposed into:
- probability of choosing a nest
- probability of choosing an alternative conditional on the nest (via inclusive value / log-sum)

PyTorch is used to compute utilities and nested probabilities, and parameters are optimized by minimizing negative log-likelihood (NLL) with model selection guided by validation performance.

## Feature engineering

A base feature set is used:

- `cost`, `ivt`, `ovt`, `freq`

Individual attributes are also included via **alternative-specific coding** (mode-specific covariates):

- `income_train`, `income_car`, `income_bus`, `income_air`
- `urban_train`, `urban_car`, `urban_bus`, `urban_air`

Candidate engineered features are evaluated via forward selection:

- `cost_log`
- `wait_time`
- `gen_time`
- `rel_cost`
- `rel_gen_time`
- `income_x_rel_cost`
- `urban_x_ovt`

Feature selection is performed with a two-objective routine where validation NLL is improved while **train recall** is treated as a priority metric for the case study.

## Training setup

- Stratified sampling is used to preserve class proportions across train/validations/test splits.
- Optimization is performed in PyTorch.
- Model selection is guided by validation NLL.

## Result summary

Table below shows the overall metrics of train/val/test:

| Split | Per-case NLL | Accuracy |
|:-----|-------------:|---------:|
| Train | 0.6193 | 0.7601 |
| Val   | 0.6295 | 0.7612 |
| Test  | 0.5967 | 0.7673 |

Mode shares (pred vs. actual) are as below:

| Split | Type   | Train | Car   | Bus  | Air   |
|:------|:-------|------:|------:|-----:|------:|
| Train | Pred   | 14.37% | 51.24% | 0.39% | 34.00% |
| Train | Actual | 14.41% | 51.19% | 0.36% | 34.04% |
| Train | Delta      | -0.04% | +0.05% | +0.02% | -0.03% |
| Val   | Pred   | 15.09% | 49.35% | 0.47% | 35.09% |
| Val   | Actual | 14.33% | 51.16% | 0.46% | 34.05% |
| Val   | Delta      | +0.76% | -1.81% | +0.01% | +1.04% |
| Test  | Pred   | 14.10% | 50.20% | 0.38% | 35.32% |
| Test  | Actual | 14.48% | 51.16% | 0.31% | 34.05% |
| Test  | Delta      | -0.39% | -0.95% | +0.07% | +1.27% |

Across train/validation/test:
- Accuracy is stable around 0.76~0.77.
- Predicted shares are close to observed shares (aggregate demand is reproduced reasonably well).

On the test set:
- **ROC AUC:** 0.8591 (strong)
- **ROC AUC (weighted):** 0.8908 (strong)
- **PR AUC:** 0.5439 (moderate)

Table below shows the confusion matrix:
| True \ Pred | pred_train | pred_car | pred_bus | pred_air |
|:-----------|-----------:|---------:|---------:|---------:|
| true_train | 9  | 60 | 0 | 25 |
| true_car   | 5  | 294 | 0 | 33 |
| true_bus   | 0  | 2  | 0 | 0 |
| true_air   | 3  | 23 | 0 | 195 |

**Reading note:** most true *train* cases are predicted as *car* (60) or *air* (25), which explains the low train recall.

While table below shows the classification report:

| Class | Precision | Recall | F1 | Support |
|:------|----------:|-------:|---:|--------:|
| train | 0.5294 | 0.0957 | 0.1622 | 94 |
| car   | 0.7757 | 0.8855 | 0.8270 | 332 |
| bus   | 0.0000 | 0.0000 | 0.0000 | 2 |
| air   | 0.7708 | 0.8824 | 0.8228 | 221 |
| **accuracy** |  |  | **0.7673** | **649** |
| **macro avg** | 0.5190 | 0.4659 | 0.4530 | 649 |
| **weighted avg** | 0.7360 | 0.7673 | 0.7267 | 649 |

**Note:** `bus` has support = 2, so bus metrics are not statistically meaningful.

Class-wise performance is still uneven:
- **car** and **air** show strong recall (around 0.88).
- **train** recall is low (around 0.10) with many true train cases predicted as **car** or **air**.
- **bus** has extremely low support (~2 samples) so class-specific bus metrics are not reliable.

This is consistent with Nested Logit model that matches aggregate shares but struggles to separate train at the individual decision level under the current signal and objective.

## Price Response (Demand Elasticity) and Pricing Optimization

Price response is analyzed via counterfactuals. The `cost` feature for a selected mode is multiplied by a factor (e.g., +1%), choice probabilities are recomputed, and expected aggregate demand is obtained by summing probabilities across observations. Own-price elasticity is estimated using a finite-difference approximation, and substitution curves are generated by sweeping a multiplier grid and plotting relative demand by mode.

Pricing is optimized with **gradient-based expected revenue maximization (PyTorch autograd + Adam)** using multiplicative price factors. In the scenario-based optimizer, demand can be softly capped with a differentiable capacity constraint to approximate inventory limits.

## Streamlit App

A Streamlit interface is provided for interactive inference and policy evaluation. The app is intended to demonstrate how a choice model can support **dynamic pricing optimization** where candidate price changes are simulated and compared using predicted demand and substitution effects before deployment.

### Running locally

1. Create and activate a virtual environment

Windows (Powershell)
```sh
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies as specified in `pyproject.toml`
```sh
pip install .
```

3. Run Streamlit
```sh
streamlit run app/Home.py
```

### Running in Docker

1. Create the backend network
```sh
docker network create backend
```
2. Create `.env` file and copy the content from `.env.example`. Adjust accordingly.
3. Build the docker image
```sh
docker compose build
```
4. Run the docker container
```sh
docker compose up -d
```

To stop the container, run:
```sh
docker compose down
```

#### NGINX Configuration
Below is the NGINX configuration to forward request to this application.
```nginx
map $http_host $proxy_host {
    '' $host;
    default $http_host;
}

map $http_x_forwarded_proto $proxy_x_forwarded_proto {
    '' $scheme;
    default $http_x_forwarded_proto;
}

map $http_x_forwarded_scheme $proxy_x_forwarded_scheme {
    '' $scheme;
    default $http_x_forwarded_scheme;
}

map $http_x_forwarded_for $proxy_x_forwarded_for {
    '' $proxy_add_x_forwarded_for;
    default $http_x_forwarded_for;
}

map $http_x_real_ip $proxy_x_real_ip {
    '' $remote_addr;
    default $http_x_real_ip;
}

map $http_upgrade $connection_upgrade {
    ''      $http_connection;
    default "upgrade";
}

server {
    listen       443 ssl;
    listen  [::]:443 ssl;
    server_tokens off;

    server_name <server_name>;

    ssl_certificate     <path_to_cert>;
    ssl_certificate_key <path_to_key>;

    location / {
        # Additional response headers
        add_header  Strict-Transport-Security   "max-age=31536000; includeSubDomains; preload";

        # Request headers passed to the origin.
        proxy_set_header Host               $proxy_host;
        proxy_set_header X-Forwarded-Scheme $proxy_x_forwarded_scheme;
        proxy_set_header X-Forwarded-Proto  $proxy_x_forwarded_proto;
        proxy_set_header X-Forwarded-For    $proxy_x_forwarded_for;
        proxy_set_header X-Real-IP          $proxy_x_real_ip;

        # WebSocket support
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        proxy_http_version 1.1;

        proxy_pass http://mode-choice:8501;
    }
}
```

**Note**
- Replace `<server_name>` with the host name you want to use.
- Replace `<path_to_cert>` with the actual path to the SSL certificate file.
- Replace `<path_to_key>` with the actual path to the SSL private key file.

## Recommendations

Several changes are likely to improve the model's performance (including train recall) more effectively aside from additional feature tweaks:
1. Cost-sensitive training where a heavier penalty can be assigned to train false negatives (class-weighted NLL).
2. Nest definition can be changed (e.g., from **Land vs. Air** to **public transport vs. private transport (car)**).
3. With extremely low **bus** support, training noise might have been introduced, so bus can be merged into an "other" bucket or additional bus samples can be added.
4. Include **nonlinear time-sensitivity** (log or piecewise) as an attribute.
5. If stronger heterogeneity is needed, a **mixed logit model** (random coefficients) can be an upgrade. 
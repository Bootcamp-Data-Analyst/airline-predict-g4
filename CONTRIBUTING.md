#  Commit Nomenclature Guide — airline-predict-g4

## For the entire team: Mariana, Rocío Lozano, Rocío López, Thami

This document defines the commit conventions for our project. Following these rules ensures a clean, readable, and professional Git history.

-----

## Format

Every commit message ought to follow this structure:

```
type(scope): short description
```

**Rules:**

- **type**: What kind of change (see table below)
- **scope**: *(optional)* Which part of the project is affected
- **short description**: What you did, in imperative mood (“add”, not “added” or “adding”)
- Maximum 72 characters total
- All in English, lowercase (except proper nouns)
- No period at the end

-----

## Types

|Type      |When to use                                         |Example                                                |
|----------|----------------------------------------------------|-------------------------------------------------------|
|`feat`    |New feature or functionality                        |`feat(pipeline): add preprocessing function`           |
|`fix`     |Bug fix                                             |`fix(preprocess): handle nulls in Arrival Delay column`|
|`refactor`|Code restructuring (no new feature, no bug fix)     |`refactor(train): simplify train_test_split logic`     |
|`docs`    |Documentation only (README, comments, guides)       |`docs: update README with setup instructions`          |
|`test`    |Adding or modifying tests                           |`test(pipeline): add preprocessing unit tests`         |
|`data`    |Changes to datasets or data files                   |`data: add reduced airlines sample csv`                |
|`style`   |Formatting, whitespace, naming (no logic change)    |`style(preprocess): rename variables for clarity`      |
|`chore`   |Maintenance tasks (dependencies, configs, gitignore)|`chore: update requirements.txt`                       |
|`build`   |Docker, CI/CD, build system changes                 |`build(docker): add Dockerfile for streamlit app`      |

-----

## Scopes (optional but recommended)

|Scope       |Applies to                          |
|------------|------------------------------------|
|`pipeline`  |Anything in `src/pipeline/`         |
|`preprocess`|`preprocess.py` specifically        |
|`train`     |Model training (`train_model.py`)   |
|`predict`   |Prediction logic (`predict.py`)     |
|`eval`      |Evaluation metrics (`evaluation.py`)|
|`persist`   |Serialization (`persistence.py`)    |
|`logging`   |Logging system (`logging.py`)       |
|`feedback`  |Feedback system (`feedback.py`)     |
|`monitoring`|Drift detection (`monitoring.py`)   |
|`app`       |Streamlit app (`app.py`)            |
|`eda`       |Notebooks and EDA                   |
|`docker`    |Docker files                        |
|`tests`     |Test files                          |

-----

## Good vs Bad Examples

###  Good commits

```
feat(preprocess): add clean_data function with null handling
feat(preprocess): add OrdinalEncoder for Class column
feat(train): implement RandomForest training with stratified split
feat(predict): add predict and predict_proba functions
feat(persist): add save/load functions for model and encoders
feat(logging): add prediction logging to CSV
feat(feedback): add feedback storage for retraining
feat(monitoring): add basic drift detection
feat(pipeline): add run_pipeline script to execute full workflow
test(pipeline): add unit tests for preprocessing and prediction
test(pipeline): add integration test for single input prediction
docs: add commit nomenclature guide
docs: update README with execution instructions
fix(preprocess): fix OrdinalEncoder transform for single input
refactor(preprocess): extract constants to module level
data: add reduced airlines sample for quick testing
chore: add joblib and pytest to requirements.txt
build(docker): configure Dockerfile for streamlit deployment
```

###  Bad commits

```
update files                     → Too vague. What files? What update?
fixed stuff                      → What stuff? What was broken?
WIP                              → Never commit "work in progress"
asdfasdf                         → Not descriptive at all
Edición del módulo preprocess    → Wrong language, no type prefix
changed preprocess.py            → No type, not descriptive enough
FEAT: ADD MODEL                  → Don't use CAPS
feat: Add preprocessing function, train model, and save it  → Too many things in one commit
```

-----

## Guidelines for Better Commits

### 1. One logical change per commit

```
#  Good: separate commits for separate changes
feat(preprocess): add clean_data function with null handling
feat(preprocess): add encode_features with OrdinalEncoder for Class

#  Bad: everything in one giant commit
feat: add all preprocessing, training, and evaluation
```

### 2. Commit after each working step

The ideal workflow:

1. Write the code for one function or feature
1. Test that it works (`python -m src.pipeline.preprocess`)
1. `git add .`
1. `git commit -m "feat(preprocess): add clean_data function"`
1. Repeat for the next feature

### 3. Use imperative mood

Write as if completing the sentence: *“This commit will…”*

```
#  Imperative (correct)
feat(train): add stratified train_test_split
fix(predict): handle empty dataframe input

#  Past tense (incorrect)
feat(train): added stratified train_test_split
fix(predict): handled empty dataframe input
```

### 4. When to use multi-line commit messages

For complex changes, add a body after a blank line:

```bash
git commit -m "feat(preprocess): add preprocess_single_input for production

This function ensures consistency between training and production.
It uses the same encoders saved during training to transform
individual user inputs from the Streamlit form."
```



## Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│             COMMIT FORMAT                            │
│                                                      │
│   type(scope): short description                     │
│                                                      │
│   TYPES:                                             │
│   feat     → new feature                             │
│   fix      → bug fix                                 │
│   refactor → restructure code                        │
│   docs     → documentation                           │
│   test     → add/modify tests                        │
│   data     → dataset changes                         │
│   style    → formatting only                         │
│   chore    → maintenance                             │
│   build    → docker, CI/CD                           │
│                                                      │
│   RULES:                                             │
│   ✓ English, lowercase                               │
│   ✓ Imperative mood ("add", not "added")             │
│   ✓ Max 72 characters                                │
│   ✓ No period at the end                             │
│   ✓ One logical change per commit                    │
│                                                      │
│   EXAMPLE:                                           │
│   feat(preprocess): add OrdinalEncoder for Class     │
└─────────────────────────────────────────────────────┘
```
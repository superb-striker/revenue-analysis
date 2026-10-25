import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

def two_proportion_ztest(df: pd.DataFrame, group_col: str, outcome_col: str,
                          group_true_label=1, group_false_label=0):
    group_true = df[df[group_col] == group_true_label]
    group_false = df[df[group_col] == group_false_label]

    count = [group_true[outcome_col].sum(), group_false[outcome_col].sum()]
    nobs = [len(group_true), len(group_false)]

    z_stat, p_value = proportions_ztest(count, nobs)

    return {
        "group_col": group_col,
        "group_true_rate": count[0] / nobs[0] if nobs[0] else float("nan"),
        "group_false_rate": count[1] / nobs[1] if nobs[1] else float("nan"),
        "group_true_n": nobs[0],
        "group_false_n": nobs[1],
        "z_stat": z_stat,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }

def chi_square_segment_test(df: pd.DataFrame, segment_col: str, outcome_col: str):
    contingency = pd.crosstab(df[segment_col], df[outcome_col])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    return {
        "segment_col": segment_col,
        "chi2_stat": chi2,
        "p_value": p_value,
        "dof": dof,
        "contingency_table": contingency,
        "significant_at_0.05": p_value < 0.05,
    }

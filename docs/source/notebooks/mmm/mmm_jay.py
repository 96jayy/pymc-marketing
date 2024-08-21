import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from pymc_marketing.mmm.delayed_saturated_mmm import MMM
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation



def plot_media_costs(df, media_columns, date_column):    #df : 데이터프레임, media_columns : 미디어 데이터 컬럼 리스트, date_column : 날짜 데이터
    n_rows = len(media_columns)
    
    fig, ax = plt.subplots(
        nrows=n_rows, ncols=1, figsize=(10, 3 * n_rows), sharex=True, sharey=True, layout="constrained"
    )
    
    if n_rows == 1:
        ax = [ax]
    
    for i, col in enumerate(media_columns):
        sns.lineplot(x=date_column, y=col, data=df, ax=ax[i], color=f"C{i}")
        ax[i].set_title(f"{col} contribution")
    
    ax[-1].set(xlabel="date")
    fig.suptitle("Media Costs Data", fontsize=16)
    plt.show()



def apply_adstock(df, columns, alphas, l_maxs, normalize=True):
    """
    여러 열에 대해 기하학적 애드스톡 변환을 적용하고, 결과를 데이터프레임에 추가합니다.

    인자:
    df : pandas.DataFrame
        원본 데이터프레임.
    columns : list of str
        애드스톡 변환을 적용할 열 이름 리스트.
    alphas : list of float
        각 열에 대한 애드스톡 감소율 (alpha) 리스트. 열의 순서와 매칭되어야 합니다.
    l_maxs : list of int
        각 열에 대한 애드스톡의 최대 지속 기간 (l_max) 리스트.
    normalize : bool
        애드스톡 변환 후 결과를 정규화할지 여부 (기본값: True).

    반환:
    df : pandas.DataFrame
        애드스톡 변환된 열을 포함한 데이터프레임.
    """
    
    if len(columns) != len(alphas) or len(columns) != len(l_maxs):
        raise ValueError("columns, alphas, l_maxs 리스트의 길이는 같아야 합니다.")
    
    for i, col in enumerate(columns):
        alpha = alphas[i]
        l_max = l_maxs[i]
        
        df[f"{col}_adstock"] = (
            geometric_adstock(x=df[col].to_numpy(), alpha=alpha, l_max=l_max, normalize=normalize)
            .eval()
            .flatten()
        )
    
    return df


def apply_saturation(df, columns, lams):
    """
    여러 열에 대해 포화 변환(logistic saturation)을 적용하고, 결과를 데이터프레임에 추가합니다.

    인자:
    df : pandas.DataFrame
        원본 데이터프레임.
    columns : list of str
        포화 변환을 적용할 열 이름 리스트.
    lams : list of float
        각 열에 대한 포화 변환의 lambda 값 리스트. 열의 순서와 매칭되어야 합니다.

    반환:
    df : pandas.DataFrame
        포화 변환된 열을 포함한 데이터프레임.
    """
    
    if len(columns) != len(lams):
        raise ValueError("columns와 lams 리스트의 길이는 같아야 합니다.")
    
    for i, col in enumerate(columns):
        lam = lams[i]
        
        df[f"{col}_saturated"] = (
            logistic_saturation(x=df[col].to_numpy(), lam=lam)
            .eval()
        )
    
    return df

def apply_adstock_saturation(df, columns, alphas, l_maxs, lams, normalize=True):
    """
    여러 열에 대해 기하학적 애드스톡 변환과 포화 변환을 순차적으로 적용하고, 결과를 데이터프레임에 추가합니다.

    인자:
    df : pandas.DataFrame
        원본 데이터프레임.
    columns : list of str
        애드스톡 및 포화 변환을 적용할 열 이름 리스트.
    alphas : list of float
        각 열에 대한 애드스톡 감소율 (alpha) 리스트. 열의 순서와 매칭되어야 합니다.
    l_maxs : list of int
        각 열에 대한 애드스톡의 최대 지속 기간 (l_max) 리스트.
    lams : list of float
        각 열에 대한 포화 변환의 lambda 값 리스트.

    반환:
    df : pandas.DataFrame
        애드스톡 및 포화 변환된 열을 포함한 데이터프레임.
    """
    
    # 먼저 애드스톡 변환 적용
    df = apply_adstock(df, columns, alphas, l_maxs, normalize=normalize)
    
    # 그런 다음 포화 변환 적용
    df = apply_saturation(df, [f"{col}_adstock" for col in columns], lams)
    
    return df

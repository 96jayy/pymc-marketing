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



def plot_transformed_data(df, columns, date_column="date_week", suptitle="Media Costs Data - Transformed"):
    """
    여러 개의 열에 대한 원본 데이터, 애드스톡 변환 및 포화 변환된 데이터를 시각화하는 함수.
    
    인자:
    df : pandas.DataFrame
        시각화할 데이터를 포함한 데이터프레임.
    columns : list of str
        원본 데이터, 애드스톡 변환, 포화 변환된 열의 기본 이름 리스트 (예: ['x1', 'x2']).
    date_column : str
        x축에 사용할 날짜 열의 이름 (기본값: 'date_week').
    suptitle : str
        전체 플롯의 제목 (기본값: 'Media Costs Data - Transformed').
        
    반환:
    None
    """

    # 열 개수에 따라 적절한 플롯의 행과 열 개수 설정
    n_columns = len(columns)
    fig, ax = plt.subplots(
        nrows=3, ncols=n_columns, figsize=(16, 9), sharex=True, sharey=False, layout="constrained"
    )

    # 각 열에 대해 원본 데이터, 애드스톡 변환, 포화 변환 데이터를 플로팅
    for i, col in enumerate(columns):
        # 첫 번째 행: 원본 데이터
        sns.lineplot(x=date_column, y=col, data=df, color=f"C{i}", ax=ax[0, i])
        ax[0, i].set_title(f"{col} - Original")

        # 두 번째 행: 애드스톡 변환 데이터
        sns.lineplot(x=date_column, y=f"{col}_adstock", data=df, color=f"C{i}", ax=ax[1, i])
        ax[1, i].set_title(f"{col} - Adstock Transformed")

        # 세 번째 행: 애드스톡 + 포화 변환 데이터
        sns.lineplot(x=date_column, y=f"{col}_adstock_saturated", data=df, color=f"C{i}", ax=ax[2, i])
        ax[2, i].set_title(f"{col} - Adstock & Saturation Transformed")

    # 전체 플롯 제목
    fig.suptitle(suptitle, fontsize=16)

    # 플롯 표시
    plt.show()


###trend와 seasonality는 arima같은 모델로 분해?

def plot_target(df, x, target):
    fig, ax = plt.subplots()
    sns.lineplot(x=x, y=target, color="black", data=df, ax=ax)
    ax.set(title="Sales (Target Variable)", xlabel="date", ylabel="y (thousands)")
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_attribution(df, beta_values, event_columns, feature_columns, intercept_col='intercept'):
    """
    기여도(Attribution) 데이터를 기반으로 막대 그래프를 생성하는 함수.
    
    인자:
    df : pandas.DataFrame
        기여도 계산에 사용될 데이터를 포함한 데이터프레임.
    beta_values : dict
        feature_columns에 대한 기여도 배수를 포함하는 딕셔너리. 예: {"x1": 3.0, "x2": 2.0}.
    event_columns : list of str
        이벤트 데이터를 포함한 열의 이름 리스트.
    feature_columns : list of str
        애드스톡 및 포화 변환 데이터를 포함한 열의 이름 리스트.
    intercept_col : str
        절편 데이터를 포함한 열의 이름 (기본값: 'intercept').

    반환:
    None
    """

    # 각 기여도의 합 계산
    contributions = [
        df[intercept_col].sum(),  # 절편(intercept) 기여도
    ]

    # feature_columns의 기여도 계산
    for feature, beta in beta_values.items():
        contributions.append((beta * df[f"{feature}_adstock_saturated"]).sum())

    # 이벤트 기여도 계산
    for event_col in event_columns:
        contributions.append(df[event_col].sum())

    # 트렌드와 계절성 기여도 추가
    contributions.append(df["trend"].sum())
    contributions.append(df["seasonality"].sum())

    # 막대 그래프 생성
    fig, ax = plt.subplots()
    
    # 막대 그래프 그리기
    labels = [intercept_col] + feature_columns + event_columns + ["trend", "seasonality"]
    
    ax.bar(
        labels,
        contributions,
        color=["C0" if x >= 0 else "C3" for x in contributions],
        alpha=0.8,
    )
    
    # 막대에 레이블 추가
    ax.bar_label(
        ax.containers[0],
        fmt="{:,.2f}",
        label_type="edge",
        padding=2,
        fontsize=15,
        fontweight="bold",
    )
    
    # 제목과 축 레이블 설정
    ax.set(title="Sales Attribution", ylabel="Sales (thousands)")

    # 그래프 출력
    plt.show()


def calculate_contribution_shares(df, feature_columns, beta_values):
    """
    각 특징 열에 대한 기여도 비율을 계산하는 함수.
    
    인자:
    df : pandas.DataFrame
        기여도 계산에 사용될 데이터를 포함한 데이터프레임.
    feature_columns : list of str
        각 특징 열 이름 리스트 (예: ['x1', 'x2', ...]).
    beta_values : dict
        각 특징 열에 대한 기여도 배수 값을 포함한 딕셔너리. 예: {'x1': 3.0, 'x2': 2.0}.
        
    반환:
    contribution_shares : dict
        각 특징 열에 대한 기여도 비율을 포함한 딕셔너리.
    """

    # 총 기여도 계산 (가중 합)
    total_contribution = sum(
        (beta_values[feature] * df[f"{feature}_adstock_saturated"]).sum() for feature in feature_columns
    )

    # 각 특징 열의 기여도 비율 계산
    contribution_shares = {}
    for feature in feature_columns:
        contribution = (beta_values[feature] * df[f"{feature}_adstock_saturated"]).sum()
        contribution_shares[feature] = contribution / total_contribution

    # 기여도 비율 출력
    for feature, share in contribution_shares.items():
        print(f"Contribution Share of {feature}: {share:.2f}")
    
    return contribution_shares

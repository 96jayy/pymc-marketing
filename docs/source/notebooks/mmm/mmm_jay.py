import matplotlib.pyplot as plt
import seaborn as sns

def plot_media_costs(df, media_columns, date_column):    
    # 미디어 채널의 수에 따라 플롯 행 개수 설정
    n_rows = len(media_columns)
    
    # 플롯 생성
    fig, ax = plt.subplots(
        nrows=n_rows, ncols=1, figsize=(10, 3 * n_rows), sharex=True, sharey=True, layout="constrained"
    )
    
    # ax가 하나일 경우 리스트 형태로 변환
    if n_rows == 1:
        ax = [ax]
    
    # 각 미디어 채널에 대한 라인 플롯 생성
    for i, col in enumerate(media_columns):
        sns.lineplot(x=date_column, y=col, data=df, ax=ax[i], color=f"C{i}")
        ax[i].set_title(f"{col} contribution")
    
    # 하단 x축 레이블 설정
    ax[-1].set(xlabel="date")
    
    # 전체 제목 설정
    fig.suptitle("Media Costs Data", fontsize=16)
    
    # 플롯 표시
    plt.show()
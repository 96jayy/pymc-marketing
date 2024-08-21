import matplotlib.pyplot as plt
import seaborn as sns

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
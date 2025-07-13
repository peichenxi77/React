import matplotlib.pyplot as plt
import pandas as pd
import json
def draw():
    # 原始损失数据
    with open("loss_logs/loss_log.json","r",encoding='utf-8') as f:
        loss_data=json.load(f)
        print("len",len(loss_data["steps"]))
        print("len",len(loss_data["losses"]))

    # 1. 对相同step的loss取平均
    df = pd.DataFrame({
        "step": loss_data["steps"],
        "loss": loss_data["losses"]
    })
    avg_loss = df.groupby("step", as_index=False)["loss"].mean()  # 按step分组并计算平均损失

    # 2. 绘图（不显示中文，使用默认字体）
    plt.figure(figsize=(10, 6))

    # 绘制平均损失曲线
    plt.plot(
        avg_loss["step"],
        avg_loss["loss"],
        marker='o',
        markersize=6,
        linestyle='-',
        linewidth=2,
        color='blue',
        label='Average Loss'
    )

    # 添加标题和坐标轴标签（英文）
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)

    # 添加网格线
    plt.grid(alpha=0.3)

    # 添加图例
    plt.legend()

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('loss_curve_avg.png', dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

draw()
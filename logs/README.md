# 训练日志

数据集: selfie2anime

对各阶段的损失进行了截取, 其中`d_loss`变化范围在5以下, `g_loss`变化范围比较大, 在2000以内, 但均值有下降趋势. 只要在训练过程中, 能达到日志中的损失(只需要有一次迭代能达到), 即表明训练正常.
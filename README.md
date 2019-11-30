# BCNN Keras 

# 天翼杯垃圾分类比赛
## 二分类问题，预测图片属于可回收垃圾的概率

## 存在的问题与解决方法：
1. 测评图片进行了随机裁剪与模糊，与训练集不一致，所以训练集经过了我的随机变换
2. 经测试 densenet+bcnn 效果最好，
3. 不同图像尺度下训练，并测试，
4. 对图像做了增强，尺度缩放，
5. 学习率阶段性调整和自动调整，
## 不足
1. 没有挑出难以分类的样本做针对性提升方案
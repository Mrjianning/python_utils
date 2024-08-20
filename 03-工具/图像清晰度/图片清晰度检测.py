import cv2
import numpy as np

"""
    Tenengrad梯度方法利用Sobel算子分别计算水平和垂直方向的梯度，在图像处理中，一般认为对焦好的图像具有更尖锐的边缘，故具有更大的梯度函数值。
    同一个场景下梯度值越高，图像越清晰。衡量的指标是经过Sobel算子处理后的图像的平均灰度值，值越大，代表图像越清晰。
"""
def Tenengrad(path):
    imageSource = cv2.imread(path)
    imageGrey = cv2.cvtColor(imageSource, cv2.COLOR_BGR2GRAY)

    # 使用 Sobel 算子计算边缘强度
    sobelx = cv2.Sobel(imageGrey, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(imageGrey, cv2.CV_64F, 0, 1, ksize=3)
    imageSobel = cv2.magnitude(sobelx, sobely)

    # 计算图像的平均灰度
    meanValue = np.mean(imageSobel)

    # 将平均灰度值转换为字符串
    meanValueString = "Articulation(Sobel Method): {:.2f}".format(meanValue)

    # 在原图上添加文本
    cv2.putText(imageSource, meanValueString, (20, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 3)
    cv2.imshow("show", imageSource)
    cv2.waitKey(0)

'''
    Laplacian梯度方法
'''
def Laplacian(path):
    imageSource = cv2.imread(path)
    imageGrey = cv2.cvtColor(imageSource, cv2.COLOR_BGR2GRAY)

    # 使用拉普拉斯算子计算边缘强度
    imageLaplacian = cv2.Laplacian(imageGrey, cv2.CV_64F)

    # 计算边缘强度的绝对值或平方的平均值
    meanValue = np.mean(np.abs(imageLaplacian))  # 或者 np.mean(imageLaplacian**2)

    # 将平均灰度值转换为字符串
    meanValueString = "Articulation(Laplacian Method): {:.2f}".format(meanValue)

    # 在原图上添加文本
    cv2.putText(imageSource, meanValueString, (20, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 3)
    cv2.imshow("show", imageSource)
    cv2.waitKey(0)

'''
    方差方法
    对焦清洗的图像相比对焦模糊的图像，它的数据之间的灰度差异应该更大，即它的方差应该较大，可以通过图像灰度数据的方差来衡量图像的清晰度，方差越大，表示清晰度越好。
'''
def variance(path):
    imageSource = cv2.imread(path)
    imageGrey = cv2.cvtColor(imageSource, cv2.COLOR_BGR2GRAY)

    # 计算灰度图像的标准差
    _, stdDev = cv2.meanStdDev(imageGrey)
    variance = stdDev[0][0] ** 2

    # 将方差转换为字符串并显示在图像上
    varianceString = "Articulation(Variance Method): {:.2f}".format(variance)
    cv2.putText(imageSource, varianceString, (20, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 3)
    cv2.imshow("show", imageSource)
    cv2.waitKey(0)

if __name__ == "__main__":
    cv2.namedWindow("show",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("show",1000,600)

    path=r"H:\360MoveData\Users\Administrator\Desktop\imaegs\11.jpg"
    Tenengrad(path)

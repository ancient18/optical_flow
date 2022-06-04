import math
import torch
import torch.nn as nn


class PositionLoss(nn.Module):
    def __init__(self):
        super(PositionLoss, self).__init__()
        self.is_cuda = torch.cuda.is_available()

    def forward(self, offset, optical_flow):
        """
        求offset到无数条optical flow的损失 注意h*w维度要一致
        :param optical_flow: b*(2*n)*h*w n条光流
        :param offset: b*18*h*w (18/2)个采样点 前一半x坐标，后一半y坐标
        :return:
        """
        b, offset_num, h, w = offset.shape
        offset_num = int(offset_num / 2)  # 可形变卷积采样点数量 一般是3*3、5*5
        _, optical_flow_num, _, _ = optical_flow.shape
        optical_flow_num = int(optical_flow_num / 2)  # 光流数量 一般是1条或2条
        loss_average = []  # 采样点的loss求均值  公式6
        for i in range(offset_num):  # 遍历所有采样点
            x = offset[:, i, :, :]
            y = offset[:, i + offset_num, :, :]
            loss_min = []  # 求dij的最小值
            for j in range(optical_flow_num):  # 遍历所有光流
                u = optical_flow[:, j, :, :]
                v = optical_flow[:, j + 1, :, :]
                x0 = torch.div(torch.add(torch.mul(u, x.pow(2)), torch.mul(u, torch.mul(v, y))), torch.add(u.pow(2), v.pow(2)))  # 采样点(x,y)到光流向量(u,v)的垂线的垂足的横坐标x0=(u*x*x+u*v*y)/(u*u+v*v)
                zero = torch.zeros([b, h, w])
                if self.is_cuda:
                    zero = zero.cuda()
                min_u = zero.min(u)
                max_u = zero.max(u)
                ge_mask = torch.zeros([b, h, w])  # 01矩阵 标记出x0矩阵中，大于等于min_u矩阵的元素
                if self.is_cuda:
                    ge_mask = ge_mask.cuda()
                le_mask = torch.zeros([b, h, w])  # 01矩阵 标记出x0矩阵中，小于等于max_u矩阵的元素
                if self.is_cuda:
                    le_mask = le_mask.cuda()
                torch.ge(x0, min_u, out=ge_mask)  # x0>=min_u
                torch.le(x0, max_u, out=le_mask)  # x0<=max_u
                min_dis1 = torch.abs(torch.div(torch.sub(torch.mul(v, x), torch.mul(u, y)), torch.sqrt(torch.add(u.pow(2), v.pow(2)))))  # 点到直线距离 除0会出现inf(负无穷)
                inf_local = torch.where(min_dis1 == float('inf'))  # 得到inf元素坐标 实际没啥元素是inf，考虑删掉此行
                # print(inf_local)
                min_dis1[inf_local] = 0  # 所有inf元素置为0
                min_dis1 = torch.mul(min_dis1, ge_mask)  # min_u <= x0 <= max_u  在此区域内的元素，使用点到直线距离计算min_dis，不在此区域的元素，使用两点间距离公式计算min_dis
                min_dis1 = torch.mul(min_dis1, le_mask)  # 区域内的元素会计算出点到直线距离，区域外的元素以及除0的元素，值会被置为0.0
                zero_local = torch.where(min_dis1 == 0)  # 找出不使用 点到直线距离 计算公式的像素位置，这些位置的值已被置为0.0
                d1 = torch.sqrt(torch.add(x.pow(2), y.pow(2)))  # 采样点(x,y)到原点(0,0)的距离d1 = math.sqrt(x * x + y * y)
                d2 = torch.sqrt(torch.add(torch.sub(x, u).pow(2), torch.sub(y, v).pow(2)))  # 采样点(x,y)到光流向量端点(u,v)的距离d2=math.sqrt((x-u)*(x-u)+(y-v)*(y-v))
                min_dis2 = d1.min(d2)  # min_dis2 = min(d1, d2)
                min_dis = min_dis1
                min_dis[zero_local] = min_dis2[zero_local]  # 不使用 点到直线距离 计算公式的像素 就使用两点间距离公式
                loss_min.append(min_dis)  # 采样点(x,y)到光流向量(u,v)的最短距离 即dij
            loss_dij = loss_min[0]
            for item in loss_min:
                loss_dij = loss_dij.min(item)  # min(di1,di2,...)
            loss_average.append(loss_dij)
        loss = (sum(loss_average) / offset_num).sum()  # 公式6 min(di1,di2)求和后除以采样点数量n  再对所有点的损失求和
        loss = loss / (h * w)  # 除以所有像素点数量
        return loss


class PositionLossVal(nn.Module):
    """
    验证矩阵优化的结果是否正确
    """

    def __init__(self):
        super(PositionLossVal, self).__init__()

    def forward(self, offset, optical_flow):
        b, offset_num, h, w = offset.shape
        offset_num = int(offset_num / 2)  # 偏移点数量
        _, optical_flow_num, _, _ = optical_flow.shape
        optical_flow_num = int(optical_flow_num / 2)  # 光流数量
        loss_list = []  # 所有点的loss求和
        for k in range(b):
            for m in range(h):
                for n in range(w):
                    loss_average = []  # 9个采样点的loss求均值
                    for i in range(offset_num):  # 遍历所有采样点
                        x = offset[k, i, m, n]
                        y = offset[k, i + 9, m, n]
                        loss_min = []  # 求dij的最小值
                        for j in range(optical_flow_num):  # 遍历所有光流
                            u = optical_flow[k, j, m, n]
                            v = optical_flow[k, j + 1, m, n]
                            # min_dis = x.pow(2) + y.pow(2)
                            x0 = (u * x * x + u * v * y) / (u * u + v * v)  # 采样点(x,y)到光流向量(u,v)的垂线的垂足的横坐标
                            min_u = min(0, u)
                            max_u = max(0, u)
                            if min_u <= x0 <= max_u:
                                min_dis = abs(v * x - u * y) / math.sqrt(u * u + v * v)
                            else:
                                d1 = math.sqrt(x * x + y * y)
                                d2 = math.sqrt((x - u) * (x - u) + (y - v) * (y - v))
                                min_dis = min(d1, d2)
                            loss_min.append(min_dis)  # 采样点(x,y)到光流向量(u,v)的最短距离 即dij
                        loss_dij = loss_min[0]
                        for item in loss_min:
                            loss_dij = min(loss_dij, item)
                        loss_average.append(loss_dij)
                    loss_list.append(sum(loss_average) / offset_num)
        loss = sum(loss_list)  # 所有通道所有位置所有采样点的总loss
        loss = loss / (h * w)
        return loss


# if __name__ == '__main__':
#     import time
#
#     offset = torch.rand([4, 50, 7, 12])
#     optical_flow = torch.rand([4, 8, 7, 12])
#     loss = PositionLoss()
#     start = time.time()
#     output = loss(optical_flow=optical_flow, offset=offset)
#     print(time.time() - start)
#     print(output)
#
#     loss = PositionLossVal()
#     start = time.time()
#     output = loss(optical_flow=optical_flow, offset=offset)
#     print(time.time() - start)
#     print(output)

    # tensor1 = torch.tensor([[[3, 2, 1], [2, 2, 2], [1, 2, 3]]])
    # tensor2 = torch.tensor([[[8, 2, 0], [1, 0, 1], [7, 1, 1]]])
    # print(tensor1.min(tensor2))
    # tensor3 = torch.tensor([[[6, 3, 0], [9, 9, 9], [7, 2, 2]]])
    # b, c, h = tensor1.shape
    # ge_mask = torch.tensor([b, c, h])
    # le_mask = torch.tensor([b, c, h])
    # torch.ge(tensor2, tensor1, out=ge_mask)
    # torch.le(tensor2, tensor3, out=le_mask)
    # print(tensor2.shape, ge_mask.shape)
    # print(tensor2 * ge_mask * le_mask)
    # out = torch.div(tensor1, tensor2)
    # out = torch.mul(out, tensor3)
    # inf_local = torch.where(out == float('inf'))
    # out[inf_local] = 0
    # print(out)

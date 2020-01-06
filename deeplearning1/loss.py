import matplotlib.pyplot as plt


def loadData():
    with open("./loss1.txt") as fp:
        data = fp.readlines()
        data = [float(v.strip()) for v in data]
        data = [v for v in data if v <1]
    return data[:]
def loadData2():
    with open("./loss2.txt") as fp:
        data = fp.readlines()
        data = [float(v.strip()) for v in data]
        data = [v for v in data if v <1]
    return data[:]

plt.plot(loadData(),label = 'train')
plt.plot(loadData2(),label = 'valid')
plt.legend()
plt.show()
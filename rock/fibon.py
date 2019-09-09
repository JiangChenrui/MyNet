def fibon(n):
    a = b = 1
    for i in range(n):
        yield a
        a, b = b, a + b


# 杨辉三角
def triangles(n):
    L = [1]
    while True:
        yield L
        L.append(0)
        L = [L[i - 1] + L[i] for i in range(len(L))]


class User(object):
    def __getattr__(self, name):
        print('调用了__getatter__方法')
        return super(User, self).__getattr__(name)

    def __setattr__(self, name, value):
        print('调用了__setatter__方法')
        return super(User, self).__setattr__(name, value)

    def __delattr__(self, name):
        print('调用了__delattr__方法')
        return super(User, self).__delattr__(name)

    def __getattribute__(self, name):
        print('调用了__getattribute__方法')
        return super().__getattribute__(name)


if __name__ == '__main__':
    # user = User()
    # user.attr1 = True
    # user.attr1
    # try:
    #     user.atter2
    # except AttributeError:
    #     pass
    # del user.attr1

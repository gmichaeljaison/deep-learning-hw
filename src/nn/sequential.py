from src.nn import Module


class Sequential(Module):
    def __init__(self, name, modules=None):
        super().__init__(name)
        self.modules = modules or list()

    def add(self, module):
        self.modules.append(module)

    def find(self, name):
        for module in self.modules:
            if module.name == name:
                return module
        return None

    def forward(self, x):
        super().forward(x)

        h = x
        for layer in self.modules:
            h = layer.forward(h)
        self.h = h
        return self.h

    def back_propagate(self, dh, lr):
        super().back_propagate(dh, lr)

        for layer in reversed(self.modules):
            layer.back_propagate(dh, lr)
            dh = layer.dx

        self.dx = self.modules[0].dx

        # self.forward(x)
        #
        # for layer in reversed(self.modules):
        #     layer.back_propagate(layer.x, dh, lr)
        #     dh = layer.dx
        #
        # self.dh = y
        # self.dx = self.modules[0].dx

    def update_gradient(self, dh):
        pass

    def update_weight(self, lr):
        pass

    def output_gradient(self, y):
        return self.modules[-1].der_x_entropy(y)

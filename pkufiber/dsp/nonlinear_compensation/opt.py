'''
    optimizers.
'''
from torch import optim

class AlterOptimizer(optim.Optimizer):
    def __init__(self, params_list, lr_list, alternate=True):
        assert len(params_list) == len(lr_list), "每组参数必须有对应的学习率"
        
        # 初始化基类
        optimizers_params = []
        for params, lr in zip(params_list, lr_list):
            optimizers_params.append({'params': params, 'lr': lr})
        super(AlterOptimizer, self).__init__(optimizers_params, {})

        self.optimizers = [
            optim.Adam(group['params'], lr=group['lr']) for group in self.param_groups
        ]
        self.alternate = alternate
        self.current_opt_index = 0  # 当前优化器的索引

    def step(self, closure=None):
        # 如果设置为轮流优化，则每次只更新当前优化器
        if self.alternate:
            loss = None
            if closure is not None:
                loss = closure()
            self.optimizers[self.current_opt_index].step()
            # 更新当前优化器索引，循环到下一个
            self.current_opt_index = (self.current_opt_index + 1) % len(self.optimizers)
            return loss
        else:
            # 同时更新所有优化器
            for optimizer in self.optimizers:
                optimizer.step()

    def zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

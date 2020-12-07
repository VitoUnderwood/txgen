# -*- coding:utf-8 -*-
# 精确导入
# 外界看到的相当于 models下Test类, from models import Test
from .RNN import Test
from .RNN import RNN
# 模糊导入, form models import * ,执行上面的精确导入的同时，还可以导入下面的RNN包，使用RNN.RNN, RNN.Tests，也可以直接使用RNN，Test
__all__ = ['RNN']
# import models 只能执行精确导入的方式，但是调用必须加上包的名称，models.Test
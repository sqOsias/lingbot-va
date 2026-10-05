"""MemoryWAM 机制级 reference checks。

这里的代码刻意与 ``wan_va`` 生产模块隔离：它只验证 token 类型、mask、
函数式缓存、streaming 等价性和梯度路径，不加载 LingBot-VA 5B checkpoint。
"""

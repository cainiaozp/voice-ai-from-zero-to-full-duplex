# Dense 与 MoE：Router 如何激活部分专家

```bash
pip install -r requirements/pytorch.txt
python extras/dense-vs-moe/moe_toy.py
```

此代码对4个输入 Token 做 Top-2 路由。共有6个专家，最后一个故意完全不被选中，因此它的参数没有当次梯度。虽然没有被选中，**它的权重仍在 `model.parameters()`，因此总参数量/权重存储不会自动下降**。生产 MoE 的负载均衡、专家并行和通信在本小例中没有实现。

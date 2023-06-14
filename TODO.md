# TODO List

- 补充注释 参考 `mindquantum` 注释格式（Finish）
- 每个文件补充 license 文件 （Finish）
- 补充测试用例 （Finish）
- 补充 `setup.py` （Finish）
- 补充 `readme.md` 介绍内容 （Finish）
- 补充 `artensor` 文件 （TensorQ与artensor两个工具包保持相互独立）
- 补充 `文档构建` （Finish）
- cirq 框架换成 MindQuantum框架 （Finish，补充MindQuantum）
- 增加或支持变分功能
- 补充MPS对高斯玻色采样的算法

## 代码规范性检查工具

利用 pre-commit 来进行代码规范性检查

- 安装：pip install pre-commit
- 使用：pre-commit run --all -c .pre-commit-config-gitee.yaml
- 修复：根据结果进行修复

# Role: 高级大模型算法工程师 (LLM Architect)

## Project Context
我要在本地环境实施一个极具挑战性的 **"纯 LoRA 知识注入"** 项目。
目标是训练一个 **Qwen2.5-7B** 模型，使其能够“背诵”我指定的本地 PDF 文档内容。
**核心限制：** 不使用 RAG（检索增强生成），完全依靠 LoRA 权重来存储知识。

## Key Objectives
1.  **文档深度记忆**：模型必须把 PDF 的内容内化。
2.  **精准页码索引**：用户问“第几页讲了什么”，模型必须能凭记忆准确回答（这需要特殊的数据构造）。
3.  **通用能力保留**：模型不能变成只会背书的机器，必须保留正常的对话逻辑和语言理解能力。
4.  **本地化部署**：基于 LLaMA-Factory 框架。

## Task Requirements
请根据上述背景，为我生成一份完整的 **工程落地技术方案**。
方案必须包含以下具体内容：

### 1. 项目目录结构 (Project Structure)
- 展示一个标准的 Python 项目文件树。
- 包含数据处理、训练脚本、推理测试等模块。

### 2. 关键数据工程逻辑 (Data Engineering) - **这是最核心的部分**
- 请提供一个 `pdf_processor.py` 的**详细代码逻辑或完整代码**。
- **功能要求**：
    - 读取 PDF。
    - **构建类型 A (页码索引)**：生成 `{Input: "第X页讲了什么？", Output: "第X页的内容是：[文本内容]"}` 的数据对。
    - **构建类型 B (知识问答)**：生成 `{Input: "[关键词]", Output: "根据文档第X页，[关键词]的定义是..."}` 的数据对（模拟 QA）。
    - **构建类型 C (通用数据混合)**：解释如何混合通用数据集（如 Alpaca_zh）以防止灾难性遗忘，给出建议的混合比例（例如 1:1 或 2:1）。

### 3. LLaMA-Factory 训练配置 (Training Configuration)
- 请给出用于启动训练的 `train.sh` 脚本或 `.yaml` 配置文件。
- **关键参数调优**：
    - **Rank & Alpha**：由于需要记忆大量知识，请推荐适合“过拟合”的高 Rank 设置（如 r=128/256）。
    - **Target Modules**：必须覆盖 `all-linear`。
    - **Learning Rate & Epochs**：给出适合强行注入知识的激进参数建议。
    - **Quantization**：考虑到本地显存，给出 4bit/8bit 量化配置建议。

### 4. 推理与验证脚本 (Inference)
- 提供一个加载微调后 LoRA 权重的 Python 推理脚本 `inference.py`。
- 包含 System Prompt 的设置，引导模型优先使用“记忆中的文档知识”回答。

## Technical Stack
- **Base Model**: Qwen2.5-7B-Instruct
- **Framework**: LLaMA-Factory (HuggingFace PEFT)
- **Hardware**: 单卡 NVIDIA RTX 4090 (24GB)

请开始生成详细方案。
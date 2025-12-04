# Janus VLM 推理性能优化说明

## 一、问题分析

在对 `janus_pro` 视觉语言模型（VLM）推理流程进行性能调优时，我们观察到 **Prefill 阶段耗时显著偏高**。初步尝试通过调整底层算子 API 来提升性能，但收效甚微——不仅推理时延改善有限，还导致精度无法对齐。

通过使用 MindSpore Profiler 工具深入分析，发现性能瓶颈主要集中在 **频繁查询 tokenizer 中特殊 token 的 ID** 上。每次访问 `image_id`、`image_start_id`、`image_end_id` 和 `pad_id` 属性时，都会重复执行字典查找操作（`self.tokenizer.vocab.get(...)`），在长序列或批量推理场景下造成大量冗余开销。

## 二、优化方案

我们将这些特殊 token ID 的获取逻辑 **从属性方法中移出**，改为在 `VLChatProcessor` 初始化阶段一次性计算并缓存，后续直接返回预存值，避免重复查询。

### 关键代码变更：

#### 1. 初始化时缓存特殊 Token ID
```python
# 在 __init__ 中提前解析并保存 ID
image_id = self.tokenizer.vocab.get(image_tag)
self.my_image_id = image_id
self.my_image_start_id = self.tokenizer.vocab.get(image_start_tag)
self.my_image_end_id = self.tokenizer.vocab.get(image_end_tag)
self.my_pad_id = self.tokenizer.vocab.get(pad_tag)
```
#### 2. 属性方法简化为直接返回缓存值
```python
     @property
     def image_id(self):
-        image_id = self.tokenizer.vocab.get(self.image_tag)
-        return image_id
+        return self.my_image_id
 
     @property
     def image_start_id(self):
-        image_start_id = self.tokenizer.vocab.get(self.image_start_tag)
-        return image_start_id
+        return self.my_image_start_id
 
     @property
     def image_end_id(self):
-        image_end_id = self.tokenizer.vocab.get(self.image_end_tag)
-        return image_end_id
+        return self.my_image_end_id
 
     @property
     def pad_id(self):
-        pad_id = self.tokenizer.vocab.get(self.pad_tag)
-        # pad_id = self.tokenizer.pad_token_id
-        # if pad_id is None:
-        #     pad_id = self.tokenizer.eos_token_id
-
-        return pad_id
+        return self.my_pad_id
```

## 三、 评测结果：
| 评分项           | 得分       |
|:------------------:|:------------:|
| 峰值显存得分     | 100        |
| Prefill时延得分  | 266.2944   |
| Decode时延得分   | 95.9177    |
| 总分             | 154.0707   |
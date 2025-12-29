# 本地多模态智能文献与图像管理助手 

---

## 1. 项目简介

本项目是一个基于 **Python 的本地多模态 AI 智能助手**，用于管理本地的学术论文（PDF）与图像数据。  
不同于传统的基于文件名或关键词的检索方式，本项目利用 **文本与图像的语义嵌入（Embedding）**，实现：

- **论文的语义搜索**
- **论文的自动分类与整理**
- **本地图像的以文搜图**

系统支持 **完全离线运行**，所有模型均可本地加载，适合在个人电脑上构建私有的本地知识库。

---

## 2. 核心功能

### 2.1 智能文献管理

#### 论文语义搜索
- 支持使用自然语言查询论文内容，例如：
  - *“What is self-attention?”*
  - *“experience replay in reinforcement learning”*
- 系统基于语义相似度返回最相关论文
- 可返回：
  - 论文路径
  - 匹配片段
  - 对应页码范围

#### 自动分类与整理
- 添加论文时，根据给定的**语义化分类描述**自动归类
- 示例分类：
  - computer vision
  - natural language processing
  - reinforcement learning

#### 批量整理
- 支持对一个混乱的 PDF 文件夹进行“一键整理”
- 自动扫描并归档到对应主题目录

---

### 2.2 智能图像管理

#### 以文搜图
- 基于 CLIP 多模态模型
- 支持使用自然语言搜索本地图片，例如：
  - *“sunset by the sea”*
  - *“running on grass”*
- 即使图片文件名与查询无关，也可以正确检索

---

## 3. 环境配置与依赖安装

### 3.1 运行环境

- 操作系统：**Windows 11**
- Python 版本：**Python 3.8**
- 运行方式：命令行（CMD / PowerShell / Conda）

### 3.2 安装依赖

建议在虚拟环境中安装：

```bat
pip install -r requirements.txt
```

---

## 4. 使用说明

> 所有命令均为 **Windows 单行命令**，可直接在 CMD / PowerShell 中复制执行。

> 使用前请自行下载模型文件到本地加载。

### 4.1 添加并分类单篇论文

```bat
python main.py add_paper ".\test_papers\Attention Is All You Need.pdf" --topics "computer vision and image understanding,natural language processing and language modeling,reinforcement learning and decision making"
```

### 4.2 批量整理论文文件夹

```bat
python main.py batch_organize ".\batch_test_papers" --topics "computer vision and image understanding,natural language processing and language modeling,reinforcement learning and decision making"
```

### 4.3 论文语义搜索

使用自然语言查询论文内容：
```bat
python main.py search_paper "self attention mechanism"
```
仅返回匹配的论文文件列表（不显示文本片段）：
```bat
python main.py search_paper "experience replay" --files-only
```

### 4.4 建立图像索引

```bat
python main.py index_images ".\test_images"
```
该步骤会对指定文件夹中的图像进行编码，并建立本地图像向量索引。

### 4.5 以文搜图

```bat
python main.py search_image "sunset by the sea"
```
系统将返回与文本语义最匹配的本地图像路径及相似度分数。

---

## 5. 技术选型说明

本项目采用模块化设计，所有模型和数据库组件均支持本地部署，能够在无网络环境下运行。

---

### 5.1 文本建模

- **SentenceTransformers**
  - 使用模型：`all-MiniLM-L6-v2`
  - 主要用途：
    - 论文语义搜索
    - 论文自动分类
  - 选择原因：
    - 模型轻量
    - 推理速度快
    - 可在 CPU 上运行，适合本地部署

---

### 5.2 图像-文本多模态建模

- **CLIP**
  - 使用模型：`ViT-B/32`
  - 主要用途：
    - 文本与图像的语义对齐
    - 以文搜图
  - 特点：
    - 图文共享语义空间
    - 支持完全本地加载
    - 无需依赖云端 API

---

### 5.3 向量数据库

- **ChromaDB**
  - 嵌入式向量数据库
  - 无需独立服务器
  - 支持本地持久化存储
  - 主要用途：
    - 存储论文语义向量
    - 存储图像语义向量
    - 支持基于相似度的快速检索
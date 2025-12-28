# IMDb Sentiment Analysis (Rust / 自作NN)

このリポジトリは、**Rustでニューラルネットワークを一から実装し、IMDb Large Movie Review Dataset を用いて感情分析（2値分類）を行う学習用プロジェクト**です。

> [!NOTE]
> `.fear` ファィルの読み込みを実装しているので、 IMDb Large Movie Review Dataset でなくとも学習が行えます。

深層学習フレームワーク（PyTorch / TensorFlow など）は使用せず、
- forward / backward
- 勾配計算
- パラメータ更新
- Dataset / Iterator / Batch 処理

を自前で実装しています。

## 使用データセット

### IMDb Large Movie Review Dataset v1.0

- 映画レビュー 50,000 件（train 25k / test 25k）
- ラベル：positive / negative（2値）

> [!NOTE]
> このリポジトリにデータセットは同封されていません。
> データセットが`aclImdb_v1/aclImdb/`下にあることを想定しています。

IMDb Dataset は以下の論文に基づいて公開されています：

> Maas et al.,  
> *Learning Word Vectors for Sentiment Analysis*, ACL 2011

```

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
author    = {Maas, Andrew L. et al.},
title     = {Learning Word Vectors for Sentiment Analysis},
booktitle = {ACL-HLT},
year      = {2011}
}

```

## 実装内容

### モデル構成（例）

- Bag of Words 入力
- Linear
- ReLU
- Linear
- Softmax
- Cross Entropy Loss

※ MNIST 用に作成した NN 構造を流用し、入力のみ BoW に対応

## 技術的特徴

- Rust 2024 Edition
- 自作 `Module` trait
- forward / backward / step 明示実装
- Iterator ベース Dataset（巨大データを一括ロードしない）

## 注意事項

- 本コードは **学習目的** であり、FFT等を使用した速度・精度最適化は行っていません
- unsafe / SIMD / 並列化などは使用していません

## 免責

本リポジトリのコードおよび学習結果について、  
いかなる損害についても責任を負いません。

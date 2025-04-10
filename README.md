# KMNIST Classifier with PyTorch Lightning & Hydra
Introduction to MLflow using Kuzushiji-MNIST dataset.

PyTorch Lightning を用いて KMNIST データセットに対する分類モデルを学習、評価しています。
また、Hydra を用いて設定管理を行い、MLFlow Logger による実験管理を実装しています。

## 特徴

- **データモジュール**: `KuzushijiMNIST` クラスにより、KMNIST データセットのロード、前処理、データローダの生成を行います。
- **モデル**: [KMNISTClassifier` クラスでは、シンプルな全結合ネットワークを構築し、3層の線形レイヤーとアクティベーション関数を使っています。
- **学習と評価**:
  - `training_step`、`validation_step`、`test_step` で損失や精度を計算し、エポック終了時に結果をログ出力します。
  - `EarlyStopping` コールバックにより、バリデーション損失の改善が見られない場合に学習を停止します。
- **実験管理**: MLFlow Logger を利用して、ハイパーパラメータや実験結果を記録します。
- **設定管理**: Hydra により、`config.yaml` などの外部設定ファイルで学習パラメータやモデル設定を管理します。

## 必要なライブラリ

以下の主要なライブラリが必要です：

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Hydra](https://hydra.cc/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [MLFlow](https://mlflow.org/)

インストール例:

```sh
pip install torch pytorch-lightning hydra-core torchmetrics torchvision mlflow
```

## 実行方法

1. `config.yaml` などで設定を行います（例: データセットパラメータ、モデルパラメータ）。
2. 以下のコマンドで学習と評価を行います:

```sh
python main.py
```

Hydra により、必要に応じて設定の上書きも可能です。

## ファイル構成

```
config.yaml
main.py
data/
 └── KMNIST/
      └── raw/
           ├── t10k-images-idx3-ubyte
           ├── t10k-images-idx3-ubyte.gz
           ├── t10k-labels-idx1-ubyte
           ├── t10k-labels-idx1-ubyte.gz
           ├── train-images-idx3-ubyte
           ├── train-images-idx3-ubyte.gz
           ├── train-labels-idx1-ubyte
           └── train-labels-idx1-ubyte.gz
```

## カスタマイズ

- **モデル構造**: `KMNISTClassifier` クラス内のレイヤーの設定を変更して、より複雑なネットワークに拡張可能です。
- **学習パラメータ**: Hydra の設定ファイル（`config.yaml`）で、学習率、隠れ層の次元、アクティベーション関数、オプティマイザなどを変更できます。
- **データ前処理**: `KuzushijiMNIST` クラス内の変換処理を変更することで、データの正規化やその他の前処理のカスタマイズが可能です。


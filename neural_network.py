import numpy as np

def sigmoid(x):
  # 活性化関数（シグモイド関数）：f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # シグモイド関数の微分：f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_actual, y_pred):
  # 実装③参照
  return ((y_actual - y_pred)**2).mean()

class NeuralNetwork:
  """
  以下のコードは学習目的のため、あくまで単純化したものです。
  実際のNNはもっと複雑であろうことは想像に難くありません。
  """
  def __init__(self):
    # 重み
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # バイアス
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_actuals):
    """
    - (n x 2)のNumPy配列で、nはデータセットのサンプル数
    - all_y_actualsはn個の要素を持つNumPy配列
    all_y_actualsの要素はデータセットの要素に対応する。
    """
    learn_rate = 0.1
    epochs = 1000 # データセット内をループする回数

    for epoch in range(epochs):
      for x, y_actual in zip(data, all_y_actuals):
        # feedforward：後でこれらの値を利用
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3 
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # 偏微分を計算
        # 命名規則：d_L_d_w1はw1の変動に対するLに変動を示す
        d_L_d_ypred = -2 * (y_actual - y_pred)

        # ニューロンo1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # ニューロンh1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # ニューロンh2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # 重みとバイアスを更新
        # ニューロンh1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # ニューロンh2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # ニューロンo1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
        
      # 各ループ（epoch）ごとに損失を計算する
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_actuals, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# データセットを定義
data = np.array([
  [-3.75, -6.5], # 花子
  [8.25, 10.5],  # 太郎
  [5.25, 5.5],   # 裕太
  [-9.75, -6],   # 明美
])
all_y_actuals = np.array([
  1, # 花子
  0, # 太郎
  0, # 裕太
  1, # 明美
])

# NNに学習させる
network = NeuralNetwork()
network.train(data, all_y_actuals)

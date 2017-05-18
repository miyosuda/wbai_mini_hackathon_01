# WBAI 第一回ミニハッカソン コード

http://www.riken.jp/pr/press/2015/20150522_1/

こちらの論文の反響回路を元にして、同じ構造をDeep Learningで再現した場合の触覚の識別性能を強化学習を用いて調べてみました。

![network_graph](./doc/network.png)



| model_type  | model class                   | 説明                                             |
| ------------|-------------------------------|--------------------------------------------------|
| plain       | SomaticSimpleNetwork          | リカレント構造の無い一番シンプルなフルコネクトのネットワーク |  
| rnn         | SomaticRecurrentNetwork       | S1,M2間のリカレント構造を模したネットワーク              |
| rnn_action  | SomaticActionRecurrentNetwork | rnnに前回のアクションを入力として追加したネットワーク      |
| plain_cnn   | SomaticCNNNetwork             | 上のplainモデルににCNNを追加したもの                 |
| rnn_cnn     | SomaticRecurrentCNNNetwork    | 上のrnnモデルにCNNを追加したもの                    |

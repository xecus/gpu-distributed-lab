# 分散学習の理論：共通ノート

## 1. 学習ステップ

データxからlogitsを計算し、次トークンの交差エントロピーを最小化する。

\[
L=-\frac{1}{BS}\sum_{b,t}\log p_\theta(x_{b,t+1}\mid x_{b,\leq t}),\qquad
\theta\leftarrow\mathrm{AdamW}(\theta,\nabla_\theta L)
\]

TransformerのAttentionは `softmax(QK^T/sqrt(d_h))V`。未来トークンは因果マスクで見えなくする。
MLPは `W_2 GELU(W_1 x)`。本教材は位置Embedding＋LayerNorm＋標準MLPで、QwenのRoPE/RMSNorm/GQA/SwiGLUとは異なる。
この違いにより、基礎モデルのtokens/sをQwenのtokens/sへ直接換算してはいけない。

Forwardの演算にBF16を使っても、optimizer stateや保持パラメータがBF16とは限らない。
BF16は指数8bit・仮数7bit、FP32は指数8bit・仮数23bit。BF16はFP16より広い範囲を持つ。
本教材の通常モデルはFP32保持＋BF16 autocast、DeepSpeedはBF16重み＋FP32 master/Adam state。
BF16では通常GradScalerを使わない。演算精度・state精度を実際に確認すること。

## 2. データ並列とBatch

DPの各rankは同じモデルを保持し、異なるサンプルを処理する。

\[
G=b\times a\times D,\qquad
 g=\frac1D\sum_{r=0}^{D-1}g_r
\]

b＝micro batch/GPU、a＝gradient accumulation、D＝DP worker数。
平均lossを使うためmanual/DDP/FSDPでは各micro lossをaで割る。DeepSpeedはengineがaccumulationを管理するため二重に割らない。
DDPでは最後以外のmicro-stepを`no_sync()`にして通信を抑える。
各rankのbatch数が不均等なら単純なrank平均は正しいサンプル平均にならない。この教材は等サイズ。

TPでは同じサンプルを全rankが協調して計算する。TPサイズTをGへ掛けてはいけない。
本教材のTPは純粋なTPでDP=1。DP×TPを実現するには直交したDeviceMeshと、DP内Samplerが別途必要。

## 3. Collectiveの通信フロー

- AllReduce：各rankのベクトルを加算し、全rankが和を得る。
- ReduceScatter：各rankのベクトルを加算し、rankごとに和の一部分を保持する。
- AllGather：rankごとの部分を集め、全rankが全体を得る。
- Ring AllReduce：ReduceScatterをD−1回、AllGatherをD−1回の隣接交換で実現する。

サイズM bytesのRing AllReduceの1rank通信量は約`2(D−1)M/D`。
時間の粗いモデルは`2(D−1)α + 2(D−1)M/(Dβ)`。
αは通信開始latency、βは有効帯域。NCCLはRing以外のアルゴリズム・複数チャネル・通信と計算の重複も使う。
自作RingがNCCLの速さを再現することは目的ではない。

NCCLは通信ライブラリ、DDPは勾配同期を管理するPyTorchラッパー、ZeROはstate分割方式。
NCCLとDDPを代替手法として比較しない。DDP/ZeRO/FSDP/TPの通信にNCCLが使われる。

## 4. メモリの見積もり

Pパラメータ、D workers、BF16 weights/grads＋FP32 master＋FP32 Adam m/vを仮定。
重み2P、勾配2P、master4P、Adam8P bytes。activation、通信buffer、一時的AllGatherは別。

| 方式 | 1rankのモデルstate近似 bytes |
|---|---|
| DP | 16P |
| ZeRO-1 | 4P + 12P/D |
| ZeRO-2 | 2P + 14P/D |
| ZeRO-3 | 16P/D |

実際のnative FP32 AdamW保持＋BF16 autocastでは、DPはparameter4P＋gradient4P＋Adam8P＝16P。
FSDPもFP32 stateの分割は16P/D付近だが、mixed-precision gatherやlayerのwrap境界がピークを増やす。
これらはbytesでありGiBへの換算は2^30で割る。GBとGiBを混ぜない。
8Bで16P≈128GB、32Bで≈512GB。32Bを8rankへ分割してもstateだけで約64GB/rankなので、sequence・activation・一時領域でH100 80GBの上限に近づく。H200 141GB/B200 180GBでは余裕が増えるので、同じ設定でもOOMの境界が変わる。

Activationはbatch/sequence/layersに依存。attentionのS²項、SDPAの実装、activation checkpointingで変わる。
Checkpointingはforwardを再計算してactivationを減らすため、低VRAMでもstepが遅くなりうる。

## 5. 各方式の通信と用途

| 方式 | 主な通信 | 選択の起点 |
|---|---|---|
| DDP | backward中のbucket AllReduce | モデル＋stateが1GPUに収まる |
| ZeRO-1 | state更新・weight同期 | optimizer stateが大きい |
| ZeRO-2 | 勾配ReduceScatter・weight同期 | 勾配も分割したい |
| ZeRO-3/FSDP | layer前のAllGather・勾配ReduceScatter | 重みも分割しないと収まらない |
| TP | layer内のAllReduce等 | 大きいlayerを分割する／高速ノード内リンク |

FSDPは本教材で学習しやすいFSDP1のAPIを固定使用。最新のFSDP2/DTensor APIと混ぜない。
ZeRO-3とFSDPは似たstate分割でも、wrap境界・prefetch・buffer・optimizer精度が違うと性能は変わる。

## 6. TPのAttention/MLP分割

列方向分割（Colwise）：Q/K/Vの出力heads、MLPの拡張次元をrankへ分配。
行方向分割（Rowwise）：出力projectionとMLP縮約を分配し、部分和をAllReduceしてhidden全体へ戻す。
本教材はQ/K/Vを別のLinearにし、rankのlocal_heads=total_heads/TとしてSDPAを計算する。
Embedding、LayerNorm、LM headは複製。全パラメータが1/Tになる実装ではない。

## 7. スケーリング

Strong scaling：Gとsequence、総更新数を固定。`speedup=T_base/T_D`、`efficiency=speedup/(D/D_base)`。
Weak scaling：micro batchとaccumを固定しGを増やす。`efficiency=(throughput_D/throughput_base)/(D/D_base)`。
Weak scalingでloss曲線が変わるのは、通信だけでなくGと処理token数も変わるため。
モデルstateの分割方式を変えた比較を、同じ手法の純粋なスケーリングと呼ばない。

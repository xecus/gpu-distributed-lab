# =============================================================================
# scripts/efa_env.sh — EFA (Elastic Fabric Adapter) 用の環境変数を設定する
#
# 目的:
#   複数ノード学習で、NCCL がノード間通信に EFA (AWS の高速ネットワーク) を
#   使うように環境変数を整える。
#
# 使い方 (実行ではなく source する。現在のシェルに環境変数を残すため):
#   source scripts/efa_env.sh
#
# 注意:
#   EFA 対応の DLAMI 上でだけ使うこと。下のパスが実際に存在するか必ず確認してから source する。
#
# 読み取る環境変数:
#   PATH, LD_LIBRARY_PATH (既存の値の前に EFA 関連のパスを追加する)
# =============================================================================

# EFA ツール (fi_info など) と Amazon 版 Open MPI のコマンドを優先して使う。
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH

# libfabric (EFA) と aws-ofi-nccl プラグインのライブラリを見つけられるようにする。
# aws-ofi-nccl は NCCL から libfabric 経由で EFA を使うための橋渡しプラグイン。
# AMI によってインストール先が違うので、候補のパスを全部並べている。
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH:-}

# libfabric のプロバイダとして EFA を明示的に選ぶ (TCP などへの意図しないフォールバックを防ぐ)。
export FI_PROVIDER=efa

# NCCL の初期化ログを出す。どのネットワークを使ったかをログで確認するため。
export NCCL_DEBUG=INFO

# NCCL の通信エラーやハングを検出したら、プロセスを止めてエラーにする
# (黙って固まり続けて GPU 代を浪費するのを防ぐ)。
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# NCCL_ALGO, NCCL_PROTO, FI_EFA_USE_DEVICE_RDMA やトポロジ XML は強制しないこと。
# OFI プラグインがプラットフォームに合った設定を自動で選ぶ。
# ログに "NET/OFI" と "provider efa" が出ていれば EFA が使われている。

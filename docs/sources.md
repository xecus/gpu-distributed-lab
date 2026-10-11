# 公式一次資料・検証時点

確認日：2026-10-10。実装・依存版・AMI・提供状況は更新されるため、repo/modelはcommit SHAを固定。
教材の理論と実装は教育用に構成し、速度やメモリの実測値を公式値から転記していない。

| 対象 | 公式資料 |
|---|---|
| AWS p5/p5e/p5en | https://aws.amazon.com/ec2/instance-types/p5/ |
| AWS p6 | https://aws.amazon.com/ec2/instance-types/p6/ |
| EFA付きDLAMI起動 | https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-efa-launching.html |
| EFA/NCCL使用 | https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-efa-using.html |
| 複数network cards | https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-acc-inst-types.html |
| AWS OFI NCCL | https://github.com/aws/aws-ofi-nccl |
| PyTorch DDP | https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html |
| PyTorch TP | https://docs.pytorch.org/docs/2.7/distributed.tensor.parallel.html |
| PyTorch FSDP1 | https://docs.pytorch.org/docs/2.7/fsdp.html |
| ZeROの原理 | https://www.deepspeed.ai/tutorials/zero/ |
| DeepSpeed保存/復元 | https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html |
| Qwen3-8B | https://huggingface.co/Qwen/Qwen3-8B |
| Qwen3-32B | https://huggingface.co/Qwen/Qwen3-32B |
| WikiText | https://huggingface.co/datasets/Salesforce/wikitext |
| GR00T | https://github.com/NVIDIA/Isaac-GR00T |
| GR00Tデータ | https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md |
| GR00Tカスタム微調整 | https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md |
| GR00T分散入口 | https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/experiment/experiment.py |
| NCCL tests | https://github.com/NVIDIA/nccl-tests |

GR00Tは現行sourceでWORLD_SIZE/LOCAL_RANK、process group、num_gpus consistencyを確認した。
launch_finetuneの単一node向け説明とmulti-node実行検証は別。固定commitの静的結果はresults/groot_static_audit.json。
FSDPの最新stableページはFSDP2へ更新されているため、この教材のFSDP1は2.7の版付きdocsを読む。

"""toy Transformer と Qwen 実験で共通に使う、計測可能な学習ループ。

レッスン間で「並列化の方法だけ」が変わるように、1 本のスクリプトで全モードを扱う:
  single     1 プロセスのみ (分散なし)
  manual     モデルを複製し、勾配を明示的な all_reduce で平均する (A06)
  ddp        torch の DistributedDataParallel (A07, A13, A14)
  zero1/2/3  DeepSpeed ZeRO の各 stage (A08-A10, B01-B03)
  fsdp       PyTorch FSDP1、Transformer ブロック単位で wrap (A11, B03)
  tp         toy モデルの attention + MLP を tensor parallel 化 (A12)

出力: rank ごとのメトリクス CSV、環境情報 JSON、再開可能なチェックポイント。
AWS の API は一切呼ばない。
"""
import argparse,contextlib,csv,json,os,time,platform,subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from lab.model import Transformer,Block,apply_tp


def parser():
    """コマンドライン引数の定義。"""
    p = argparse.ArgumentParser()

    # --- 並列化モードとモデル ---
    p.add_argument(
        '--mode',
        choices=['single', 'manual', 'ddp', 'zero1', 'zero2', 'zero3', 'fsdp', 'tp'],
        default='single',
    )
    p.add_argument('--model', default='toy', help='toy or Hugging Face repo id')
    p.add_argument('--revision', default='main', help='pin to HF commit for reproducibility')
    p.add_argument('--lora', action='store_true')
    p.add_argument('--tokens', help='uint32 token memmap produced by lab.prepare_data')

    # --- ステップ数・バッチサイズ (global batch = batch * accum * DP 数) ---
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--seq', type=int, default=256)

    # --- toy モデルの形状 ---
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)

    # --- 最適化・乱数 ---
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=123)

    # --- 出力とチェックポイント ---
    p.add_argument('--out', default='runs/debug')
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--keep-checkpoints', type=int, default=2)
    p.add_argument('--resume', help='step_N checkpoint directory (same world size/mode)')

    # --- オプション機能 ---
    p.add_argument('--activation-checkpointing', action='store_true')
    p.add_argument('--profile', action='store_true', help='one step trace, excluded from timing')
    p.add_argument(
        '--sampler',
        action='store_true',
        help='use DistributedSampler over fixed indexed corpus windows',
    )
    p.add_argument('--export-hf', help='FSDP resume + gather to HF directory, without training')
    p.add_argument('--cpu', action='store_true', help='CPU smoke tests; single/manual/ddp only')
    return p


def main():
    # =====================================================================
    # 引数解析
    # =====================================================================
    a = parser().parse_args()
    if a.steps <= 0 or a.warmup < 0 or min(a.batch, a.accum, a.seq) < 1:
        raise ValueError('invalid sizes')

    # =====================================================================
    # 分散初期化
    # =====================================================================
    # torchrun が RANK / WORLD_SIZE / LOCAL_RANK を環境変数で渡してくれる。
    # 既定値があるので、素の `python -m lab.train` でも 1 プロセスとして動く。
    #   rank  : 全プロセス中の通し番号 (0 .. world-1)
    #   world : 全プロセス数 (GPU 実行時は使う GPU の総数)
    #   local : このノード内での番号 (= 使う GPU の番号)
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if a.mode == 'single' and world != 1:
        raise ValueError('single requires 1 process')
    if a.cpu and a.mode not in ['single', 'manual', 'ddp']:
        raise ValueError('GPU required for this mode')

    # 各プロセスは自分の LOCAL_RANK 番の GPU だけを使う。
    device = torch.device('cpu' if a.cpu else f'cuda:{local}')
    if not a.cpu:
        torch.cuda.set_device(device)

    # 複数プロセス時、または DeepSpeed / FSDP (1 プロセスでも process group が必要) のときに
    # 通信バックエンドを初期化する。GPU 間は NCCL、CPU テストでは gloo を使う。
    if world > 1 or a.mode.startswith('zero') or a.mode == 'fsdp':
        dist.init_process_group('gloo' if a.cpu else 'nccl')
    active = dist.is_initialized()

    # 全 rank で同じ seed → モデルの初期重みが全 rank で一致する。
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds_config = None
    hf_ds = None

    # ZeRO: optimizer・BF16・勾配累積はすべて DeepSpeed が管理する。
    # そのためここでは設定 dict を組み立てるだけで、optimizer は自前で作らない。
    if a.mode.startswith('zero'):
        ds_config = {
            'train_micro_batch_size_per_gpu': a.batch,
            'gradient_accumulation_steps': a.accum,
            # DeepSpeed は「micro batch * accum * world == train_batch_size」を検証する
            'train_batch_size': a.batch * a.accum * world,
            'bf16': {'enabled': True},
            # 'zero1' → 1, 'zero2' → 2, 'zero3' → 3
            'zero_optimization': {'stage': int(a.mode[-1])},
            'steps_per_print': 1000000,
            'optimizer': {
                'type': 'AdamW',
                'params': {
                    'lr': a.lr,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8,
                    'weight_decay': 0.01,
                },
            },
        }
        if a.model != 'toy' and a.mode == 'zero3':
            # HF モデルを ZeRO-3 で読む場合、このオブジェクトが生きている間は
            # from_pretrained が重みを最初から分割 (partition) した状態でロードしてくれる。
            from transformers.integrations import HfDeepSpeedConfig
            hf_ds = HfDeepSpeedConfig(ds_config)  # from_pretrained の間は参照を保持しておく

    # =====================================================================
    # モデル構築
    # =====================================================================
    if a.model == 'toy':
        model = Transformer(dim=a.dim, layers=a.layers, heads=a.heads, seq=a.seq)
    else:
        if not a.tokens:
            raise ValueError('real models require --tokens, synthetic ids are not a corpus')
        from transformers import AutoModelForCausalLM,AutoConfig

        # FSDP: 実際の重みをロードするのは rank0 だけ。他の rank は 'meta' デバイス上に
        # 形だけのモデル (メモリを確保しない) を作り、FSDP の sync_module_states で
        # rank0 から重みを受け取る。こうするとホスト RAM 上の重みは
        # ノード内 GPU 数ぶん (例: 8) ではなく 1 コピーで済む。
        if a.mode == 'fsdp' and rank != 0:
            config = AutoConfig.from_pretrained(a.model, revision=a.revision)
            with torch.device('meta'):
                model = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa')
        else:
            model = AutoModelForCausalLM.from_pretrained(
                a.model,
                revision=a.revision,
                torch_dtype=torch.float32,
                attn_implementation='sdpa',
            )

        # LoRA: 元の重みは凍結し、小さな低ランク行列だけを学習する。
        if a.lora:
            from peft import LoraConfig,get_peft_model
            model = get_peft_model(
                model,
                LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    task_type='CAUSAL_LM',
                    target_modules=[
                        'q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj',
                    ],
                ),
            )

        # Activation checkpointing: forward の中間活性を保存せず backward で再計算し、
        # 計算量と引き換えに GPU メモリを節約する。
        if a.activation_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
            model.config.use_cache = False  # KV cache は再計算と両立しないので無効化

    # toy モデルの activation checkpointing は各 Block を wrapper で包んで実現する。
    if a.model == 'toy' and a.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
        for i, b in enumerate(model.blocks):
            model.blocks[i] = checkpoint_wrapper(b)

    # 並列化ラッパーで包む前に、全パラメータ数 (LoRA で凍結した分も含む) を数えて
    # 環境 JSON に記録する。
    # 注意: 実モデル + zero3 では from_pretrained 時点で既に分割済み (param.data が空) のため
    # numel() が 0 になり、この値は正しくない (本来のサイズは p.ds_numel にある)。
    params = sum(p.numel() for p in model.parameters())

    # =====================================================================
    # 並列化ラッパー
    # =====================================================================
    # FSDP は optimizer 作成「前」に wrap する必要がある (optimizer は分割後の
    # パラメータを参照しなければならないため)。ZeRO は optimizer を自分で作る。
    if a.mode == 'fsdp':
        from functools import partial
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Transformer ブロック 1 個を FSDP の 1 単位 (all_gather / 解放の単位) にする。
        wrap_class = Block
        if a.model != 'toy':
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
            wrap_class = Qwen3DecoderLayer
        model = FSDP(
            model,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={wrap_class},
            ),
            # rank0 の重みを全 rank へ broadcast する (meta で作った rank はこれで重みを得る)
            sync_module_states=True,
            # meta テンソルを持つサブモジュールごとに FSDP が呼ぶ (だから recurse=False):
            # 未初期化のまま実デバイスに確保し、値は sync_module_states で rank0 から上書きされる。
            # rank0 や toy モデルには meta が無いので呼ばれない。
            param_init_fn=lambda module:module.to_empty(device=device,recurse=False),
            device_id=device,
            # 計算・勾配通信・バッファは BF16 で行う (マスター重みは FP32 のまま)
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
        )
    elif not a.mode.startswith('zero'):
        # ZeRO は deepspeed.initialize、FSDP は device_id でデバイス転送されるので、
        # それ以外のモードだけここで移す。
        model.to(device)

    # Tensor parallel: 各 Block の attention (q/k/v/proj) と MLP (fc1/fc2) の重みを world 個に
    # 分割し、全 rank で 1 つの Block を協調計算する。embedding・norm・head は複製のまま。
    if a.mode == 'tp':
        if world < 2:
            raise ValueError('TP requires >=2 GPU processes')
        if a.model != 'toy':
            raise ValueError('TP lesson supports the toy model only')
        # 注: 下のエラーメッセージは主語が逆。正しくは
        # 「world size が FFN 次元 (4*dim) を割り切る必要がある」。
        if (4 * a.dim) % world:
            raise ValueError('FFN dim must divide world size')
        model = apply_tp(model, world)

    # DDP: backward 中にバケット単位で勾配を all_reduce し、計算と通信を重ねる。
    if a.mode == 'ddp':
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=None if a.cpu else [local])

    engine = None
    if a.mode.startswith('zero'):
        import deepspeed
        # DeepSpeed が model / optimizer / BF16 / 勾配累積をまとめて管理する engine を返す。
        engine, opt, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=ds_config,
        )
        model = engine
    else:
        # TP では DTensor (分割された q/k/v/proj/fc1/fc2) と通常の tensor (複製された
        # embedding・norm・head) が混在する。multi-tensor の foreach カーネルは型の混ざった
        # リストを受け付けないため、foreach=False で 1 パラメータずつ更新させる。
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=a.lr,
            foreach=False if a.mode == 'tp' else None,
        )

    # manual モード: seed が同じでも念のため rank0 の重みを全 rank に broadcast し、
    # 全レプリカの初期値を確実に一致させる (DDP は内部で同じことをやっている)。
    if a.mode == 'manual' and active:
        for p in model.parameters():
            dist.broadcast(p.data, 0)

    # =====================================================================
    # データ生成
    # =====================================================================
    # --tokens 指定時は、事前にトークナイズ済みのコーパスを memmap で読む
    # (ファイル全体をメモリに載せず、必要な部分だけ OS がページインする)。
    tokens = None
    if a.tokens:
        import numpy as np
        tokens = np.memmap(a.tokens, dtype=np.uint32, mode='r')
        if len(tokens) < a.seq + 2:
            raise ValueError('token corpus too small')

    # コーパスの SHA-256 を計算し、再開時に「同じデータか」を検証できるようにする。
    data_fingerprint = None
    if a.tokens:
        import hashlib
        h = hashlib.sha256()
        with open(a.tokens, 'rb') as file:
            # 8 MiB ずつ読んでハッシュに流し込む (巨大ファイルでもメモリを食わない)
            for chunk in iter(lambda:file.read(8*1024*1024), b''):
                h.update(chunk)
        data_fingerprint = h.hexdigest()

    sampler_cache = {}

    # バッチは (seed, step, micro, rank, world) だけで決まる決定的な関数 (TP では rank=0 扱い)。
    # dataloader の状態を保存しなくても、再開時にまったく同じデータ列を再現できる。
    def batch(step, micro):
        # TP の worker は全員「同じ」バッチを見る (1 つのモデルを分担して計算するため)。
        # DP の worker は rank ごとに異なるデータ列を見る (データを分担するため)。
        data_rank = 0 if a.mode == 'tp' else rank

        # --- DistributedSampler 版: 固定インデックス集合を rank 間で重複なく分配 ---
        if a.sampler:
            from torch.utils.data.distributed import DistributedSampler
            size = 65536
            replicas = 1 if a.mode == 'tp' else world
            sampler = DistributedSampler(
                range(size),
                num_replicas=replicas,
                rank=data_rank,
                seed=a.seed,
                drop_last=True,
            )
            per_rank = len(sampler)
            # この rank が通算で何サンプル目か (rank ローカル) → そこから epoch 番号を求める。
            # 注: per_rank が batch の倍数でないと、epoch 境界をまたぐバッチの末尾は
            # 次 epoch ではなく現 epoch の先頭に折り返す (% per_rank) ため、数サンプルが重複する。
            offset = (step * a.accum + micro) * a.batch
            epoch = offset // per_rank
            # epoch ごとのシャッフル結果をキャッシュ (set_epoch で並びが変わる)
            if epoch not in sampler_cache:
                sampler.set_epoch(epoch)
                sampler_cache.clear()
                sampler_cache[epoch] = list(sampler)
            indices = [sampler_cache[epoch][(offset + i) % per_rank] for i in range(a.batch)]
            if tokens is None:
                # 合成データ: インデックスから決定的に開始値を作り、連番 (mod 256) を並べる
                starts = torch.tensor([((i*2654435761+a.seed)%256) for i in indices])[:, None]
                x = (starts + torch.arange(a.seq + 1)) % 256
            else:
                import numpy as np
                # インデックス → コーパス上の開始位置 (乗算ハッシュで散らす)
                positions = [(i*2654435761+a.seed)%(len(tokens)-a.seq-1) for i in indices]
                x = torch.tensor(
                    np.stack([tokens[i:i+a.seq+1] for i in positions]).astype('int64')
                )
            # 入力 x[:, :-1] と、1 トークンずらした正解 x[:, 1:] (次トークン予測)
            x = x.to(device)
            return x[:, :-1], x[:, 1:]

        # --- 既定版: (seed, step, micro, rank) から専用の乱数生成器を作る ---
        g = torch.Generator().manual_seed(a.seed+step*100003+micro*997+data_rank*10000019)
        if tokens is None:
            starts = torch.randint(0, 256, (a.batch, 1), generator=g)
            x = (starts + torch.arange(a.seq + 1)) % 256
        else:
            positions = torch.randint(0, len(tokens)-a.seq-1, (a.batch,), generator=g).tolist()
            import numpy as np
            x = torch.tensor(np.stack([tokens[i:i+a.seq+1] for i in positions]).astype('int64'))
        x = x.to(device)
        return x[:, :-1], x[:, 1:]

    # =====================================================================
    # チェックポイント保存
    # =====================================================================
    # state_dict を取り出す / 読み込むときのコンテキスト。
    def state_context():
        if a.mode == 'fsdp':
            from torch.distributed.fsdp import (
                StateDictType,
                LocalStateDictConfig,
                FullyShardedDataParallel as FSDP,
            )
            # LOCAL_STATE_DICT: 各 rank が自分のシャードだけを保存する (集約しないので速い)。
            # offload_to_cpu=True は torch 2.7 で壊れる: ShardedTensor.cpu() が requires_grad を
            # 落としてメタデータ検証に失敗する。
            # どのみち torch.save が GPU シャードをホストにコピーする。
            return FSDP.state_dict_type(model,StateDictType.LOCAL_STATE_DICT,LocalStateDictConfig())
        return contextlib.nullcontext()

    # DDP を剥がして中身のモデルを返す: state_dict のキーに 'module.' 接頭辞が付かないようにする。
    def bare():
        return model.module if a.mode == 'ddp' else model

    # 各 rank が自分のシャード (rank{r}.pt または DeepSpeed のファイル群) と
    # meta_rank{r}.json を書き出す。
    def checkpoint(step):
        dest = out / f'step_{step}'
        dest.mkdir(exist_ok=True)
        meta = {
            'step': step,
            'world': world,
            'mode': a.mode,
            'seed': a.seed,
            'args': vars(a),
            'data_sha256': data_fingerprint,
        }
        if engine:
            # DeepSpeed は分割された optimizer 状態も含めて自前の形式で保存する。
            engine.save_checkpoint(str(dest), tag='ds', client_state=meta)
        else:
            with state_context():
                sd = bare().state_dict()
            # 乱数状態も保存し、再開後に同じ乱数列を続けられるようにする。
            torch.save(
                {
                    'model': sd,
                    'optimizer': opt.state_dict(),
                    'meta': meta,
                    'rng': torch.get_rng_state(),
                    'cuda_rng': None if a.cpu else torch.cuda.get_rng_state(),
                },
                dest / f'rank{rank}.pt',
            )
        (dest / f'meta_rank{rank}.json').write_text(json.dumps(meta, indent=2))

        # 全 rank が書き終わるまで待ってから古いチェックポイントを消す。
        if active:
            dist.barrier()
        # 削除は各ノードの local rank 0 だけが行う (同じディレクトリを複数プロセスで消さない)。
        if local == 0 and a.keep_checkpoints > 0:
            import shutil
            dirs = sorted(
                [d for d in out.glob('step_*') if d.is_dir() and d.name[5:].isdigit()],
                key=lambda d:int(d.name[5:]),
            )
            for old in dirs[:-a.keep_checkpoints]:
                shutil.rmtree(old, ignore_errors=True)
        # 削除が終わるまで他の rank を先に進ませない。
        if active:
            dist.barrier()

    # =====================================================================
    # チェックポイント再開
    # =====================================================================
    start = 0
    # world size・mode・コーパス SHA-256・モデル構成やデータ列を決める引数 (下のリスト) の
    # どれかが変わっていたら再開を拒否する。lr・steps・warmup などは変更してよい。
    # (シャードの分割方法やデータ列が変わり、黙って誤った学習になるのを防ぐ)
    if a.resume:
        meta = json.loads((Path(a.resume) / f'meta_rank{rank}.json').read_text())
        if (meta['world'], meta['mode']) != (world, a.mode):
            raise ValueError('resume topology/mode mismatch')
        if meta.get('data_sha256') != data_fingerprint:
            raise ValueError('token corpus checksum changed')
        old = meta['args']
        for key in [
            'model', 'revision', 'lora', 'tokens', 'batch', 'accum', 'seq',
            'dim', 'layers', 'heads', 'seed', 'activation_checkpointing', 'sampler',
        ]:
            if old.get(key) != vars(a).get(key):
                raise ValueError(f'resume setting differs: {key}')

        if engine:
            loaded, _ = engine.load_checkpoint(a.resume, tag='ds')
            if loaded is None:
                raise RuntimeError('DeepSpeed checkpoint not found')
        else:
            sd = torch.load(
                Path(a.resume) / f'rank{rank}.pt',
                map_location='cpu',
                weights_only=False,
            )
            with state_context():
                bare().load_state_dict(sd['model'])
            opt.load_state_dict(sd['optimizer'])
            torch.set_rng_state(sd['rng'])
            if not a.cpu:
                torch.cuda.set_rng_state(sd['cuda_rng'])
        start = meta['step']

    # --export-hf: FSDP のシャードを rank0 に集めて Hugging Face 形式で保存し、学習せず終了。
    if a.export_hf:
        if a.mode != 'fsdp' or not a.resume or a.model == 'toy':
            raise ValueError('export-hf requires real-model FSDP resume')
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )
        # FULL_STATE_DICT: 全シャードを all_gather して完全な重みを作る。
        # GPU メモリ溢れを避けるため CPU に退避し、rank0 だけが結果を保持する。
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state = model.state_dict()
        if rank == 0:
            model.module.save_pretrained(a.export_hf, state_dict=state)
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(a.model, revision=a.revision).save_pretrained(a.export_hf)
        if active:
            dist.barrier()
            dist.destroy_process_group()
        return

    # =====================================================================
    # 学習ループ (1 回の optimizer 更新)
    # =====================================================================
    # GPU は非同期に動くので、時間計測の前後で「GPU の処理完了」を待つ必要がある。
    def sync():
        if not a.cpu:
            torch.cuda.synchronize(device)

    # optimizer の更新 1 回 = accum 個の micro-batch。
    # DDP は最後以外の micro-batch で勾配 all_reduce をスキップする (no_sync)。
    # manual モードは勾配を累積し終えた後に明示的に平均する。
    def update(step):
        if not engine:
            opt.zero_grad(set_to_none=True)
        losses = []
        for micro in range(a.accum):
            x, y = batch(step, micro)
            # no_sync(): この中の backward では勾配をローカルに足し込むだけで通信しない。
            # 最後の micro-batch でだけ all_reduce すれば結果は同じで、通信量が 1/accum になる。
            no_sync = (
                model.no_sync()
                if a.mode == 'ddp' and micro < a.accum - 1
                else contextlib.nullcontext()
            )
            with no_sync:
                # GPU では BF16 autocast で行列演算を高速化する。
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=not a.cpu,
                ):
                    result = model(x) if a.model == 'toy' else model(input_ids=x)
                    logits = result if a.model == 'toy' else result.logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                losses.append(loss.detach().float())
                if engine:
                    # DeepSpeed は engine.backward 内で loss を自動で 1/accum する
                    # (下の手動割り算は不要)。また micro-step 数を数え、accum 回目の
                    # step() でだけ optimizer 更新と zero_grad を行う。
                    engine.backward(loss)
                    engine.step()
                else:
                    # accum で割っておくと、累積後の勾配が「大きなバッチの平均勾配」と等しくなる。
                    (loss / a.accum).backward()
        if not engine:
            if a.mode == 'manual' and active:
                # 手動データ並列: 全 rank の勾配を合計 (all_reduce) して world で割り、平均にする。
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad.div_(world)
            opt.step()
        return torch.stack(losses).mean()

    # =====================================================================
    # 環境情報の記録
    # =====================================================================
    meta = {
        'args': vars(a),
        'world': world,
        'rank': rank,
        'host': platform.node(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'data_sha256': data_fingerprint,
        'parameter_count': params,
        'gpu': None if a.cpu else torch.cuda.get_device_name(device),
    }
    try:
        meta['git'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        meta['git'] = None
    (out / f'environment_rank{rank}.json').write_text(json.dumps(meta, indent=2))

    # =====================================================================
    # 計測
    # =====================================================================
    fields = [
        'step', 'rank', 'world', 'mode', 'global_batch', 'loss',
        'step_s', 'tokens_s', 'allocated_gib', 'reserved_gib',
    ]
    # 計測方法: 開始前に barrier + synchronize、終了後に synchronize し、
    # 全 rank の最大値を step 時間とする。
    #   barrier     → 全 rank が同時にスタートラインに立つ (前ステップの遅れを持ち越さない)
    #   synchronize → 発行済みの GPU カーネルが本当に終わるまで待つ (非同期実行のため)
    #   max         → 分散学習は一番遅い rank に律速されるので、最大値が実際の step 時間
    with (out / f'metrics_rank{rank}_from{start}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(start, a.steps):
            if active:
                dist.barrier()
            sync()
            if not a.cpu:
                torch.cuda.reset_peak_memory_stats(device)

            t = time.perf_counter()
            loss = update(step)
            sync()
            elapsed = time.perf_counter() - t

            # 計測区間は update() と直後の cuda.synchronize まで (勾配通信は含む)。
            # 開始前の barrier、この下の時間の MAX 集約、loss の all_reduce は step 時間に含めない。
            stats = torch.tensor([elapsed], device=device)
            if active:
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            # DP では rank ごとに loss が違うので平均する (TP は全 rank 同じ loss なので不要)。
            if active and a.mode != 'tp':
                dist.all_reduce(loss)
                loss /= world

            # global batch: TP は全 rank が同じデータを見るので DP 数は 1 として数える。
            gb = a.batch * a.accum * (1 if a.mode == 'tp' else world)

            # warmup 中も optimizer は更新する (記録しないだけ)。
            # --steps は warmup を含む全更新回数。
            # warmup 後のステップだけを記録し、CUDA/cuBLAS 初期化・NCCL 通信路確立・メモリ確保など
            # の初回ノイズを除く。step は通算番号なので、warmup を過ぎた地点からの再開では
            # 除外されない。
            if step >= a.warmup:
                writer.writerow(dict(
                    step=step + 1,
                    rank=rank,
                    world=world,
                    mode=a.mode,
                    global_batch=gb,
                    loss=loss.item(),
                    step_s=stats.item(),
                    tokens_s=gb * a.seq / stats.item(),
                    allocated_gib=0 if a.cpu else torch.cuda.max_memory_allocated(device) / 2**30,
                    reserved_gib=0 if a.cpu else torch.cuda.max_memory_reserved(device) / 2**30,
                ))
                f.flush()

            if rank == 0 and (step + 1) % 10 == 0:
                print(
                    f'step={step+1} loss={loss.item():.4f} seconds={stats.item():.4f}',
                    flush=True,
                )
            if a.save_every > 0 and (step + 1) % a.save_every == 0:
                checkpoint(step + 1)

        # 最終ステップが定期保存と重ならなかった場合は、最後に 1 回保存する。
        if a.steps > start and (a.save_every <= 0 or a.steps % a.save_every):
            checkpoint(a.steps)

    # =====================================================================
    # プロファイル
    # =====================================================================
    # 計測ループの後に 1 回だけ追加の更新をトレースする。
    # こうすればトレースのオーバーヘッドが step_s に混入しない。
    if a.profile:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if not a.cpu:
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=acts,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            update(a.steps)
        # chrome://tracing や Perfetto で開ける形式で出力する。
        prof.export_chrome_trace(str(out / f'trace_rank{rank}.json'))

    if active:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

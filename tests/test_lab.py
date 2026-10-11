# =============================================================================
# tests/test_lab.py — 教材コード (lab/) と AWS 安全装置の CPU テスト
#
# 目的:
#   GPU を使わず、CPU だけで「分散学習の正しさ」に関わる性質を確認する。
#   高価な GPU ノードを借りる前に、ロジックの誤りをローカルで見つけるのが狙い。
#
# 実行例:
#   OMP_NUM_THREADS=1 python -m unittest discover -s tests
#   OMP_NUM_THREADS=1 LAB_TEST_NO_SOCKETS=1 python -m unittest discover -s tests
#
# 読み取る環境変数:
#   LAB_TEST_NO_SOCKETS=1 : Gloo の TCP ソケットが使えない環境 (一部のコンテナなど) で、
#                           2 プロセスを起動するテストをスキップする。
# =============================================================================
import os,sys,tempfile,subprocess,unittest,json
from pathlib import Path

import torch

from lab.model import Transformer

# リポジトリのルート (tests/ の 1 つ上)。サブプロセスの cwd に使う。
ROOT=Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):

    def run_train(self,out,mode='single',steps=4,resume=None):
        """lab.train を小さな設定・CPU でサブプロセスとして実行するヘルパー。

        mode が 'manual' / 'ddp' のときは torchrun (torch.distributed.run) で
        2 プロセスを起動し、それ以外は 1 プロセスで `python -m lab.train` を実行する。
        resume を渡すと、そのチェックポイントから学習を再開する。
        """
        cmd=[sys.executable]

        # 分散モードは torchrun 経由で 2 プロセス起動する。
        # ポートを固定 (29501) し、ローカルホストでランデブーさせる。
        if mode in ['manual','ddp']:
            cmd+=[
                '-m','torch.distributed.run',
                '--master_addr=127.0.0.1','--master_port=29501',
                '--nproc_per_node=2','--module',
            ]
        else:
            cmd+=['-m']

        # テスト用のごく小さなモデル・データ設定 (数秒で終わる大きさ)。
        # --save-every 2 なので step_2 と step_4 にチェックポイントが保存される。
        cmd+=[
            'lab.train','--cpu','--mode',mode,
            '--dim','32','--layers','1','--heads','4','--seq','16','--batch','2',
            '--steps',str(steps),'--warmup','0','--save-every','2','--out',str(out),
        ]

        if resume:
            cmd+=['--resume',str(resume)]

        # OMP_NUM_THREADS=1 でスレッド数を固定し、結果のばらつきと CPU の取り合いを防ぐ。
        subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            env={**os.environ,'OMP_NUM_THREADS':'1'},
            capture_output=True,
            timeout=120,
        )

    def load(self,path):
        """チェックポイント (rank ごとの .pt) を CPU 上に読み込むヘルパー。"""
        return torch.load(path,map_location='cpu',weights_only=False)

    def test_causal_and_gradients(self):
        """因果マスクが効いていること、全パラメータに勾配が流れることを確認する。"""
        torch.manual_seed(1)
        m=Transformer(dim=32,layers=1,heads=4,seq=16)

        # x と y は先頭 8 トークンが同じで、後半 8 トークンだけが異なる入力。
        x=torch.randint(0,256,(2,16))
        y=x.clone()
        y[:,8:]=(y[:,8:]+1)%256

        # 因果的 (未来を見ない) モデルなら、前半 8 位置の出力は後半の違いに影響されない。
        # ここが一致しなければ、未来のトークンが漏れている (マスクのバグ)。
        torch.testing.assert_close(m(x)[:,:8],m(y)[:,:8])

        # 適当な損失で逆伝播し、すべてのパラメータに勾配があることを確認する。
        # 勾配が None のパラメータがあると、DDP などで「未使用パラメータ」エラーの原因になる。
        m(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_resume_exact_and_loss(self):
        """途中保存から再開した学習が、通しで学習した結果とビット単位で一致することを確認する。"""
        with tempfile.TemporaryDirectory() as tmp:
            full=Path(tmp)/'full'
            split=Path(tmp)/'split'

            # full : 0 → 4 ステップを通しで学習
            # split: 0 → 2 ステップで止め、step_2 から 4 ステップ目まで再開
            self.run_train(full,steps=4)
            self.run_train(split,steps=2)
            self.run_train(split,steps=4,resume=split/'step_2')

            a=self.load(full/'step_4/rank0.pt')
            b=self.load(split/'step_4/rank0.pt')

            # 許容誤差 0 で完全一致を要求する。モデルだけでなく、オプティマイザ状態・
            # 乱数状態が正しく復元され、データ列が step から決定的に再現されていないと
            # 一致しない。(CPU・single モードでの確認。分散モードの再開は acceptance.sh で確認)
            for k in a['model']:
                torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)

    @unittest.skipIf(
        os.environ.get('LAB_TEST_NO_SOCKETS')=='1',
        'container blocks Gloo TCP sockets; run on AWS host',
    )
    def test_manual_matches_ddp(self):
        """手書きの勾配 all-reduce (manual) が PyTorch DDP と同じ結果になることを確認する。"""
        with tempfile.TemporaryDirectory() as tmp:
            manual=Path(tmp)/'manual'
            ddp=Path(tmp)/'ddp'

            # どちらも 2 プロセス (Gloo バックエンド) で 4 ステップ学習する。
            self.run_train(manual,mode='manual')
            self.run_train(ddp,mode='ddp')

            a=self.load(manual/'step_4/rank0.pt')
            b=self.load(ddp/'step_4/rank0.pt')

            # 浮動小数点の加算順序の違いはあり得るので、ごく小さな誤差は許容する。
            # これが通れば「DDP の中身は勾配の平均を取っているだけ」という理解が確認できる。
            for k in a['model']:
                torch.testing.assert_close(a['model'][k],b['model'][k],rtol=1e-5,atol=1e-6)

    def test_average_gradient_equals_global_batch(self):
        """データ並列の原理: ミニバッチを分割した勾配の平均 = 全体バッチの勾配 を確認する。"""
        import copy

        # m と n は同じ初期重みを持つ 2 つのモデル。
        torch.manual_seed(9)
        m=Transformer(dim=32,layers=1,heads=4,seq=16)
        n=copy.deepcopy(m)

        x=torch.randint(0,256,(4,16))
        target=torch.randint(0,256,(4,16))

        # (1) 4 サンプル全体で 1 回だけ損失を計算して逆伝播する (1 GPU 相当)。
        loss=torch.nn.functional.cross_entropy(m(x).reshape(-1,256),target.reshape(-1))
        loss.backward()

        # (2) 2 サンプルずつに分け、別々に勾配を求める (2 GPU にデータを分けた状況を模擬)。
        partial=[]
        for shard in [slice(0,2),slice(2,4)]:
            n.zero_grad()
            torch.nn.functional.cross_entropy(
                n(x[shard]).reshape(-1,256),
                target[shard].reshape(-1),
            ).backward()
            partial.append([p.grad.clone() for p in n.parameters()])

        # (1) の勾配と、(2) の勾配の平均が一致すれば、all-reduce で平均を取る
        # データ並列 (DDP) が 1 GPU で大きなバッチを使うのと等価であることが分かる。
        for i,p in enumerate(m.parameters()):
            torch.testing.assert_close(
                p.grad,
                (partial[0][i]+partial[1][i])/2,
                rtol=1e-4,
                atol=1e-7,
            )

    def test_aws_guard(self):
        """承認用の環境変数なしでは、有料リソース作成スクリプトが必ず拒否することを確認する。"""
        # AWS_PAID_RESOURCE_APPROVAL を取り除いた環境で launch_after_approval.sh を実行する。
        # 安全装置が壊れると高額な GPU インスタンスが起動しかねないので、テストで守る。
        r=subprocess.run(
            ['bash','aws/launch_after_approval.sh'],
            cwd=ROOT,
            env={k:v for k,v in os.environ.items() if k!='AWS_PAID_RESOURCE_APPROVAL'},
            capture_output=True,
        )

        # 終了コード 2 と、標準エラーの "Refusing" メッセージを期待する。
        self.assertEqual(r.returncode,2)
        self.assertIn(b'Refusing',r.stderr)

    def test_plan_never_launches(self):
        """plan.py が生成する JSON の内容 (MaxCount=1、タイプ別の EFA 数) を確認する。"""
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'plan.json'

            # (1) デフォルトの p5.48xlarge: EFA 32 本、起動台数は 1 台。
            subprocess.run(
                [
                    sys.executable,'aws/plan.py',
                    '--ami','ami-example','--subnet','subnet-example','--sg','sg-example',
                    '--profile','example','--placement','example','--run-id','test',
                    '--out',str(dest),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            r=json.loads(dest.read_text())
            self.assertEqual(len(r['NetworkInterfaces']),32)
            self.assertEqual(r['MaxCount'],1)

            # (2) p6-b200.48xlarge を指定: EFA は 8 本になる。
            subprocess.run(
                [
                    sys.executable,'aws/plan.py',
                    '--ami','ami-example','--subnet','subnet-example','--sg','sg-example',
                    '--profile','example','--placement','example','--run-id','test',
                    '--instance-type','p6-b200.48xlarge','--out',str(dest),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            r=json.loads(dest.read_text())
            self.assertEqual(r['InstanceType'],'p6-b200.48xlarge')
            self.assertEqual(len(r['NetworkInterfaces']),8)

    def test_collate_groot_single_row(self):
        """GR00T の run は status と metrics が同じディレクトリにあっても 1 行にまとめる。

        以前は status_node0.csv 由来の行と groot_metrics.csv 由来の行の 2 行が出ていた。
        OOM などで environment が無い通常の run は、従来どおり status から 1 行作る。
        """
        import csv
        header='id,node_rank,nodes,gpus,status,returncode,wall_seconds,estimated_usd,command\n'
        with tempfile.TemporaryDirectory() as tmp:
            runs=Path(tmp)/'runs'
            groot=runs/'B04'
            oom=runs/'B01'/'zero3_1node'
            groot.mkdir(parents=True)
            oom.mkdir(parents=True)
            (groot/'status_node0.csv').write_text(header+'B04,0,1,8,ok,0,1234.5,40.1,cmd\n')
            (groot/'groot_metrics.csv').write_text('step,loss\n10,0.5\n')
            (groot/'wall_node0.json').write_text('{"gpus": 8, "status": "ok", "elapsed_s": 1200.0}')
            (oom/'status_node0.csv').write_text(header+'B01,0,1,8,oom,1,300,9.8,cmd\n')
            out=Path(tmp)/'out.csv'
            subprocess.run(
                [sys.executable,'scripts/collate.py','--runs',str(runs),'--out',str(out)],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            rows=list(csv.DictReader(out.open()))
            by_id={}
            for row in rows:
                by_id.setdefault(row['id'],[]).append(row)
            self.assertEqual(len(by_id['B04']),1)
            self.assertEqual(by_id['B04'][0]['estimated_usd'],'40.1')
            self.assertEqual(by_id['B04'][0]['nodes'],'1')
            self.assertEqual(by_id['B01'][0]['status'],'oom')


if __name__=='__main__':
    unittest.main()

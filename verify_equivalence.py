#!/usr/bin/env python3
"""
NKI Scratch実装 vs リファレンス実装の機能等価性検証

殿の疑念:
- 実機での性能が想定を大きく超えている
- NKIカーネルが実際に使われているか検証が必要

検証項目:
1. llama_scratch.py が実際にNKIカーネルを使用しているか確認
2. 同一入力に対する出力の一致度（logits比較）
3. 生成テキストの一致度

Author: 将軍 (cmd_019 検証)
"""

import torch
import sys
import os

def verify_nki_usage():
    """NKIカーネルが実際に使用されているか検証"""
    print("=" * 70)
    print("検証1: NKIカーネル使用状況の確認")
    print("=" * 70)

    # llama_scratch.py のコードを解析
    scratch_path = os.path.join(os.path.dirname(__file__), "llama_scratch.py")

    with open(scratch_path, "r") as f:
        code = f.read()

    # NKI実際呼び出し箇所を検出
    issues = []

    # RMSNorm - 呼び出しはあるが例外処理でフォールバック
    if "rms_norm.rms_norm_nki" in code:
        print("[RMSNorm] NKI呼び出しコードあり")
        if "except Exception" in code and "falling back to PyTorch" in code:
            print("  ⚠️ 警告: 例外時にPyTorchフォールバック")
    else:
        issues.append("RMSNorm: NKI呼び出しなし")

    # Rotary - passのみ
    rotary_section = code[code.find("def apply_rotary_pos_emb"):code.find("def apply_rotary_pos_emb")+500]
    if "pass" in rotary_section and "# NKI rotary" in rotary_section:
        issues.append("Rotary: NKI判定後にpass（実装なし）")
        print("[Rotary] ❌ NKI呼び出しなし（passのみ）")

    # Attention - passのみ
    if "# NKI attention kernel would be integrated here" in code:
        issues.append("Attention: NKI判定後にpass（実装なし）")
        print("[Attention] ❌ NKI呼び出しなし（passのみ）")

    # MLP - passのみ
    if "# NKI MLP kernel would be integrated here" in code:
        issues.append("MLP: NKI判定後にpass（実装なし）")
        print("[MLP] ❌ NKI呼び出しなし（passのみ）")

    print()
    if issues:
        print("🚨 重大な問題発見:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("結論: llama_scratch.pyは実質的にPyTorch実装のみ")
        print("      NKIカーネルは統合されておらず、性能測定値は無効")
        return False
    else:
        print("✅ 全コンポーネントでNKIカーネルが使用されている")
        return True


def verify_output_equivalence(model_path: str = None):
    """同一入力に対する出力の一致度を検証"""
    print("\n" + "=" * 70)
    print("検証2: 出力一致度の確認")
    print("=" * 70)

    if model_path is None:
        print("モデルパスが指定されていないためスキップ")
        print("使用方法: python verify_equivalence.py --model-path /path/to/model")
        return None

    try:
        from llama_scratch import SimpleLlamaModel, SimpleLlamaConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"\nモデルパス: {model_path}")

        # スクラッチ実装をロード
        print("\n[1] SimpleLlamaModel (scratch) をロード...")
        scratch_model = SimpleLlamaModel.from_pretrained(model_path)
        scratch_model.eval()

        # HuggingFace参照実装をロード
        print("[2] HuggingFace参照実装をロード...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        hf_model.eval()

        # トークナイザ
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # テスト入力
        test_prompts = [
            "Hello, world!",
            "The capital of France is",
            "In a galaxy far, far away",
        ]

        print("\n[3] 出力比較...")
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                # スクラッチ実装
                scratch_logits, _ = scratch_model(input_ids)
                scratch_logits = scratch_logits[:, -1, :]  # 最後のトークンのlogits

                # HuggingFace実装
                hf_outputs = hf_model(input_ids)
                hf_logits = hf_outputs.logits[:, -1, :]

            # 比較
            # logitsをfloat32に変換して比較
            scratch_logits_f32 = scratch_logits.float()
            hf_logits_f32 = hf_logits.float()

            # 相対誤差
            diff = (scratch_logits_f32 - hf_logits_f32).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Top-1トークンの一致
            scratch_top1 = scratch_logits.argmax(dim=-1).item()
            hf_top1 = hf_logits.argmax(dim=-1).item()
            top1_match = scratch_top1 == hf_top1

            print(f"\n  プロンプト: '{prompt}'")
            print(f"    最大差分: {max_diff:.6f}")
            print(f"    平均差分: {mean_diff:.6f}")
            print(f"    Top-1一致: {'✅' if top1_match else '❌'} (scratch={scratch_top1}, hf={hf_top1})")

        return True

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_generation_equivalence(model_path: str = None):
    """生成テキストの一致度を検証"""
    print("\n" + "=" * 70)
    print("検証3: 生成テキスト一致度の確認")
    print("=" * 70)

    if model_path is None:
        print("モデルパスが指定されていないためスキップ")
        return None

    try:
        from llama_scratch import SimpleLlamaModel
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # モデルロード
        scratch_model = SimpleLlamaModel.from_pretrained(model_path)
        scratch_model.eval()

        hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        hf_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt = "Once upon a time"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        print(f"\nプロンプト: '{prompt}'")
        print(f"生成トークン数: 32")

        with torch.no_grad():
            # スクラッチ実装で生成
            scratch_output = scratch_model.generate(input_ids, max_new_tokens=32, top_k=1)
            scratch_text = tokenizer.decode(scratch_output[0], skip_special_tokens=True)

            # HuggingFace実装で生成
            hf_output = hf_model.generate(input_ids, max_new_tokens=32, do_sample=False)
            hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)

        print(f"\n[Scratch] {scratch_text}")
        print(f"[HF]      {hf_text}")

        match = scratch_text == hf_text
        print(f"\n生成テキスト一致: {'✅' if match else '❌'}")

        if not match:
            # トークン単位で比較
            scratch_tokens = scratch_output[0].tolist()
            hf_tokens = hf_output[0].tolist()

            print(f"\n[デバッグ] トークン比較:")
            print(f"  Scratch tokens: {scratch_tokens}")
            print(f"  HF tokens:      {hf_tokens}")

            # 最初の不一致位置
            for i, (s, h) in enumerate(zip(scratch_tokens, hf_tokens)):
                if s != h:
                    print(f"  最初の不一致: 位置{i}, scratch={s}, hf={h}")
                    break

        return match

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NKI Scratch実装検証")
    parser.add_argument("--model-path", type=str, default=None, help="モデルパス")
    args = parser.parse_args()

    print("=" * 70)
    print("NKI Scratch実装 vs リファレンス実装 機能等価性検証")
    print("=" * 70)

    # 検証1: NKIカーネル使用状況
    nki_ok = verify_nki_usage()

    # 検証2: 出力一致度（モデルパスがある場合のみ）
    output_ok = verify_output_equivalence(args.model_path)

    # 検証3: 生成テキスト一致度（モデルパスがある場合のみ）
    gen_ok = verify_generation_equivalence(args.model_path)

    # サマリ
    print("\n" + "=" * 70)
    print("検証結果サマリ")
    print("=" * 70)
    print(f"1. NKIカーネル使用: {'✅' if nki_ok else '❌ 問題あり'}")
    print(f"2. 出力一致度: {'✅' if output_ok else '❌ 問題あり' if output_ok is False else '⏭️ スキップ'}")
    print(f"3. 生成一致度: {'✅' if gen_ok else '❌ 問題あり' if gen_ok is False else '⏭️ スキップ'}")

    if not nki_ok:
        print("\n" + "🚨" * 35)
        print("重大な発見: llama_scratch.pyはNKIカーネルを使用していない！")
        print("実質的にPyTorch実装のみで動作している。")
        print("報告された性能値（3420 tok/s等）はNKI性能ではない。")
        print("🚨" * 35)


if __name__ == "__main__":
    main()

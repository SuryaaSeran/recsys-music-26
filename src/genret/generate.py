"""Stage A inference: trie-constrained beam decode -> candidate pool with logprobs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.genret.model import attach_lean_vocab
from src.genret.tokens import SemTokenizer
from src.genret.trie import CfTrie


@dataclass
class Candidate:
    track_id: str
    cf_tuple: tuple
    logprob: float       # summed transition log-prob over the 4 cf tokens
    lognorm: float       # length-normalized (for Stage B fusion)


class GenRetriever:
    def __init__(self, model, tok, sem, trie, device):
        self.model = model
        self.tok = tok
        self.sem = sem
        self.trie = trie
        self.device = device
        self.fn = trie.prefix_allowed_tokens_fn()

    @classmethod
    def load(cls, ckpt_dir: str, base: str, device: str = "mps", dtype: str = "bfloat16"):
        tok = AutoTokenizer.from_pretrained(ckpt_dir)
        sem = SemTokenizer(tok)                       # tokens already present -> grid resolves
        raw = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype))
        raw.resize_token_embeddings(len(tok), mean_resizing=False)
        model = PeftModel.from_pretrained(raw, ckpt_dir)   # loads LoRA (+ may restore a full lm_head)
        # Re-attach the lean split head AFTER peft, overriding any restored full head, so
        # generation never builds [beams, prompt_len, vocab] logits.
        inner = model.base_model.model
        lv = attach_lean_vocab(inner, sem.new_lo)
        ck = torch.load(Path(ckpt_dir) / "new_token_embeddings.pt", map_location="cpu")
        lv.new_emb.data.copy_(ck["new_rows"])
        model.eval().to(device)
        trie = CfTrie.build(sem)
        return cls(model, tok, sem, trie, device)

    @torch.no_grad()
    def generate_pool(self, context: str, pool_size: int = 200, num_beams: int | None = None,
                      diverse: bool = False) -> list[Candidate]:
        nb = num_beams or max(pool_size, 256)         # >=233 so step-1 is exhaustive
        enc = self.tok(context, return_tensors="pt").to(self.device)
        plen = enc.input_ids.shape[1]
        # Use ONLY max_length (= prompt + 5 cf/eos tokens). If max_new_tokens is also set
        # it takes precedence and the KV cache pre-allocates to generation_config.max_length
        # (131072) x beams -> 134GB on MPS.
        kw = dict(num_beams=nb, num_return_sequences=pool_size, max_length=plen + 5,
                  prefix_allowed_tokens_fn=self.fn, do_sample=False,
                  return_dict_in_generate=True, output_scores=True, length_penalty=1.0,
                  pad_token_id=self.tok.eos_token_id)
        if diverse:
            kw.update(num_beam_groups=min(20, nb), diversity_penalty=0.5)
        out = self.model.generate(**enc, **kw)
        gen = out.sequences[:, enc.input_ids.shape[1]:]
        trans = self.model.compute_transition_scores(
            out.sequences, out.scores, out.beam_indices, normalize_logits=True)

        cands, seen = [], {}
        for row, scores in zip(gen.tolist(), trans):
            cf = [t for t in row if t in self.sem.tok2levelcode]
            if len(cf) != 4:
                continue
            quad = self.sem.tokens_to_cf_codes(cf)
            lp = float(scores[:4].sum())              # 4 cf-token transition logprobs
            for tid in self.trie.decode_tuple_to_tracks(quad):
                if tid not in seen or lp > seen[tid]:
                    seen[tid] = lp
                    cands.append(Candidate(tid, quad, lp, lp / 4.0))
        # one entry per track, best score, ranked
        best = {}
        for c in cands:
            if c.track_id not in best or c.logprob > best[c.track_id].logprob:
                best[c.track_id] = c
        return sorted(best.values(), key=lambda c: c.logprob, reverse=True)

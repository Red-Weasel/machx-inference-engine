# Vision goldens from the publisher's code on CPU (bf16, transformers in the reference venv), for the engine's port of
# Inkling 2 Mini's vision tower.
#
#   <ref venv>/bin/python -I oracle_vision.py OUT_DIR [STEPS]
#
# Run inside a capped scope (the full model on the CPU):
#   systemd-run --user --scope -p MemoryMax=90G -p MemorySwapMax=0 -- <python> -I oracle_vision.py ...
#
# Draws a fixed test image (240 x 160: a red disc and a blue square on white = 6 x 4 image tokens), renders one user turn
# [image, question] with the model's processor / chat template, and writes to OUT_DIR:
#   image.png, meta.json ({"prompt_ids", "image_grid_thw", "generated", "generated_text", "steps"}),
#   pixel_values.npy [tokens, 40, 40, 3] (the processor's output), tower_embed.npy (after pre_layernorm, [1 + patches, 1024]),
#   tower_layer0.npy (after layer 0), tower_last.npy (last_hidden_state), image_features.npy (pooler_output = the merger's
#   output, [tokens, 2304]: the rows that replace the placeholders), logits_stepK.npy, routes.json ({"stepK": [48][T][8]}).
import sys, os, json, re, time, numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoTokenizer, AutoModelForImageTextToText

P = os.path.expanduser("~/models/Inkling-2-Mini")
OUT = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

img = Image.new("RGB", (240, 160), (255, 255, 255))
d = ImageDraw.Draw(img)
d.ellipse((20, 30, 120, 130), fill=(220, 20, 20))
d.rectangle((150, 50, 210, 110), fill=(20, 40, 220))
img.save(os.path.join(OUT, "image.png"))

# The image processor needs torchvision (not in the venv, no installs): its _preprocess re-done in numpy, exactly --
# pad to multiples of 40 with -1 on the 0-255 scale, x / 255, (x - 0.5) / 0.5, one [40, 40, 3] patch per image token.
tok = AutoTokenizer.from_pretrained(P)
a = np.asarray(img.convert("RGB"), dtype=np.float32)                     # [H, W, 3]
Hh, Ww = a.shape[:2]
rows, cols = -(-Hh // 40), -(-Ww // 40)
pad = np.full((rows * 40, cols * 40, 3), -1.0, np.float32)
pad[:Hh, :Ww] = a
pad = (pad * np.float32(0.00392156862745098) - np.float32(0.5)) / np.float32(0.5)
pv = torch.from_numpy(pad.reshape(rows, 40, cols, 40, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, 40, 40, 3).copy())
grid = torch.tensor([[1, rows, cols]])
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the circle? Answer in one word."}]}]
text = tok.apply_chat_template(messages, tokenize=False)
assert text.count("<|unused_200054|>") == 1, text
text = text.replace("<|unused_200054|>", "<|unused_200054|>" * (rows * cols))    # Inkling2Processor.replace_image_token
ids = tok(text, return_tensors="pt")["input_ids"]
np.save(os.path.join(OUT, "pixel_values.npy"), pv.float().numpy())
print("prompt tokens", ids.shape[1], "pixel_values", tuple(pv.shape), "grid", grid.tolist(), flush=True)

model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
print(f"loaded in {time.time() - t0:.0f} s; threads {torch.get_num_threads()}", flush=True)
vt = model.model.vision_tower
lm = model.model.language_model

saved = {}
def save_hook(name):
    def f(mod, inp, out):
        if name in saved: return
        h = out[0] if isinstance(out, (tuple, list)) else out
        saved[name] = h.detach().float().cpu().numpy().copy()
    return f
vt.pre_layernorm.register_forward_hook(save_hook("tower_embed"))
vt.layers[0].register_forward_hook(save_hook("tower_layer0"))
with torch.no_grad():
    vo = vt(pixel_values=pv, image_grid_thw=grid)
np.save(os.path.join(OUT, "tower_embed.npy"), saved["tower_embed"])
np.save(os.path.join(OUT, "tower_layer0.npy"), saved["tower_layer0"])
np.save(os.path.join(OUT, "tower_last.npy"), vo.last_hidden_state.float().numpy())
np.save(os.path.join(OUT, "image_features.npy"), vo.pooler_output.float().numpy())
print("image features", tuple(vo.pooler_output.shape), f"{time.time() - t0:.0f} s", flush=True)

routes = []
def rhook(mod, inp, out):
    routes.append(out[2].detach().cpu().numpy().astype(int).tolist())
for n, m in lm.named_modules():
    if type(m).__name__ == "Inkling2LoopTopKRouter": m.register_forward_hook(rhook)

meta = {"prompt_ids": ids[0].tolist(), "image_grid_thw": grid.tolist(), "prompt_text": text, "steps": []}
allroutes = {}
past = None
cur = ids
with torch.no_grad():
    for k in range(steps + 1):
        routes.clear()
        t1 = time.time()
        kw = dict(input_ids=cur, past_key_values=past, use_cache=True)
        if k == 0: kw.update(pixel_values=pv, image_grid_thw=grid)
        out = model(**kw)
        past = out.past_key_values
        lg = out.logits[0, -1].float().cpu().numpy()
        np.save(os.path.join(OUT, f"logits_step{k}.npy"), lg)
        assert len(routes) == 48, len(routes)
        allroutes[f"step{k}"] = routes.copy()
        top = np.argsort(-lg, kind="stable")[:10]
        nxt = int(top[0])
        meta["steps"].append({"step": k, "seconds": round(time.time() - t1, 2), "top10": [[int(t), float(lg[t])] for t in top], "next": nxt})
        print(f"step {k}: {cur.shape[1]} tokens -> {nxt} {tok.decode([nxt])!r}", flush=True)
        cur = torch.tensor([[nxt]])
meta["generated"] = [s["next"] for s in meta["steps"]]
meta["generated_text"] = tok.decode(meta["generated"])
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
json.dump(allroutes, open(os.path.join(OUT, "routes.json"), "w"))
print("generated", meta["generated"], repr(meta["generated_text"]), f"total {time.time() - t0:.0f} s")

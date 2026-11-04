"""Localize z=6101 departure to masks, snapping or independently measured CT motion."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from src.trace.thin import mask_to_curve
from src.trace.sam3_tracker import ridge_snap
from src.trace.guards import _project, _polyline, check_winding


def main():
    cache = ROOT / "reports/curve_fix/curves_corrected"
    out = ROOT / "reports/departure_fix"
    out.mkdir(parents=True, exist_ok=True)
    previous, stages, masks, images = None, [], [], []
    for frame in range(2):
        with np.load(cache / f"mask_{frame:05d}.npz") as d:
            mask, transform = d["mask"], d["transform"]
        xs, ys, ox, oy = transform
        gray = np.asarray(Image.open(cache / "frames3" / f"{frame:05d}.jpg"))[:, :, 0]
        seed = ((4820-ox)/xs, (3403-oy)/ys) if previous is None else previous.mean(axis=0)
        raw = mask_to_curve(mask, seed=seed)
        if previous is not None:
            f = np.linalg.norm(raw[0]-previous[0]) + np.linalg.norm(raw[-1]-previous[-1])
            r = np.linalg.norm(raw[-1]-previous[0]) + np.linalg.norm(raw[0]-previous[-1])
            if r < f:
                raw = raw[::-1].copy()
        previous = raw.copy()
        small = ridge_snap(raw, gray)
        final = np.array(json.loads((ROOT / "reports/ridge_fix/curves_corrected" / f"{frame:05d}.jsonl").read_text())["points"])
        stages.append({"raw": raw * [xs, ys] + [ox, oy],
                       "small_snap": small * [xs, ys] + [ox, oy], "final": final})
        masks.append(mask)
        images.append(tifffile.imread(ROOT / f"data/cache/scroll1_6100_6200/{6100+frame:05d}.tif"))
    both = np.concatenate([p for s in stages for p in s.values()])
    x0, y0 = np.floor(both.min(0)-35).astype(int)
    x1, y1 = np.ceil(both.max(0)+35).astype(int)
    width, height = x1-x0, y1-y0
    canvas = Image.new("RGB", (width*3, (height+30)*2), "#141414")
    result = {"crop": [int(x0),int(y0),int(x1),int(y1)], "stages": {}}
    for col, key in enumerate(stages[0]):
        a, b = stages[0][key], stages[1][key]
        distance, _ = _project(b, _polyline(a)[0])
        result["stages"][key] = {"distance_percentiles": np.percentile(distance,[50,90,95,100]).tolist(),
                                 "guard": check_winding(b,a),
                                 "bbox0": [a.min(0).tolist(),a.max(0).tolist()],
                                 "bbox1": [b.min(0).tolist(),b.max(0).tolist()]}
        for row in range(2):
            crop = images[row][y0:y1,x0:x1].astype(float)
            lo,hi = np.percentile(crop,[1,99])
            image = Image.fromarray(np.uint8(np.clip((crop-lo)/max(hi-lo,1)*255,0,255))).convert("RGB")
            draw=ImageDraw.Draw(image)
            for curve,color in ((a,"cyan"),(b,"orange")):
                draw.line([tuple(p-[x0,y0]) for p in curve],fill=color,width=1)
            canvas.paste(image,(col*width,row*(height+30)+30))
            ImageDraw.Draw(canvas).text((col*width+5,row*(height+30)+6), f"CT {6100+row}: {key}; cyan=6100 orange=6101",fill="white")
    canvas.save(out / "stage_comparison.png")
    # Independent image evidence, without aligning the traced curves. A single
    # translation is diagnostic here, not an automatic authorization to reanchor.
    a = images[0][y0:y1,x0:x1].astype(float)
    b = images[1][y0:y1,x0:x1].astype(float)
    motion=[]
    for dy in range(-6,7):
        for dx in range(-6,7):
            aa=a[8:-8,8:-8]; bb=b[8+dy:height-8+dy,8+dx:width-8+dx]
            av=aa-aa.mean(); bv=bb-bb.mean()
            score=float((av*bv).sum()/max(np.linalg.norm(av)*np.linalg.norm(bv),1e-9))
            motion.append((score,dx,dy))
    motion.sort(reverse=True)
    result["ct_motion_ncc"] = [{"score":s,"dx":x,"dy":y} for s,x,y in motion[:5]]
    result["mask_iou"] = float((masks[0]&masks[1]).sum()/(masks[0]|masks[1]).sum())
    # Check the competing explanation explicitly: lateral motion of a whole
    # layer spacing, outside the small registration search used by the tracker.
    template = images[0][3240:3520,4750:4860].astype(float)
    template -= template.mean()
    energy = (template*template).sum()
    wide = []
    for dy in range(-5,6):
        for dx in range(-30,31):
            candidate=images[1][3240+dy:3520+dy,4750+dx:4860+dx].astype(float)
            candidate -= candidate.mean()
            score=float((template*candidate).sum()/max(np.sqrt(energy*(candidate*candidate).sum()),1e-9))
            wide.append({"score":score,"dx":dx,"dy":dy})
    result["wide_ct_motion_ncc"] = {
        "template_window_xyxy":[4750,3240,4860,3520],
        "search_dx":[-30,30],"search_dy":[-5,5],
        "best":max(wide,key=lambda r:r["score"]),
        "best_large_lateral_shift":max((r for r in wide if abs(r["dx"])>=10),key=lambda r:r["score"])}
    (out/"diagnosis.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"stages":{k:{a:b for a,b in v.items() if a != "guard"} for k,v in result["stages"].items()},"motion": result["ct_motion_ncc"],"mask_iou": result["mask_iou"]},indent=2))


if __name__ == "__main__":
    main()

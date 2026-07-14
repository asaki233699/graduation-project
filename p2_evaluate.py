# p2_evaluate.py
import os, sys, time, argparse, json, re
import numpy as np
from difflib import SequenceMatcher
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapely.geometry import Polygon
from mmocr.apis import MMOCRInferencer
from mmocr.utils import poly_iou

def compute_det_metrics(gt_polygons, pred_polygons, match_iou_thr=0.5):
    gt_num, pred_num = len(gt_polygons), len(pred_polygons)
    if gt_num == 0 and pred_num == 0:
        return dict(precision=1.0, recall=1.0, hmean=1.0, gt_num=0, pred_num=0, match_num=0)
    if gt_num == 0:
        return dict(precision=0.0, recall=1.0, hmean=0.0, gt_num=0, pred_num=pred_num, match_num=0)
    if pred_num == 0:
        return dict(precision=0.0, recall=0.0, hmean=0.0, gt_num=gt_num, pred_num=0, match_num=0)
    def to_shapely(lst):
        out = []
        for p in lst:
            if len(p) < 6: continue
            coords = [(p[i], p[i+1]) for i in range(0, len(p), 2)]
            try:
                poly = Polygon(coords)
                if poly.is_valid and not poly.is_empty: out.append(poly)
            except: pass
        return out
    gt = to_shapely(gt_polygons)
    pd = to_shapely(pred_polygons)
    if len(gt) == 0 or len(pd) == 0:
        return dict(precision=float(len(pd)>0), recall=float(len(gt)>0),
                    hmean=0.0, gt_num=gt_num, pred_num=pred_num, match_num=0)
    iou = np.zeros((len(gt), len(pd)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pd):
            try: iou[i,j] = poly_iou(g, p)
            except: iou[i,j] = 0.0
    mg, mp = set(), set()
    for gi, pi in zip(*np.where(iou >= match_iou_thr)):
        if gi in mg or pi in mp: continue
        mg.add(gi); mp.add(pi)
    m = len(mg)
    prec = m / pred_num
    rec = m / gt_num
    d = prec + rec
    h = 0.0 if d == 0 else 2.0 * prec * rec / d
    return dict(precision=round(prec,4), recall=round(rec,4), hmean=round(h,4),
                gt_num=gt_num, pred_num=pred_num, match_num=m)

def _lev_norm(a, b):
    mx = max(len(a), len(b))
    if mx == 0: return 0.0
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a):
        cur = [i+1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j+1]+1, cur[j]+1, prev[j]+(0 if ca==cb else 1)))
        prev = cur
    return prev[-1] / mx

def compute_rec_metrics(gt_texts, pred_texts):
    total = len(gt_texts)
    if total == 0: return {}
    vs = re.compile(r"[^A-Za-z0-9\u4e00-\u9fa5]")
    em, icm, icsm = 0, 0, 0
    tgc, tpc, ttc, tned = 0, 0, 0, 0.0
    for gt, pr in zip(gt_texts, pred_texts):
        if gt == pr: em += 1
        if gt.lower() == pr.lower(): icm += 1
        gc = vs.sub("", gt.lower())
        pc = vs.sub("", pr.lower())
        if gc == pc: icsm += 1
        tgc += len(gc); tpc += len(pc)
        sm = SequenceMatcher(None, pc, gc)
        ttc += sum(e2-s2 for op,_,_,s2,e2 in sm.get_opcodes() if op=="equal")
        tned += _lev_norm(pc, gc)
    e = 1e-8
    return dict(
        word_acc=round(em/(total+e),4),
        word_acc_ignore_case=round(icm/(total+e),4),
        word_acc_ignore_case_symbol=round(icsm/(total+e),4),
        char_recall=round(ttc/(tgc+e),4),
        char_precision=round(ttc/(tpc+e),4),
        one_minus_ned=round(1.0-tned/(total+e),4))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--normal_dir", default="demo")
    p.add_argument("--lowlight_dir", default="lowlight_dataset")
    p.add_argument("--output_dir", default="outputs/phase2_eval")
    p.add_argument("--no_save_vis", action="store_true")
    a = p.parse_args()
    sd = os.path.dirname(os.path.abspath(__file__))
    nd = a.normal_dir if os.path.isabs(a.normal_dir) else os.path.join(sd, a.normal_dir)
    ld = a.lowlight_dir if os.path.isabs(a.lowlight_dir) else os.path.join(sd, a.lowlight_dir)
    od = a.output_dir if os.path.isabs(a.output_dir) else os.path.join(sd, a.output_dir)
    on = os.path.join(od, "normal"); ol = os.path.join(od, "lowlight")
    os.makedirs(on, exist_ok=True); os.makedirs(ol, exist_ok=True)
    sv = not a.no_save_vis
    print("="*60)
    print("Phase 1  DBNet + CRNN  Standard Metrics")
    print("="*60)
    print(f"Normal:  {nd}")
    print(f"LowLight: {ld}")
    print(f"Output:  {od}")
    print()
    print("NOTE: normal-light results used as pseudo-GT.")
    print()
    print("[1/3] Loading DBNet+CRNN...")
    ocr = MMOCRInferencer(det="DBNet", rec="CRNN")
    print("      Done.")
    ve = (".jpg",".png",".jpeg",".bmp",".tiff")
    nf = {f for f in os.listdir(nd) if f.lower().endswith(ve)}
    lf = {f for f in os.listdir(ld) if f.lower().endswith(ve)}
    com = sorted(nf & lf)
    if not com:
        print(f"ERROR: no common images!")
        sys.exit(1)
    print(f"\n[2/3] {len(com)} images. Running...")
    ds, rs, pi = [], [], []
    for idx, nm in enumerate(com, 1):
        print(f"\n  [{idx}/{len(com)}] {nm}")
        npth = os.path.join(nd, nm); lpth = os.path.join(ld, nm)
        t0 = time.time()
        rn = ocr(npth, out_dir=on, return_vis=True) if sv else ocr(npth, return_vis=False)
        tn = time.time()-t0
        pn = rn["predictions"][0]
        gp = pn.get("det_polygons", []); gt = pn.get("rec_texts", [])
        t0 = time.time()
        rl = ocr(lpth, out_dir=ol, return_vis=True) if sv else ocr(lpth, return_vis=False)
        tl = time.time()-t0
        pl = rl["predictions"][0]
        pp = pl.get("det_polygons", []); pt = pl.get("rec_texts", [])
        dr = compute_det_metrics(gp, pp); ds.append(dr)
        nt = min(len(gt), len(pt))
        rr = compute_rec_metrics(gt[:nt], pt[:nt]); rs.append(rr)
        print(f"      normal: {len(gp)}det/{len(gt)}rec | lowlight: {len(pp)}det/{len(pt)}rec")
        print(f"      H={dr['hmean']:.4f} P={dr['precision']:.4f} R={dr['recall']:.4f} | WA={rr.get('word_acc',0):.4f} 1NED={rr.get('one_minus_ned',0):.4f}")
        print(f"      t: {tn:.2f}s | {tl:.2f}s")
        pi.append(dict(name=nm, gt_det=len(gp), pred_det=len(pp), gt_rec=len(gt), pred_rec=len(pt), det=dr, rec=rr))
    print("\n"+"="*60)
    print("[3/3] Summary")
    print("="*60)
    ad = {k:round(np.mean([d[k] for d in ds]),4) for k in ["precision","recall","hmean"]}
    tg = sum(d["gt_num"] for d in ds)
    tp_ = sum(d["pred_num"] for d in ds)
    tm = sum(d["match_num"] for d in ds)
    mp = tm/tp_ if tp_ else 0
    mr = tm/tg if tg else 0
    mh = (2*mp*mr/(mp+mr)) if (mp+mr) else 0
    print(f"\n  --- Detection ---")
    print(f"  Per-img avg  P={ad['precision']:.4f}  R={ad['recall']:.4f}  H={ad['hmean']:.4f}")
    print(f"  Global        P={mp:.4f} ({tm}/{tp_})  R={mr:.4f} ({tm}/{tg})  H={mh:.4f}")
    print(f"\n  --- Recognition ---")
    rlbl = [("word_acc","Word Acc exact"),("word_acc_ignore_case","Word Acc ig.case"),
            ("word_acc_ignore_case_symbol","Word Acc ig.csym"),("char_recall","Char Recall"),
            ("char_precision","Char Precision"),("one_minus_ned","1-NED")]
    for k,lb in rlbl:
        vs = [r.get(k,0) for r in rs]
        print(f"  {lb:<22} {np.mean(vs) if vs else 0:.4f}")
    print(f"\n  {'Image':<25} {'Hmean':<8} {'Prec':<8} {'Rec':<8} {'WA':<8} {'1NED':<8}")
    print(f"  {'-'*65}")
    for it in pi:
        print(f"  {it['name']:<25} {it['det']['hmean']:<8.4f} {it['det']['precision']:<8.4f} {it['det']['recall']:<8.4f} {it['rec'].get('word_acc',0):<8.4f} {it['rec'].get('one_minus_ned',0):<8.4f}")
    print(f"  {'-'*65}")
    print(f"  {'Average':<25} {ad['hmean']:<8.4f} {ad['precision']:<8.4f} {ad['recall']:<8.4f} {np.mean([r.get('word_acc',0) for r in rs]):<8.4f} {np.mean([r.get('one_minus_ned',0) for r in rs]):<8.4f}")
    rf = os.path.join(od, "metrics.json")
    with open(rf,"w",encoding="utf-8") as f:
        json.dump(dict(per_image=pi, avg_det=ad,
            macro_det=dict(precision=round(mp,4),recall=round(mr,4),hmean=round(mh,4)),
            avg_rec={k:round(np.mean([r.get(k,0) for r in rs]),4) for k,_ in rlbl}),
            f, ensure_ascii=False, indent=2)
    print(f"\nDone! -> {rf}")

if __name__ == "__main__":
    main()

def merge_intervals(intervals):
    iv = []
    for s, e in intervals:
        if s > e:
            raise ValueError(f"start > end in ({s}, {e})")
        if s < e:
            iv.append((s, e))
    iv.sort()
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out

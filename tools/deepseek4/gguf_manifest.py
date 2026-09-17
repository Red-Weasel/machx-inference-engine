import struct, sys, os, glob, re, collections

# GGML type -> (block elems, block bytes)
TRAITS = {
 0:('F32',1,4), 1:('F16',1,2), 2:('Q4_0',32,18), 3:('Q4_1',32,20), 6:('Q5_0',32,22), 7:('Q5_1',32,24),
 8:('Q8_0',32,34), 9:('Q8_1',32,36), 10:('Q2_K',256,84), 11:('Q3_K',256,110), 12:('Q4_K',256,144),
 13:('Q5_K',256,176), 14:('Q6_K',256,210), 15:('Q8_K',256,292), 16:('IQ2_XXS',256,66), 17:('IQ2_XS',256,74),
 18:('IQ3_XXS',256,98), 19:('IQ1_S',256,50), 20:('IQ4_NL',32,18), 21:('IQ3_S',256,110), 22:('IQ2_S',256,82),
 23:('IQ4_XS',256,136), 24:('I8',1,1), 25:('I16',1,2), 26:('I32',1,4), 27:('I64',1,8), 28:('F64',1,8),
 29:('IQ1_M',256,56), 30:('BF16',1,2), 39:('MXFP4',32,17),
}

def parse(path):
    f = open(path,'rb')
    def rd(n):
        b=f.read(n)
        if len(b)<n: raise EOFError
        return b
    def u32(): return struct.unpack('<I',rd(4))[0]
    def u64(): return struct.unpack('<Q',rd(8))[0]
    assert rd(4)==b'GGUF'
    u32(); nt=u64(); nkv=u64()
    def rstr(): return rd(u64()).decode('utf-8','replace')
    def rval(t):
        if t==0: return rd(1)[0]
        if t==1: return struct.unpack('<b',rd(1))[0]
        if t==2: return struct.unpack('<H',rd(2))[0]
        if t==3: return struct.unpack('<h',rd(2))[0]
        if t==4: return u32()
        if t==5: return struct.unpack('<i',rd(4))[0]
        if t==6: return struct.unpack('<f',rd(4))[0]
        if t==7: return rd(1)[0]!=0
        if t==8: return rstr()
        if t==9:
            et=u32(); n=u64()
            return [rval(et) for _ in range(n)]
        if t==10: return u64()
        if t==11: return struct.unpack('<q',rd(8))[0]
        if t==12: return struct.unpack('<d',rd(8))[0]
        raise ValueError(t)
    kv={}
    for _ in range(nkv):
        k=rstr(); t=u32(); kv[k]=rval(t)
    out=[]
    for _ in range(nt):
        nm=rstr(); nd=u32(); shape=[u64() for _ in range(nd)]; tt=u32(); off=u64()
        out.append((nm,shape,tt))
    return kv,out

allt=[]
for p in sorted(glob.glob(sys.argv[1]+'/*.gguf')):
    try:
        kv,ts = parse(p)
        allt += ts
        print(f"{os.path.basename(p)[-22:]}: {len(ts)} tensors")
    except Exception as e:
        print(f"{os.path.basename(p)[-22:]}: parse stopped ({type(e).__name__})")

print(f"\nTOTAL TENSORS PARSED: {len(allt)}")

def nbytes(shape, tt):
    n=1
    for s in shape: n*=s
    name,be,bb = TRAITS.get(tt,(f'T{tt}',1,4))
    return n*bb//be, name

# dtype histogram + bytes
hist=collections.Counter(); bytesby=collections.Counter()
for nm,sh,tt in allt:
    b,name = nbytes(sh,tt); hist[name]+=1; bytesby[name]+=b
print("\n=== dtype histogram ===")
for k in sorted(hist, key=lambda x:-bytesby[x]):
    print(f"  {k:9s} {hist[k]:5d} tensors  {bytesby[k]/1e9:9.2f} GB")
print(f"  {'TOTAL':9s} {len(allt):5d} tensors  {sum(bytesby.values())/1e9:9.2f} GB")

# routed experts vs everything else
exp_b=0; other_b=0; exp_n=0
for nm,sh,tt in allt:
    b,_=nbytes(sh,tt)
    if '_exps.' in nm: exp_b+=b; exp_n+=1
    else: other_b+=b
print(f"\n=== placement split ===")
print(f"  routed experts (*_exps.*): {exp_n} tensors, {exp_b/1e9:.2f} GB")
print(f"  everything else:           {len(allt)-exp_n} tensors, {other_b/1e9:.2f} GB")

# per-layer breakdown
lay=collections.defaultdict(int); layn=collections.defaultdict(list)
nonlayer=[]
for nm,sh,tt in allt:
    m=re.match(r'blk\.(\d+)\.(.+)', nm)
    b,name=nbytes(sh,tt)
    if m:
        lay[int(m.group(1))]+=b
        layn[int(m.group(1))].append((m.group(2),tuple(sh),name))
    else:
        nonlayer.append((nm,tuple(sh),name,b))
print(f"\n=== layers found: {sorted(lay)} ===")
if lay:
    import statistics
    print(f"  per-layer bytes: min {min(lay.values())/1e6:.1f} MB  max {max(lay.values())/1e6:.1f} MB")

print("\n=== non-layer tensors ===")
for nm,sh,name,b in sorted(nonlayer):
    print(f"  {nm:34s} {str(sh):22s} {name:8s} {b/1e6:9.1f} MB")

# per-layer tensor-set diff: find layers whose tensor NAME SET differs from the modal set
sets={i:frozenset(x[0] for x in v) for i,v in layn.items()}
modal=collections.Counter(sets.values()).most_common(1)
if modal:
    base=modal[0][0]
    print(f"\n=== modal per-layer tensor set ({modal[0][1]} layers, {len(base)} tensors) ===")
    for t in sorted(base):
        ex=[x for x in layn[[i for i in sets if sets[i]==base][0]] if x[0]==t][0]
        print(f"  {t:32s} {str(ex[1]):22s} {ex[2]}")
    print("\n=== layers DIFFERING from modal ===")
    for i in sorted(sets):
        if sets[i]!=base:
            print(f"  blk.{i}: +{sorted(sets[i]-base)}  -{sorted(base-sets[i])}")
    if all(s==base for s in sets.values()):
        print("  (none — all parsed layers identical)")

from itertools import permutations

# Exact arithmetic: one ready selected Q8 value (score 1, weight 1),
# derived_gain=1, null_weight=1; pre-derived source/relay input b.
# Use doubled coordinates: x=2*b+v, so no rounding is needed.
def e(i, n=4):
    return tuple(int(j == i) for j in range(n))

def input2(a, v, q):
    b = e(q[a]) + (0,)*4
    return tuple(2*t+u for t,u in zip(b, (0,)*4+v))

def gate_reader(x):
    return max(range(4), key=lambda j:x[j]) if sum(t*t for t in x[4:]) >= 1 else None

def content_reader(x, q):
    return q[max(range(4), key=lambda j:x[4+j])] if sum(t*t for t in x[4:]) >= 1 else None

checks = dict(pins=0, withdrawal=0, restoration=0, no_memory=0,
              wrong_value_gate=0, wrong_value_content=0, norm_match=0)
for q1,q2 in permutations(list(permutations(range(4))),2):
    for a0,a1 in permutations(range(4),2):
        for q in (q1,q2):
            for a,wrong in ((a0,a1),(a1,a0)):
                intact = input2(a,e(a),q)
                withheld = input2(a,(0,)*4,q)
                alternate = input2(a,e(wrong),q)
                assert gate_reader(intact) == content_reader(intact,q) == q[a]
                checks['pins'] += 1
                assert gate_reader(withheld) is None and content_reader(withheld,q) is None
                checks['withdrawal'] += 1
                assert gate_reader(input2(a,e(a),q)) == q[a]
                checks['restoration'] += 1
                assert gate_reader((0,)*8) is None and content_reader((0,)*8,q) is None
                checks['no_memory'] += 1
                assert intact != alternate and sum(t*t for t in intact) == sum(t*t for t in alternate)
                checks['norm_match'] += 1
                assert gate_reader(alternate) == q[a]
                checks['wrong_value_gate'] += 1
                assert content_reader(alternate,q) == q[wrong] != q[a]
                checks['wrong_value_content'] += 1
print(checks)

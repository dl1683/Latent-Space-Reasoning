# ENDURANCE-TINY-CERT-1: self-contained finite proof and positive policy

2026-10-04 REGISTERED DESIGN CANDIDATE before implementation/smoke/FULL.
The earlier SMT experiment10338110 reportsUNSAT atN4k2L2H6 but has no checked
certificate and remains TINY-VOID. This is a NEW fixed certificate-capable
unit, not a relabel/retry of that outcome. CPUonly/GPUunused. The scalable
paidcompiler remains active separately; no finite result replaces its CAN.

## Fixed scope and formula design

Complete arbitrary stationary endpoints under the admitted binary closedwear
relaxation, N4k2L2. Two frozen horizons: H6 (508records504edges1016queries)
and H5 (252records248edges504queries). Everyrawx and binaryFLIPword; all
hiddenwear histories retained separately. QUERY-only terminalH. No preferred
init/linearity/symmetry/onebittransition restriction. Exactlysame bitpositions
and logicalXOR update semantics as ENDURANCE_TINY_6_PROTOCOL.

Pure propositional CNF, no SMTtheory certificate. Policy: Boolean decoder
bits d[s][i]; onehot I[x][s] and T[i][s][t] for all states0..15. Everymapentry
selects exactlyone successor, including irrelevant states (totalextension
is allowed by admittedrelaxation). Each boundary record hasonehotstate u[s]
andfourphysical bitvariables v[j], with v[j] equivalentlyOR of u[s] whose
statebitj=1. Root u equivalentI[x]. Update for every s,t:

    u_parent[s] AND T[i][s][t] -> u_child[t].

u onehot ensures this is EXACT mappedstate. For each node,state s,logicalbit i,
u[s] implies d[s][i] equals its concrete required logicalbit.

Wear L2 represented by twoBoolean thresholds w1=(wear>=1), w2=(wear>=2).
Root w1=physicalbit, w2=false. For edge delta=oldbit XORnewbit:

    neww1 <-> oldw1 OR delta
    neww2 <-> oldw2 OR (oldw1 AND delta)
    NOT(oldw2 AND delta).

Also w2->w1. These precisely represent integerwear0/1/2, with overflow3
forbidden, never modulo wrap. Tseitin auxiliaries useFULL equivalences
for AND/OR/XOR. All onehot constraints haveatleastone andpairwiseatmostone.
DIMACS integerCNF plus complete variable/name/protocol mapping is saved.
Freshdesign must prove CNFcoverage/equivalence beforecode. CompleteUNSAT
excludes endpointrelaxation, hence actualmachines atthat finiteH; notscale.

## Solver and independent certificate checker

Installed PySAT1.9.dev7 already exists. Official primarysolverAPI confirms
Glucose3 supports `with_proof=True` and `get_proof()` DRUP:
https://pysathq.github.io/docs/html/api/solvers.html
That APIfact is not an admission that this installedbackend will work.
Frozen solver Glucose3, oneconfiguration, no assumptions/symmetrybreaking/
portfolio/adaptivehorizon. Samebranchfor bothH; H6thenH5. Reportexactversion,
backend/options/statistics/observedstatus/runtime, all hashes/attempt/git/env.

H6 UNSAT is eligible ONLY with saved full DRUPproof independently checked
against originalDIMACS. Separate checker uses DIFFERENT MiniSat22 backend
as UNITPROPAGATION ENGINE ONLY, never calls solve/search/proofgeneration.
For each addedclauseC, assume negationofEVERYliteralC and require unitconflict
against originalCNF plus previouslyVALIDATED additions, then addC. Empty
clause must itself pass. Deletion lines may be ignored: retaining already
entailed clauses preserves satisfiability and cannot make a false theorem
valid. This relaxation of syntactic DRUP checking needs fresh explicitproof
review. No RATstep accepted; non-RUP or malformedproof =>VOID. Backend/API
rootconflict handling and empty-assumption behavior require meaningfulsmokes.
Proof/CNF/readback/hashes arebound, not trust in solverUNSATstatus. Checker
is independent executable/backend, not formalverificationofC++/hardware.

H5 SAT policy recoveredfromonehotdecoder/I/T, independently concreteinteger
replayed by validate_endurance_tiny.validate, all252endpoints/248edges/504queries.
H6SAT ifunexpected receivesall508 concretechecks andsignals disagreement
withpreviousSMT, not invalidity byassumption. AnyactualmachineCANstillneeds
separate compiled-bit/control/wear/source/work gate underadmittednewkernel.

## Resources and gate chain

FULL fixed twohorizons includes solver+proofserialization+certificatecheck+
SATreplay, CPU<=4, aggregatejobRAM<=8GiB, aggregatewall<=3600s frompreparation
throughpublication. Reuse explicitWindowsjob/watchdog/snapshot/rootexit rules
of independently reviewedlauncher, adaptingduration/sources/names visibly.
One resource/configuration plan; no scaling/adaptivefallback ifcap/UNKNOWN.
Logs/proof/binarysource/CNF mapping/environment/receipts and rootobserved exit
persist. Any rootnonzeroexit VOID overrides intermediateartifactstatuses.

Register -> fresh DESIGN -> implement -> smoke -> fresh code/formula/checker/
resource review -> FULL -> freshindependentraw/certadmission -> record/commit.
Smoke fixed H0/1/2 ordinary N1k1L2 and N2k2L2, completeN1policy assignments
overH0..3 crosschecked againstoldSMTformula (256fixedpolicy controls), invalid
wear/nonpreferredinit controls, validtinyDRUP/injectedinvalidclause/empty
proof/alteredCNF tests, optimizedchecks/rootfail/resource watchdog. No full
N4solver beforecodegate; generatingCNFforcounts andstructure isallowedsmoke.

## Decision tokens

CERT-ENDPOINT-CONFIRM: checkedcompleteH6UNSAT plusconcretelyverifiedH5SAT;
finite endpoint optimum5 only, actualpaidmachineCANunadmitted.
CERT-CANNOT-ONLY: checkedH6certificate butH5positiveunclosed; no fullpair.
CERT-SAT6: verifiedH6SAT; earlierSMT/sourcebuginvestigation required, noCANyet.
CERT-VOID: cap/UNKNOWN/error/nonzeroexit/unsupportedbackend/failedcertificate/
uncertifiedUNSAT/badwitness. Individualvalidclauses/results may be reported
only atactualgate/scope; no globalmissionorprimarynovelty admission.

No requestedpapers/GPU/hypercomputation. ActualCPUcertification resolvesone
finite question; primaryjournal/strictdelta/tightloss atN2k/lowr remainOPEN.

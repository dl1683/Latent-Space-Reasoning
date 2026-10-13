# Native proof-stream flush repair: fixed Glucose3, no solver substitution

2026-10-04 EMPIRICAL SMOKE PASS after code repair, NOT FULL admission.
Root explicitWindowsCRT fflush(NULL) on ucrtbase andmsvcrt makes native
Glucose3 proof readable. Same N1k1L2H2 now returns9 DRUP lines ratherthan[];
bothflushcalls return0, solverdelete completes androotdiagnosticexit0.
This directly supports a buffering defect; no platform-widefailureclaim.
Current process only; no external files/globalenvironment/otherbackendsearch.
Currentworker flushesbeforeproofread and enablesloggingbeforeappendformula.
Allproofs independentlychecked, no trustingUNSAT/flushsuccess.

Full REGISTERED smoke subsequently exits0/CERT-SMOKE-PASS:
- six ordinary controlsN1/N2 k=N L2 H0/1/2, fourSAT/twoUNSAT checkedproofs;
-256 fixedN1policy/CNF/integerSMT equivalences overH0..3;
-24 fullAND/OR/XORtruthassignments;
-invalid/empty/truncatedproof andalteredCNFidentity rejected;
-originalemptyclause/conflictingrootunits certify0 byindependentunitprop;
-FULL H6/H5 CNF generatedonly (NOsolver):18864vars259824clauses and
9648vars130544clauses; exact508/252historycountchecks.
Artifacts results/endurance_tiny_cert_smoke/smoke.json anditsrawcontrols.

Freshcode/checker/resource gate REQUIRED beforeFULL. Newboundedlauncher
run_endurance_tiny_cert.py snapshotsallclosure, hashesnativePySATbackend,
sharespreviouslyreviewedWindowsjob/metadatautility, CPU4/RAM8GiB/3600
aggregatewatchdog/workerwait cap, H6thenH5 fixedGlucose3, unitpropMiniSat22.
Rawrootnonzero alwaysVOID; pendingoutput isnotadmission.
Bytes boundbyevidence/designtokens areprotected by.gitattributes fromGit
newlineconversion. Originalfailedproof0 persistsingit623a7ae; live smoke
nowvalid9-lineproof. PriorSMT10338110 remainsVOID, no6-updateimpossibility
or5-updateCAN established. Scalablecompilerandstrictpriority stillOPEN.

Subsequent actual ENDURANCE_TINY_CERT_FAILURE_REVIEW independently measures
0 pipe bytes before native flush/80 after and verifies imported fflush is
ucrtbase. Current implementation uses that matched CRT only; a finally block
inside Glucose context flushes before descriptor close on BOTH success and
loading/solving errors. Injected exception after tinyUNSAT exits0, cleanupPASS.
Full registered smoke rerun exits0; source-bound root exit receipt now exists
at results/endurance_tiny_cert_smoke/root_exit.json. Still no FULL newunit run.

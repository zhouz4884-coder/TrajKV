# Sources, notices and scope

| Component | Upstream | Retained scope |
|---|---|---|
| KIVI quantization | https://github.com/jy-yuan/KIVI | MIT notice in`legacy/viscontract/third_party/KIVI/LICENSE`; quantization source subset |
| CalibQuant operators | https://github.com/insuhan/calibquant | MIT notice in`legacy/viscontract/third_party/CalibQuant/LICENSE`; original file SHA retained |
| KVarN reference | https://github.com/huawei-csl/KVarN | Apache2.0 notice in`third_party/KVarN/LICENSE`; selected helpers and explicitly marked numerical ports |
| TextVQA evaluation | Facebook Research MMF / EvalAI answer processing | Original BSD3-Clause notice in`third_party/textvqa_metric_source/license.txt`; standalone adapter marked in code |
| InternVL3-8B-hf | https://huggingface.co/OpenGVLab/InternVL3-8B-hf | Model weights are obtained upstream under its model terms; not redistributed here |

First-party code and generated research outputs are published as a research preview; no new standard first-party license is assigned by this export. Upstream code/data/model rights and original notices retain their own scope. This repository does not relicense upstream datasets or weights.

Rotations, product/vector quantization, residual coding and output-distribution calibration are existing components. Their use is attributed in the literature notes; this archive records which combinations and mechanism claims were actually tested, including negative findings.

# Focused literature review: AdaptiveML / provisional SARA-IDS

**Reviewed:** 2026-11-26
**Purpose:** identify close prior work and scope a defensible research question. SARA is a provisional project name, not an established or verified novel contribution.

## Search and verification method

Searches covered energy-aware TinyML IDS, adaptive model selection under edge constraints, IDS edge/cloud offloading, cross-layer self-adaptation, resource-aware federated IDS, Non-IID/federated distillation, and the dataset cited by the repository. Publisher pages, DOI landing pages, and institutional author repositories were prioritized. The records below include direct links to the publisher or author repository. Where a publisher page exposed only an abstract or metadata, that limit is stated rather than inferred away. This is a focused search, not a systematic review: it does not establish novelty.

## Relevant studies

### 1. Edge-IIoTset dataset

- **Citation:** Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, and Helge Janicke, “Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning,” *IEEE Access*, vol. 10, pp. 40281–40306, 2022. [DOI: 10.1109/ACCESS.2022.3165809](https://doi.org/10.1109/ACCESS.2022.3165809)
- **Method/dataset:** describes an IoT/IIoT testbed dataset intended for centralized and federated IDS evaluation; the paper reports features from several sources and a substantially larger original schema than this checkout's reduced table.
- **Limitations/relevance:** this repository's 25,000-row CSV has 23 columns and the saved model schema has 18 transformed features. No source file manifest, original dataset version, license record, or checksum mapping to an official release is present. Therefore this checkout should be described as an *Edge-IIoTset-derived* artifact until provenance is verified; it must not be represented as the complete published dataset.
- **Overlap/open question:** establishes a dataset context, not an adaptive decision contribution. Reproducibility requires recording the exact official files, filtering rules, label mapping, and license.

### 2. On-device IDS energy measurement

- **Citation:** Nazli Tekin, Abbas Acar, Ahmet Aris, A. Selcuk Uluagac, and Vehbi Cagri Gungor, “Energy consumption of on-device machine learning models for IoT intrusion detection,” *Internet of Things*, vol. 21, article 100670, 2023. [DOI: 10.1016/j.iot.2022.100670](https://doi.org/10.1016/j.iot.2022.100670)
- **Method/dataset:** compares cloud-, edge-, and IoT-device training approaches, plus conventional and TinyML inference, for a smart-home IDS context. The paper reports measuring both execution time and power; it finds Decision Trees favorable in its evaluated setup.
- **Limitations/relevance:** its device/application and attack setting differ from this IIoT prototype; results cannot be transferred to this board or dataset. It demonstrates why power measurement must cover the device executing inference and why estimates cannot support physical energy claims.
- **Overlap/open question:** directly overlaps the energy-vs-detection motivation. SARA would need measured, workload-matched energy and security results on its own hardware to add evidence.

### 3. Online edge DNN configuration and model selection

- **Citation:** Kongyange Zhao, Zhi Zhou, Xu Chen, Ruiting Zhou, Xiaoxi Zhang, Shuai Yu, and Di Wu, “EdgeAdaptor: Online Configuration Adaption, Model Selection and Resource Provisioning for Edge DNN Inference Serving at Scale,” *IEEE Transactions on Mobile Computing*, vol. 22, 2023, pp. 5870–5886. [DOI: 10.1109/TMC.2022.3189186](https://doi.org/10.1109/TMC.2022.3189186)
- **Method:** online optimization jointly adjusts application configurations, model choices, and edge resource provisioning for accuracy, latency, and resource cost.
- **Limitations/relevance:** the retrieved publisher metadata describes DNN service workloads, not the repository's six-class IIoT dataset or a security-recall constraint under malicious resource exhaustion.
- **Overlap/open question:** model switching and resource-aware provisioning are established ideas. A differentiating claim would need to specify and test an IDS-specific security constraint and its consequences under controlled attacks and benign load.

### 4. Reinforcement-learned model selection for edge anomaly detection

- **Citation:** Javad Forough, Monowar Bhuyan, and Erik Elmroth, “Reinforced model selection for resource efficient anomaly detection in edge clouds,” *Future Generation Computer Systems*, vol. 176, article 108161, 2026. [DOI: 10.1016/j.future.2025.108161](https://doi.org/10.1016/j.future.2025.108161)
- **Method:** adapted DQN selects anomaly-detection models based on system state; the article reports testbed evaluation and ablations, and allows a resource/latency/accuracy trade-off.
- **Limitations/relevance:** the publisher abstract reports less than an 8% F1 decrease as an acceptable efficiency trade-off in its evaluated task. That is precisely why an IDS research question may need explicit per-class false-negative constraints, but it does not prove that such a constraint is new.
- **Overlap/open question:** directly overlaps edge model selection under resources. A fair comparison should include its resource-aware objective or a faithful baseline and quantify any security floor violations.

### 5. Self-adaptive cyber defense balancing security and energy

- **Citation:** Saeid Jamshidi, Ashkan Amirniab, Amin Nikanjam, Kawser Wazed Nafi, Foutse Khomh, and Samira Keivanpour, “Self-adaptive cyber defense for sustainable IoT: A DRL-based IDS optimizing security and energy efficiency,” *Journal of Network and Computer Applications*, vol. 239, article 104176, 2025. [DOI: 10.1016/j.jnca.2025.104176](https://doi.org/10.1016/j.jnca.2025.104176)
- **Method:** SecuEdge-DRL uses a MAPE-K self-adaptation loop and security policies; the publisher article describes a Raspberry Pi gateway evaluation with attack scenarios and CPU/energy measurements.
- **Limitations/relevance:** it uses CICIDS2017 and gateway-scale hardware, not the current reduced Edge-IIoTset artifact or ESP32. The publisher describes adaptive defense policies; the exact overlap with a model-recall constraint must be checked against the full method before claiming a gap.
- **Overlap/open question:** high overlap with adaptive security/energy trade-offs. SARA's name or a weighted risk score alone is not a research novelty.

### 6. Adaptive edge/cloud threat-detection collaboration

- **Citation:** Jawad Ahmad, Tamara Zhukabayeva, Shahid Latif, and Lazzat Zholshiyeva, “Adaptive Edge-Cloud Collaboration for Dynamic Threat Detection in IoT Networks,” *2025 IEEE Conference on Standards for Communications and Networking (CSCN)*. [DOI: 10.1109/CSCN67557.2025.11230705](https://doi.org/10.1109/CSCN67557.2025.11230705)
- **Method:** the IEEE abstract describes a context-aware engine using threat confidence, CPU, memory, and network latency to offload between a lightweight edge Random Forest and cloud DNN.
- **Limitations/relevance:** the public publisher abstract identifies IDSIoT2024; full experimental detail was not available in the retrieved publisher record. Results are not comparable to this project's dataset without matching the evaluation protocol.
- **Overlap/open question:** direct overlap with cross-layer resource-aware collaborative inference. The candidate project distinction would need a stricter, explicitly validated IDS security floor and adversarial/benign resource-stress evaluation; this remains unproven.

### 7. Federated IDS with distillation for Non-IID data

- **Citation:** Haonan Peng, Chunming Wu, and Yanfeng Xiao, “FD-IDS: Federated Learning with Knowledge Distillation for Intrusion Detection in Non-IID IoT Environments,” *Sensors*, vol. 25, no. 14, article 4309, 2025. [DOI: 10.3390/s25144309](https://doi.org/10.3390/s25144309)
- **Method:** uses mutual-information feature selection, a proximal term, and round-wise knowledge distillation, evaluated on Edge-IIoTset and N-BaIoT.
- **Limitations/relevance:** it addresses federated training heterogeneity rather than the SARA inference/offloading policy. Dataset preprocessing and partition details must be compared before importing reported results.
- **Overlap/open question:** FL, Non-IID partitions, and distillation are already explored together. This repository's prior DP/Krum/FL claims should not be presented as novel; its new CPU MLP FedAvg path is a reproducible software simulation, not an independent-node system.

### 8. Trust-aware FL IDS

- **Citation:** Muhammad Asad, Santosh Kumar Banbhrani, and Chen Wei, “Trust-Aware Federated Learning for Privacy-Preserving IoT Intrusion Detection,” *Electronics*, vol. 15, no. 20, article 4584, 2026. [DOI: 10.3390/electronics15204584](https://doi.org/10.3390/electronics15204584)
- **Method:** the publisher/institution record describes secure aggregation with local differential privacy, trust-based client filtering, gradient-deviation monitoring, and thresholded communication, evaluated on CICIDS2017 and Edge-IIoTset including poisoning settings.
- **Limitations/relevance:** publisher-level abstract only was used in this focused pass; formal privacy and robustness details require reading the full article and checking its assumptions.
- **Overlap/open question:** client trust, poisoning, privacy, and adaptive communication are active topics. The project must not reuse DP or Byzantine labels without implementing and validating the exact threat model and guarantee.

### 9. Resource-aware federated IDS with SDN

- **Citation:** Hossein Baghalzadeh, Nahideh Derakhshanfard, Masoud Kargar, and Ali Ghaffari, “A resource-aware framework for energy and bandwidth efficient IoT intrusion detection using lightweight federated learning over SDN,” *Scientific Reports*, 2026. [DOI: 10.1038/s41598-026-74166-3](https://doi.org/10.1038/s41598-026-74166-3)
- **Method:** hierarchical FL plus SDN-assisted IDS and resource/communication accounting, evaluated on five datasets including Edge-IIoTset.
- **Limitations/relevance:** the article itself distinguishes model-based energy/communication accounting from physical measurements and notes no formal privacy or Byzantine-robustness guarantees for its method.
- **Overlap/open question:** substantial overlap with resource-aware FL and network enforcement. The proposed system's general cross-layer/FL framing is therefore not by itself a gap.

## Comparison matrix

| Capability | Closest prior work found | Current AdaptiveML implementation | Candidate empirical question |
|---|---|---|---|
| Resource-aware model selection | EdgeAdaptor; Forough et al.; SecuEdge-DRL | Validation-macro-recall floors with estimated energy costs and host stress; no full baselines yet | Does a hard per-class security floor reduce attack false negatives under benign and malicious stress at acceptable measured cost? |
| Edge/cloud inference offload | Ahmad et al. (CSCN 2025) and collaborative edge inference literature | Authenticated HTTPS peer call with timeout and local fallback; no multi-node evaluation yet | Under measured loss/latency/battery conditions, when is remote inference preferable to local fallback? |
| Physical cross-layer telemetry | Tekin et al.; resource-aware IIoT testbeds | ESP32 telemetry and host resource API; device not connected or calibrated in this environment | What synchronized signals predict feasibility, and do they add value beyond network and model uncertainty without treating benign load as attack? |
| Federated IDS | Ferrag et al.; FD-IDS; Asad et al.; Baghalzadeh et al. | Local simulated CPU MLP FedAvg with Dirichlet partitions, validation gate, checkpoints; no separate network clients | How does resource-aware participation affect per-client minority-attack recall, convergence, and measured energy on independent clients? |
| Dataset/feature compatibility | Edge-IIoTset paper | Reduced 23-column local CSV; 18 ordered preprocessed features | Can the raw-data provenance be verified and can live extraction reproduce the exact schema without leakage or semantic mismatch? |

## Novelty assessment and decision

The current evidence **does not support a novelty claim** for adaptive energy-aware IDS, resource-aware model selection, edge/cloud threat offloading, federated IDS, or their broad combination. Closely related work already covers each and recent work overlaps several in one system. A narrower possible question is whether an experimentally calibrated per-class recall constraint prevents security degradation when an adaptive IDS is resource-stressed, and whether trusted offloading improves that outcome under measured network and device constraints. That is a hypothesis to investigate, not a verified research gap.

Before making a novelty claim, extend the review to systematic databases and exact algorithm-level comparisons, especially the full methods of SecuEdge-DRL, the 2025 edge/cloud collaboration paper, Forough et al., and 2026 resource-aware FL work. Compare datasets, feature contracts, threat models, resource-stress conditions, baselines, and physical measurement protocols. Report a gap only if those comparisons establish one.

## References

All links above point to publisher DOI pages, publisher article pages, or institutional author copies. No search-result-only statistic is treated as a result of this project.

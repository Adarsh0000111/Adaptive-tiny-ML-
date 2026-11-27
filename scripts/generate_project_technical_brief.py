"""Build the project/data/model/FedAvg technical brief PDF."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/adaptiveml-project-and-fedavg-brief.pdf"
NAVY = colors.HexColor("#14263D")
BLUE = colors.HexColor("#2463A6")
TEAL = colors.HexColor("#138A8A")
PALE = colors.HexColor("#EFF5FA")
PALE_TEAL = colors.HexColor("#EAF6F4")
INK = colors.HexColor("#203040")
MUTED = colors.HexColor("#596B7D")
LINE = colors.HexColor("#D6E0E8")
AMBER = colors.HexColor("#A65B08")

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_FILE = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_MONO_FILE = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
if FONT.exists() and FONT_BOLD_FILE.exists() and FONT_MONO_FILE.exists():
    pdfmetrics.registerFont(TTFont("DV", str(FONT)))
    pdfmetrics.registerFont(TTFont("DV-Bold", str(FONT_BOLD_FILE)))
    pdfmetrics.registerFont(TTFont("DV-Mono", str(FONT_MONO_FILE)))
    FONT_BODY, FONT_BOLD, FONT_MONO = "DV", "DV-Bold", "DV-Mono"
else:
    FONT_BODY, FONT_BOLD, FONT_MONO = "Helvetica", "Helvetica-Bold", "Courier"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", fontName=FONT_BOLD, fontSize=26, leading=32, textColor=NAVY, spaceAfter=8))
styles.add(ParagraphStyle(name="CoverSub", fontName=FONT_BODY, fontSize=12, leading=18, textColor=MUTED, spaceAfter=10))
styles.add(ParagraphStyle(name="H1x", fontName=FONT_BOLD, fontSize=19, leading=24, textColor=NAVY, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle(name="H2x", fontName=FONT_BOLD, fontSize=11.5, leading=15, textColor=BLUE, spaceBefore=7, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="Bodyx", fontName=FONT_BODY, fontSize=8.6, leading=12.2, textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="Smallx", fontName=FONT_BODY, fontSize=7.2, leading=9.6, textColor=INK))
styles.add(ParagraphStyle(name="Tinyx", fontName=FONT_BODY, fontSize=6.6, leading=8.4, textColor=MUTED))
styles.add(ParagraphStyle(name="CellHead", fontName=FONT_BOLD, fontSize=7.2, leading=9, textColor=colors.white))
styles.add(ParagraphStyle(name="Cell", fontName=FONT_BODY, fontSize=7, leading=9, textColor=INK))
styles.add(ParagraphStyle(name="CellSmall", fontName=FONT_BODY, fontSize=6.4, leading=8.1, textColor=INK))
styles.add(ParagraphStyle(name="Callout", fontName=FONT_BODY, fontSize=8.2, leading=11.5, textColor=NAVY))
styles.add(ParagraphStyle(name="Formula", fontName=FONT_MONO, fontSize=9, leading=14, textColor=NAVY))
styles.add(ParagraphStyle(name="KPI", fontName=FONT_BOLD, fontSize=15, leading=18, textColor=NAVY, alignment=1))
styles.add(ParagraphStyle(name="KPILabel", fontName=FONT_BODY, fontSize=7, leading=9, textColor=MUTED, alignment=1))


def P(text: str, style: str = "Bodyx") -> Paragraph:
    return Paragraph(text, styles[style])


def table(data, widths, *, header=True, row_bgs=True, font="Cell", padding=5):
    prepared = []
    for r, row in enumerate(data):
        prepared.append([cell if isinstance(cell, (Paragraph, Table)) else P(str(cell), "CellHead" if header and r == 0 else font) for cell in row])
    t = Table(prepared, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("LINEBELOW", (0, 0), (-1, 0), 0.8, BLUE)]
    if row_bgs:
        start = 1 if header else 0
        for i in range(start, len(data)):
            if (i - start) % 2 == 1:
                commands.append(("BACKGROUND", (0, i), (-1, i), PALE))
    t.setStyle(TableStyle(commands))
    return t


def callout(text: str, fill=PALE_TEAL, width=170 * mm):
    t = Table([[P(text, "Callout")]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LINEBEFORE", (0, 0), (0, 0), 3, TEAL if fill == PALE_TEAL else BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def page_chrome(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.setFont(FONT_BOLD, 7.5)
        canvas.drawString(20 * mm, h - 12 * mm, "ADAPTIVEML  /  SARA-IDS")
        canvas.setStrokeColor(LINE)
        canvas.line(20 * mm, h - 15 * mm, w - 20 * mm, h - 15 * mm)
    canvas.setStrokeColor(LINE)
    canvas.line(20 * mm, 14 * mm, w - 20 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_BODY, 7)
    canvas.drawString(20 * mm, 9 * mm, "Technical brief | 27 November 2026")
    canvas.drawRightString(w - 20 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def metric_pct(mapping, key):
    return f"{mapping[key] * 100:.2f}%"


def build():
    cnn = json.loads((ROOT / "results/experiments/high_cnn_evaluation.json").read_text())
    validation = json.loads((ROOT / "results/experiments/artifact_validation.json").read_text())
    old_fl = json.loads((ROOT / "results/experiments/networked_federated_local_demo.json").read_text())
    cnn_fl = json.loads((ROOT / "results/experiments/networked_federated_cnn_local_demo.json").read_text())
    dataset = np.load(ROOT / "data/processed/processed_dataset.npz", allow_pickle=False)
    names = [str(x) for x in dataset["feature_names"].tolist()]
    classes = [str(x) for x in dataset["class_names"].tolist()]
    train_counts = np.bincount(dataset["y_train_multi"], minlength=len(classes))
    output = OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(output), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm,
                          topMargin=22 * mm, bottomMargin=20 * mm, title="AdaptiveML / SARA-IDS Project and FedAvg Brief",
                          author="AdaptiveML project team")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="pages", frames=[frame], onPage=page_chrome)])
    story = []

    # Page 1: project snapshot
    story += [Spacer(1, 10 * mm), P("ADAPTIVEML / SARA-IDS", "H2x"),
              P("Project, models, data and<br/>federated learning", "CoverTitle"),
              P("A plain-language technical brief for the current implementation | 27 November 2026", "CoverSub"),
              Spacer(1, 5 * mm),
              callout("Current state: the high laptop tier is a trained 1D CNN; the low ESP32 tree and medium Random Forest remain. A two-client CNN FedAvg run has completed for two rounds using separate local processes over HTTP loopback. Its candidate did not pass the validation gate and remains separate from the deployed tiers. This is not a two-computer or independent-site experiment."),
              Spacer(1, 7 * mm), P("What the project does", "H2x"),
              P("The IDS classifies packet-level network traffic into six broad labels. SARA (Security-Aware Resource Adaptation) chooses an execution tier using threat/model evidence and available resources. Low-tier inference runs on the ESP32-S3 over USB; medium and high inference run on the laptop. The sensors provide physical or electrical context to SARA and are not treated as cyberattack evidence.")]
    kpis = [
        [P("18", "KPI"), P("3", "KPI"), P("6", "KPI")],
        [P("ordered packet inputs", "KPILabel"), P("inference tiers", "KPILabel"), P("traffic classes", "KPILabel")],
    ]
    story += [Spacer(1, 4 * mm), table(kpis, [56 * mm, 56 * mm, 56 * mm], header=False, row_bgs=False, padding=7),
              Spacer(1, 5 * mm), P("Current model roles", "H2x"),
              table([
                  ["Tier", "Model", "Where it runs", "Purpose"],
                  ["Low", "Fixed-point Decision Tree", "ESP32-S3 over USB", "Smallest local model; device reads one preprocessed packet vector."],
                  ["Medium", "Random Forest", "Laptop", "Intermediate CPU model for packet classification."],
                  ["High", "1D CNN", "Laptop", "Convolutional model over the 18 ordered feature fields."],
              ], [18 * mm, 43 * mm, 36 * mm, 71 * mm], font="CellSmall")]
    story += [PageBreak()]

    # Page 2: CNN details and evidence
    story += [P("01  |  Models and the high-tier CNN", "H1x"),
              P(f"The high model now loads as <b>1D CNN</b> in the API. Its architecture version is <font name='{FONT_MONO}'>packet-feature-cnn1d-18-32-64-dense-128-64-6-v2</font>. It uses Conv1D layers (1 to 32 channels, then 32 to 64), batch normalization, ReLU, max pooling, and a dense 128-to-64-to-6 classifier. The dense head also receives the original 18 values so pooling does not discard field positions.")]
    val_mlp = cnn["baseline_mlp_validation"]
    val_cnn = cnn["candidate_cnn_validation"]
    mlp_test = old_fl["baselines"]["heavy"]["test"]
    cnn_test = cnn["candidate_test_metrics"]
    story += [P("Validation and held-out test comparison", "H2x"),
              table([
                  ["High model", "Split", "Accuracy", "Macro recall", "Macro-F1"],
                  ["Prior MLP baseline", "Validation", metric_pct(val_mlp, "accuracy"), metric_pct(val_mlp, "macro_recall"), metric_pct(val_mlp, "macro_f1")],
                  ["1D CNN", "Validation", metric_pct(val_cnn, "accuracy"), metric_pct(val_cnn, "macro_recall"), metric_pct(val_cnn, "macro_f1")],
                  ["Prior MLP baseline", "Test", metric_pct(mlp_test, "accuracy"), metric_pct(mlp_test, "macro_recall"), metric_pct(mlp_test, "macro_f1")],
                  ["1D CNN", "Test", metric_pct(cnn_test, "accuracy"), metric_pct(cnn_test, "macro_recall"), metric_pct(cnn_test, "macro_f1")],
              ], [37 * mm, 25 * mm, 31 * mm, 39 * mm, 36 * mm]),
              Spacer(1, 3 * mm),
              callout("Promotion used validation only: the CNN had to remain within 2 percentage points of the prior MLP on validation accuracy, macro recall, and macro-F1. It passed. The test split was measured after that choice and was not used to pick or tune the CNN.", fill=PALE),
              Spacer(1, 4 * mm), P("Training setup", "H2x"),
              P(f"The central high-tier model trained on {cnn['data']['training_samples']:,} training rows for {cnn['training']['epochs_run']} epochs (batch size {cnn['training']['batch_size']}, learning rate {cnn['training']['learning_rate']}). It used square-root inverse-frequency class weights and validation loss for early stopping. Training took {cnn['training']['training_seconds']:.1f} seconds on the current host. The test set contains {cnn['data']['test_samples']:,} rows.")]
    story += [P("Important interpretation", "H2x"),
              P("This is a feature-axis CNN: the 1D convolution runs across the fixed order of feature columns. It is not a temporal CNN that consumes a sequence of packets. The current live input remains one packet-derived 18-value vector at a time. The saved split is packet-level and has no capture/session ID, so it does not establish generalization to independent network sessions.")]
    story += [PageBreak()]

    # Page 3: dataset and exact features
    story += [P("02  |  Dataset and the 18 inputs", "H1x"),
              P("The active pipeline uses the project's processed packet dataset, derived from the official Edge-IIoTset ML CSV. It retains 13 numeric fields and creates five one-hot protocol indicators. Current models and network clients expect exactly this ordered schema. An older 38-feature representation is not the active schema and must not be mixed with this model artifact.")]
    split = dataset
    story += [table([
        ["Split", "Rows", "Use"],
        ["Train", f"{len(split['y_train_multi']):,}", "Fit model weights; local FedAvg client partitions come from this split in the one-PC demo."],
        ["Validation", f"{len(split['y_val_multi']):,}", "Early stopping, comparisons, and candidate promotion gates."],
        ["Test", f"{len(split['y_test_multi']):,}", "Final report after model selection; not used for model selection."],
    ], [30 * mm, 27 * mm, 115 * mm]), Spacer(1, 4 * mm),
        table([
            ["#", "Feature", "Field meaning / representation"],
            ["1", names[0], "TCP FIN flag indicator."],
            ["2", names[1], "TCP RST flag indicator."],
            ["3", names[2], "TCP SYN without ACK indicator."],
            ["4", names[3], "TCP SYN plus ACK indicator."],
            ["5", names[4], "TCP ACK flag indicator."],
            ["6", names[5], "TCP flag bitmask."],
            ["7", names[6], "TCP payload length in bytes."],
            ["8", names[7], "TCP destination port."],
            ["9", names[8], "UDP inter-packet time delta in the dataset."],
            ["10", names[9], "DNS query-name length in the dataset."],
            ["11", names[10], "HTTP response indicator."],
            ["12", names[11], "MQTT control/message type."],
            ["13", names[12], "MQTT remaining length."],
            ["14-18", "protocol_http, protocol_mqtt, protocol_other, protocol_tcp, protocol_udp", "Five one-hot values; exactly one indicates the selected protocol class."],
        ], [13 * mm, 65 * mm, 94 * mm], font="CellSmall", padding=3)]
    story += [Spacer(1, 4 * mm), P("Six output classes", "H2x"),
              P("Access_Compromise (Backdoor, MITM, Password); DDoS (HTTP, ICMP, TCP, UDP); Normal; Ransomware; Scanning (Fingerprinting, OS fingerprinting, port scanning, vulnerability scanner); Web_Attack (SQL injection, Uploading, XSS). The numeric model labels are 0 through 5 in the order shown in the saved class mapping.")]
    class_table = [["Class", "Train rows", "Class", "Train rows"]]
    for i in range(3):
        row = []
        for class_id in (i, i + 3):
            row.extend([f"{class_id}: {classes[class_id]}", f"{int(train_counts[class_id]):,}"])
        class_table.append(row)
    story += [Spacer(1, 2 * mm), table(class_table, [55 * mm, 30 * mm, 55 * mm, 30 * mm], font="CellSmall", padding=3)]
    story += [PageBreak()]

    # Page 4: packets and SARA/sensor roles
    story += [P("03  |  Packet path and SARA's decisions", "H1x"),
              P("The IDS features come from packet metadata, not directly from the sensor board. Captured TCP/UDP packets are mapped to the training schema, transformed with the saved imputer/scaler/one-hot encoder, and then given to SARA and the selected model.")]
    flow = [
        [P("TCP/UDP<br/>packet", "CellHead"), P("Extract packet<br/>metadata", "CellHead"), P("Preprocess into<br/>18 ordered values", "CellHead"), P("SARA chooses<br/>execution tier", "CellHead"), P("Class + confidence<br/>to dashboard", "CellHead")],
        [P("No saved payloads or IPs", "CellSmall"), P("Packet-level, not flow-level", "CellSmall"), P("Train-fitted transform", "CellSmall"), P("Low: USB/ESP32<br/>Medium/high: laptop", "CellSmall"), P("Live labels are unknown", "CellSmall")],
    ]
    flow_table = Table(flow, colWidths=[34 * mm, 34 * mm, 37 * mm, 37 * mm, 30 * mm])
    flow_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE), ("BACKGROUND", (0, 1), (-1, 1), PALE),
                                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                    ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                                    ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story += [flow_table, Spacer(1, 5 * mm), P("What SARA uses and what the sensors mean", "H2x"),
              table([
                  ["Signal", "Scientific/operational role in this project"],
                  ["Packet model outputs", "Threat class, confidence, normalized uncertainty, and disagreement across low/medium/high model predictions inform risk and escalation."],
                  ["Host compute", "CPU and RAM pressure describe the laptop/API host; they affect resource-aware execution or FL scheduling, not threat labels by themselves."],
                  ["INA219 voltage/current/power", "Only fresh and electrically plausible readings are qualified. Voltage/current may block ESP32 execution at configured limits. Power is checked against V x I; accumulated energy is logged, not interpreted as battery charge."],
                  ["DHT22 temperature/humidity", "Ambient context only. It is not the ESP32 die temperature and is not attack evidence."],
                  ["MPU6500 + SW-420", "Movement/vibration event context only. Physical vibration is not treated as a cyberattack signal."],
                  ["Network link measurements", "RTT, packet loss, and link utilization can gate networked FL readiness when fresh. Aggregate host counters are not substituted for the model's packet feature schema."],
              ], [42 * mm, 130 * mm], font="CellSmall"), Spacer(1, 3 * mm),
              callout("Live feature parity caveat: DNS query-name length is now extracted as a length to match the trained field. UDP timing is still an online approximation rather than a guaranteed match to the dataset capture semantics; validate live feature parity before making live accuracy claims.", fill=colors.HexColor("#FFF5E8"))]
    story += [PageBreak()]

    # Page 5: FedAvg
    story += [P("04  |  FedAvg: data segregation and aggregation", "H1x"),
              P("In the authenticated network path, each computer holds its own labeled client dataset and trains a local copy of the shared CNN. The coordinator starts from the current centrally trained high-tier CNN and returns model weights for each round. Client examples and labels stay on the client.")]
    story += [P("How the demo shards were made", "H2x"),
              P("The two-client demo partitions the project's 25,579-row training split by class using a Dirichlet distribution. For each class c, draw client proportions p_c ~ Dirichlet(alpha * 1_K), with alpha = 0.5 and K = 2; shuffle that class's row indices, split them using p_c, then combine each client's class portions. Smaller alpha generally creates more unequal class shares. A fixed seed made the run reproducible.")]
    story += [P("Recorded shared-data shard sizes: node-a = 13,462 rows; node-b = 12,117 rows. They are disjoint shards of the same training dataset, not independently collected site data.", "Callout"),
              Spacer(1, 3 * mm), P("One FedAvg round", "H2x"),
              table([
                  ["Step", "What happens"],
                  ["1. Assign", "Coordinator sends current CNN weights, round ID, 18-feature/six-class schema, preprocessing hash, and shared class weights."],
                  ["2. Train locally", "Each client initializes the same CNN from the assigned weights and trains on its own X_i, y_i rows."],
                  ["3. Submit", "Client sends updated CNN parameter tensors, sample count n_i, and scalar local loss. It does not upload X_i, y_i, packet payloads, or packet rows."],
                  ["4. Aggregate", "Coordinator averages compatible parameter tensors, weighted by each client's sample count."],
                  ["5. Validate", "Coordinator evaluates the candidate on its own held-out validation split. Best validation candidate is saved; only a candidate passing the gate can replace the high tier."],
              ], [32 * mm, 140 * mm], font="CellSmall"), Spacer(1, 4 * mm),
              callout("Sample-weighted FedAvg: w_global(t+1) = sum over participating clients i of (n_i / N) * w_i(t+1), where N = sum_i n_i. Each w_i(t+1) is the locally trained model parameter set returned by client i.", fill=PALE),
              Spacer(1, 4 * mm), P("Privacy and trust limits", "H2x"),
              P("The network protocol uses HTTPS for remote coordinator connections and node authentication tokens, but it has no secure aggregation or differential privacy. Model updates can still reveal information. The coordinator validates with its own saved validation split; it does not receive client feature rows. Client ground-truth labels must be trustworthy. The ESP32 is not a training client.")]
    story += [PageBreak()]

    # Page 6: evidence, what has/hasn't happened, runbook
    story += [P("05  |  Evidence and what remains", "H1x"),
              P("Separate completed training evidence from code capability. The central high CNN is trained and active. Both FedAvg implementations use the same CNN architecture. The recorded CNN FedAvg result below is a two-process loopback run; it does not establish communication between separate computers or independent-site performance.")]
    old_candidate = old_fl["candidate"]["validation"]
    cnn_fl = json.loads((ROOT / "results/experiments/networked_federated_cnn_local_demo.json").read_text())
    best_cnn_fl = cnn_fl["candidate"]["best_validation"]
    round_rows = cnn_fl["candidate"]["rounds"]
    story += [table([
        ["Evidence item", "Status and exact scope"],
        ["High-tier 1D CNN", f"Trained centrally and active. Test: {metric_pct(cnn_test, 'accuracy')} accuracy, {metric_pct(cnn_test, 'macro_recall')} macro recall, {metric_pct(cnn_test, 'macro_f1')} macro-F1."],
        ["CNN + FedAvg", "Two authenticated client processes completed 2 HTTP loopback rounds, one local epoch per round; both clients participated in both rounds (4 accepted updates)."],
        ["Best FedAvg candidate", f"Validation: {metric_pct(best_cnn_fl, 'accuracy')} accuracy, {metric_pct(best_cnn_fl, 'macro_recall')} macro recall, {metric_pct(best_cnn_fl, 'macro_f1')} macro-F1. It failed the gate (required 90% macro recall) and was saved separately; the deployed high CNN was not replaced."],
        ["Round validation", f"Round 1: {metric_pct(round_rows[0]['validation_metrics'], 'accuracy')} accuracy / {metric_pct(round_rows[0]['validation_metrics'], 'macro_recall')} macro recall / {metric_pct(round_rows[0]['validation_metrics'], 'macro_f1')} macro-F1. Round 2: {metric_pct(round_rows[1]['validation_metrics'], 'accuracy')} / {metric_pct(round_rows[1]['validation_metrics'], 'macro_recall')} / {metric_pct(round_rows[1]['validation_metrics'], 'macro_f1')}. Test set was not evaluated for this run."],
        ["Older MLP run", f"Historical 60-round demo used MLP, not CNN; its candidate validation recall was {old_candidate['macro_recall']*100:.2f}%, and it was not promoted."],
        ["Independent-site / multi-PC data", "Not demonstrated. Both local demo shards came from one training split on one host. Real clients need separate locally labeled packet data and a run across separate computers."],
        ["Generalization", "The dataset split is random at packet level and has no session/capture IDs. Independent labeled captures are needed to validate real network performance."],
    ], [43 * mm, 129 * mm], font="CellSmall"), Spacer(1, 4 * mm),
              P("Next: run across separate computers", "H2x"),
              P("Prepare independently collected, locally labeled packet data on each client using the project's exact 18-feature pipeline. Configure authenticated node IDs/tokens. Put the coordinator behind HTTPS reachable by both clients, then run one client process on each separate computer. Verify every round's participants and accepted updates through the coordinator status endpoint.")]
    story += [table([
        ["Coordinator", "python -m src.api.app (configure SARA_FL_NODE_TOKENS and coordinator auth first)"],
        ["Client data", "python scripts/prepare_fl_client_data.py --csv client-labeled.csv --out node-a.npz"],
        ["Client process", "python -m src.federated.client --coordinator https://COORDINATOR:8765 --node-id node-a --data node-a.npz --wait-for-run"],
        ["Verify", "GET /api/federated/networked/status: complete run, 2+ participating nodes per round, accepted updates, validation metrics, and promotion result."],
    ], [30 * mm, 142 * mm], header=False, row_bgs=True, font="CellSmall", padding=4),
              Spacer(1, 4 * mm),
              P("Source dataset attribution", "H2x"),
              P("Mohamed Amine Ferrag et al., Edge-IIoTset, DOI 10.1109/ACCESS.2022.3165809. The project's provenance records the dataset license as CC BY-NC-SA 4.0; follow its attribution, share-alike, and non-commercial terms."),
              P("Implementation sources: README.md; src/data/edge_packet_schema.py; src/preprocessing/pipeline.py; src/models/cnn.py; scripts/train_high_cnn.py; src/federated/client.py; src/federated/networked.py; results/experiments/high_cnn_evaluation.json; results/experiments/networked_federated_local_demo.json; results/experiments/networked_federated_cnn_local_demo.json.", "Tinyx")]

    doc.build(story)
    print(output.resolve())


if __name__ == "__main__":
    build()

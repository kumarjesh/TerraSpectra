import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print accurate total page count
    along with running headers and footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Running Header (pages 2+)
        if self._pageNumber > 1:
            self.setFont('Helvetica-Bold', 7.5)
            self.setFillColor(colors.HexColor('#1b4332'))
            self.drawString(40, 755, "TERRASPECTRA &bull; WEEK 3 IMPLEMENTATION PLAN")
            self.setFont('Helvetica', 7.5)
            self.setFillColor(colors.HexColor('#64748b'))
            self.drawRightString(572, 755, "Hybrid 3D-CNN + Vision Transformer Architecture")
            self.setStrokeColor(colors.HexColor('#cbd5e1'))
            self.setLineWidth(0.5)
            self.line(40, 750, 572, 750)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor('#e2e8f0'))
        self.setLineWidth(0.5)
        self.line(40, 36, 572, 36)
        
        self.setFont('Helvetica', 7.5)
        self.setFillColor(colors.HexColor('#64748b'))
        self.drawString(40, 26, "TerraSpectra &bull; Deep Spectral-Spatial Learning &amp; Model Explainability")
        self.drawRightString(572, 26, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf(filename="TerraSpectra_Week3_Implementation_Plan.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    PRIMARY = colors.HexColor('#1b4332')       # Deep Forest Green
    SECONDARY = colors.HexColor('#2d6a4f')     # Emerald Green
    TEXT_DARK = colors.HexColor('#0f172a')     # Slate 900
    TEXT_MUTED = colors.HexColor('#475569')    # Slate 600
    BG_CARD = colors.HexColor('#f1f5f9')       # Slate 100
    BORDER = colors.HexColor('#cbd5e1')        # Slate 300
    ALERT_BG = colors.HexColor('#ecfdf5')      # Emerald 50
    ALERT_BORDER = colors.HexColor('#10b981')  # Emerald 500

    styles.add(ParagraphStyle(
        name='CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=21,
        textColor=PRIMARY,
        spaceAfter=2
    ))

    styles.add(ParagraphStyle(
        name='CoverSubtitle',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=SECONDARY,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name='SectionHeaderCustom',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=PRIMARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='SubSectionHeaderCustom',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=SECONDARY,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='BodyCustom',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=TEXT_DARK,
        spaceAfter=3
    ))

    styles.add(ParagraphStyle(
        name='BulletCustom',
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=TEXT_DARK,
        leftIndent=10,
        firstLineIndent=-7,
        spaceAfter=2
    ))

    styles.add(ParagraphStyle(
        name='CodeSnippet',
        fontName='Courier',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#0f172a'),
        backColor=BG_CARD,
        spaceAfter=3
    ))

    styles.add(ParagraphStyle(
        name='TableHead',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    ))

    styles.add(ParagraphStyle(
        name='TableCell',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=TEXT_DARK
    ))

    styles.add(ParagraphStyle(
        name='TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=TEXT_DARK
    ))

    styles.add(ParagraphStyle(
        name='CalloutText',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#064e3b')
    ))

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("TERRASPECTRA &bull; WEEK 3 IMPLEMENTATION PLAN", styles['CoverTitle']))
    story.append(Paragraph("Hybrid 3D-CNN + Vision Transformer Architecture &amp; Explainability &bull; Technical Blueprint", styles['CoverSubtitle']))

    # Metadata Row Table
    meta_data = [
        [
            Paragraph("<b>Target Domain:</b> Precision Ag / Advanced CV", styles['TableCell']),
            Paragraph("<b>Sprint Objective:</b> 3D-CNN + ViT Hybrid Model", styles['TableCell']),
            Paragraph("<b>Input Modality:</b> 3D Patches (16&times;16&times;16)", styles['TableCell'])
        ],
        [
            Paragraph("<b>Feature Extraction:</b> 3D-CNN Spatial-Spectral", styles['TableCell']),
            Paragraph("<b>Global Attention:</b> ViT Self-Attention Encoder", styles['TableCell']),
            Paragraph("<b>Interpretability:</b> Spectral Saliency (Gradients)", styles['TableCell'])
        ]
    ]
    meta_table = Table(meta_data, colWidths=[177, 177, 178])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_CARD),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 3))

    story.append(Paragraph("1. Week 3 Mission &amp; Architectural Rationale", styles['SectionHeaderCustom']))
    story.append(HRFlowable(width="100%", thickness=1.2, color=PRIMARY, spaceBefore=1, spaceAfter=4))

    story.append(Paragraph(
        "<b>The Deep Learning Motivation:</b> While the Week 2 baseline Random Forest achieved strong performance, standard pixel-wise classifiers treat neighboring spatial pixels independently and cannot capture subtle textural canopy variations or long-range cross-band correlations. Standard 2D-CNNs flatten spectral bands into 2D image channels, completely discarding continuous physical wavelength curves. Pure Vision Transformers (ViTs) lack inductive spatial biases and require massive labeled datasets that do not exist in agricultural hyperspectral imaging.",
        styles['BodyCustom']
    ))
    story.append(Paragraph(
        "<b>The Hybrid 3D-CNN + ViT Solution:</b> Week 3 implements an elegant two-stage hybrid neural architecture in PyTorch:",
        styles['BodyCustom']
    ))
    story.append(Paragraph("&bull; <b>Stage 1 (3D Convolutional Frontend):</b> Uses 3D convolution kernels $(K_d \times K_h \times K_w = 3 \times 3 \times 3)$ to simultaneously extract localized spatial context (crop canopy geometry, row structure) and local spectral continuity (smooth absorption gradients).", styles['BulletCustom']))
    story.append(Paragraph("&bull; <b>Stage 2 (Vision Transformer Encoder):</b> Flattens and projects the 3D feature volume into spectral sequence tokens, using Multi-Head Self-Attention to discover global, non-linear correlations across disparate wavelengths (e.g., correlating the 700 nm red-edge shift with 1400 nm SWIR moisture absorption).", styles['BulletCustom']))

    # Callout Box
    callout_data = [[
        Paragraph(
            "<b>Week 3 Objectives:</b><br/>"
            "1. <b>PyTorch Hybrid Network:</b> Implement <code>Hybrid3DCNN_ViT</code> combining 3D convolutions with a 2-layer ViT encoder.<br/>"
            "2. <b>Spatial Cross-Validation:</b> Enforce field-block holdout training to guarantee true agricultural generalization.<br/>"
            "3. <b>Model Explainability (Spectral Saliency):</b> Calculate gradient-based feature attribution to prove the model targets the red edge (690–740 nm).<br/>"
            "4. <b>Analytics Side Panel:</b> Expand the GIS dashboard with live spectral anomaly curves and management zone metrics.",
            styles['CalloutText']
        )
    ]]
    callout_table = Table(callout_data, colWidths=[532])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), ALERT_BG),
        ('BOX', (0,0), (-1,-1), 0.8, ALERT_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 3))

    story.append(Paragraph("2. Week 3 Day-by-Day Implementation Roadmap", styles['SectionHeaderCustom']))
    story.append(HRFlowable(width="100%", thickness=1.2, color=PRIMARY, spaceBefore=1, spaceAfter=4))

    roadmap_data = [
        [Paragraph("Day", styles['TableHead']), Paragraph("Module / Focus Area", styles['TableHead']), Paragraph("Technical Engineering Tasks", styles['TableHead']), Paragraph("Verification Gate", styles['TableHead'])],
        [
            Paragraph("<b>Day 1</b>", styles['TableCellBold']),
            Paragraph("3D Patch Dataset Generator", styles['TableCell']),
            Paragraph("&bull; Build sliding-window patch extractor: $(16 \times 16 \times 16)$ spatial-spectral tensors.<br/>&bull; Enforce field-separated spatial holdouts.<br/>&bull; Implement class balancing to handle 5-acre disease scarcity.", styles['TableCell']),
            Paragraph("Dataset tensors shape: <code>(N, 1, 16, 16, 16)</code> verified with zero data leakage.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Day 2</b>", styles['TableCellBold']),
            Paragraph("3D-CNN Frontend Engineering", styles['TableCell']),
            Paragraph("&bull; Build 3D Conv layers with BatchNorm3d and ReLU.<br/>&bull; Implement spatial pooling to reduce token dimensionality.<br/>&bull; Linear projection layer to Transformer embedding space ($D=64$).", styles['TableCell']),
            Paragraph("3D convolution forward pass executes in &lt;15 ms per patch.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Day 3</b>", styles['TableCellBold']),
            Paragraph("Vision Transformer (ViT) Encoder", styles['TableCell']),
            Paragraph("&bull; Implement Multi-Head Self-Attention ($H=4$, 2 encoder layers).<br/>&bull; Add learnable 1D spectral positional encodings.<br/>&bull; Global spectral average pooling and 3-class classification head.", styles['TableCell']),
            Paragraph("Full forward pass tensor shapes verified: <code>(B, 3)</code> logits.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Day 4</b>", styles['TableCellBold']),
            Paragraph("Training &amp; Spatial Cross-Validation", styles['TableCell']),
            Paragraph("&bull; Train hybrid model using Weighted Cross-Entropy Loss.<br/>&bull; AdamW optimizer with cosine learning rate schedule.<br/>&bull; Early stopping based on holdout validation F1-score.", styles['TableCell']),
            Paragraph("Disease-class F1-score &ge; 90.0% on spatially separated field holdouts.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Day 5</b>", styles['TableCellBold']),
            Paragraph("Model Explainability &amp; Saliency", styles['TableCell']),
            Paragraph("&bull; Compute gradient-based feature attribution w.r.t input bands.<br/>&bull; Generate <code>data/spectral_saliency.json</code>.<br/>&bull; Verify PC2 (red-edge inflection) is the primary diagnostic driver.", styles['TableCell']),
            Paragraph("Saliency confirms &gt;60% attribution concentrated on red-edge and water bands.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Day 6</b>", styles['TableCellBold']),
            Paragraph("Analytics Panel &amp; GIS Integration", styles['TableCell']),
            Paragraph("&bull; Integrate Recharts/Chart.js anomaly comparison in web dashboard.<br/>&bull; Display top diagnostic drivers and feature attributions in UI.<br/>&bull; Refine Zone B4 drill-down telemetry.", styles['TableCell']),
            Paragraph("Dashboard displays side-by-side healthy vs. stressed spectral curve comparison.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Day 7</b>", styles['TableCellBold']),
            Paragraph("Verification Suite &amp; Week 3 Gate", styles['TableCell']),
            Paragraph("&bull; Automated pytest suite for 3D tensor shapes, training, and saliency.<br/>&bull; Compare hybrid 3D-CNN+ViT metrics vs. Week 2 baseline.<br/>&bull; Prepare Week 3 presentation deliverables.", styles['TableCell']),
            Paragraph("100% test pass rate. Hybrid outperforms baseline F1-score.", styles['TableCell'])
        ]
    ]

    roadmap_table = Table(roadmap_data, colWidths=[40, 120, 262, 110])
    roadmap_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(roadmap_table)

    story.append(PageBreak())

    # ==================== PAGE 2 ====================
    story.append(Paragraph("3. Technical Deep Dive: 3D-CNN + ViT Hybrid Architecture", styles['SectionHeaderCustom']))
    story.append(HRFlowable(width="100%", thickness=1.2, color=PRIMARY, spaceBefore=1, spaceAfter=4))

    story.append(Paragraph("3.1 Mathematical Architecture Formulation", styles['SubSectionHeaderCustom']))
    story.append(Paragraph(
        "Let the input patch be X &isin; &real;<sup>B &times; 1 &times; D &times; H &times; W</sup>, where batch size is B, channel count is 1, spectral depth is D = 16 (from PCA), and spatial patch size is H = W = 16.",
        styles['BodyCustom']
    ))
    story.append(Paragraph("&bull; <b>3D Convolutional Operation:</b> The 3D kernel K &isin; &real;<sup>C<sub>out</sub> &times; C<sub>in</sub> &times; K<sub>d</sub> &times; K<sub>h</sub> &times; K<sub>w</sub></sup> computes localized spatio-spectral activations: Y(d, h, w) = &sum; X(c, d+i, h+j, w+k) &middot; K(c, i, j, k). This preserves continuous band transitions without flattening.", styles['BulletCustom']))
    story.append(Paragraph("&bull; <b>Spectral-Spatial Tokenization:</b> After 3D convolution and spatial max pooling (1, 2, 2), the tensor has shape (B, 16, D, 8, 8). We permute and reshape to (B, D, 16 &times; 8 &times; 8) and apply linear projection W<sub>proj</sub> &isin; &real;<sup>1024 &times; d<sub>model</sub></sup> to obtain spectral sequence tokens T &isin; &real;<sup>B &times; D &times; 64</sup>.", styles['BulletCustom']))
    story.append(Paragraph("&bull; <b>Multi-Head Self-Attention (MHSA):</b> For each head h, queries, keys, and values are projected: Q<sub>h</sub> = T &middot; W<sub>Q</sub>, K<sub>h</sub> = T &middot; W<sub>K</sub>, V<sub>h</sub> = T &middot; W<sub>V</sub>. Attention weights capture non-local inter-band dependencies: Attention(Q, K, V) = softmax((Q &middot; K<sup>T</sup>) / &radic;d<sub>k</sub>) &middot; V.", styles['BulletCustom']))

    story.append(Spacer(1, 2))

    story.append(Paragraph("3.2 Model Explainability &amp; Spectral Saliency (Feature Attribution)", styles['SubSectionHeaderCustom']))
    story.append(Paragraph(
        "Black-box neural networks are untrusted by agronomists. We compute gradient-based saliency maps to quantify which spectral components drive the prediction for Class 2 (Pre-Symptomatic Infection):",
        styles['BodyCustom']
    ))
    
    code_text = (
        "# Spectral Band Gradient Attribution\n"
        "model.eval()\n"
        "input_patch.requires_grad = True\n"
        "logits = model(input_patch)\n"
        "loss = logits[0, target_class_2]  # Target: Pre-Symptomatic Infection\n"
        "loss.backward()\n"
        "# Saliency is mean absolute gradient across spatial patch dimensions:\n"
        "spectral_importance = input_patch.grad.abs().squeeze().mean(dim=(1, 2))\n"
    )
    story.append(Paragraph(code_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles['CodeSnippet']))

    story.append(Paragraph("3.3 Model Comparison Matrix: Baseline vs. Hybrid 3D-CNN + ViT", styles['SubSectionHeaderCustom']))
    comp_data = [
        [Paragraph("Evaluation Metric / Property", styles['TableHead']), Paragraph("Week 2 Baseline (Random Forest)", styles['TableHead']), Paragraph("Week 3 Hybrid (3D-CNN + ViT)", styles['TableHead']), Paragraph("Architectural Advantage", styles['TableHead'])],
        [
            Paragraph("<b>Spatial Context</b>", styles['TableCellBold']),
            Paragraph("Pixel-wise only (0 spatial context)", styles['TableCell']),
            Paragraph("<b>16 &times; 16 spatial neighborhood</b>", styles['TableCell']),
            Paragraph("Filters out isolated pixel noise using canopy texture.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Spectral Correlation</b>", styles['TableCellBold']),
            Paragraph("Orthogonal linear projection (PCA)", styles['TableCell']),
            Paragraph("<b>Non-linear Multi-Head Attention</b>", styles['TableCell']),
            Paragraph("Discovers complex cross-band chlorophyll interactions.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Disease-Class F1-Score</b>", styles['TableCellBold']),
            Paragraph("90.23% (on spatial holdout)", styles['TableCell']),
            Paragraph("<b>&gt; 92.5% - 95.0%</b>", styles['TableCell']),
            Paragraph("Reduces false positives across healthy crop borders.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Inference Latency</b>", styles['TableCellBold']),
            Paragraph("~8.2 ms / tile (CPU)", styles['TableCell']),
            Paragraph("<b>~22.4 ms / tile (GPU / CPU)</b>", styles['TableCell']),
            Paragraph("Easily maintains &lt;30s full 1,000-acre scene inference.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Explainability</b>", styles['TableCellBold']),
            Paragraph("Gini feature importance", styles['TableCell']),
            Paragraph("<b>Gradient-based Spectral Saliency</b>", styles['TableCell']),
            Paragraph("Directly proves sensitivity to red-edge inflection shift.", styles['TableCell'])
        ]
    ]
    comp_table = Table(comp_data, colWidths=[110, 120, 130, 172])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(comp_table)

    story.append(Spacer(1, 2))

    story.append(Paragraph("3.4 Week 3 Verification Gate &amp; Acceptance Criteria", styles['SubSectionHeaderCustom']))
    gate_data = [
        [Paragraph("Acceptance Gate", styles['TableHead']), Paragraph("Required Threshold", styles['TableHead']), Paragraph("Verification Protocol", styles['TableHead'])],
        [
            Paragraph("<b>1. Tensor Dimensions</b>", styles['TableCellBold']),
            Paragraph("Output shape: <code>(Batch, 3)</code>", styles['TableCell']),
            Paragraph("Assert model forward pass accepts (B, 1, 16, 16, 16) and outputs 3 class logits.", styles['TableCell'])
        ],
        [
            Paragraph("<b>2. Model Generalization</b>", styles['TableCellBold']),
            Paragraph("Pre-Symptomatic F1 &ge; 90.0%", styles['TableCell']),
            Paragraph("Evaluate on spatially separated field holdouts to verify zero autocorrelation leakage.", styles['TableCell'])
        ],
        [
            Paragraph("<b>3. Biophysical Saliency</b>", styles['TableCellBold']),
            Paragraph("Red-Edge driver &gt; 50% importance", styles['TableCell']),
            Paragraph("Verify that gradient attribution ranks PC2 (Red-Edge) as top diagnostic factor.", styles['TableCell'])
        ],
        [
            Paragraph("<b>4. Memory Profile</b>", styles['TableCellBold']),
            Paragraph("Peak RAM &lt; 150 MB during training", styles['TableCell']),
            Paragraph("Profile patch extraction and GPU batch tensor memory using <code>tracemalloc</code>.", styles['TableCell'])
        ]
    ]
    gate_table = Table(gate_data, colWidths=[110, 125, 297])
    gate_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(gate_table)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf()

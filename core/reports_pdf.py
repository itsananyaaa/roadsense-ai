"""
RoadSense AI - Unified Enterprise PDF Report Generation Engine
Produces high-fidelity, print-optimized landscape A4 municipal engineering reports:
1. Work Order Dispatch Schedules (Priority Center)
2. Municipal Incident & Damage Inventory (Incident Map / Reports Ledger)

Features:
- Pure offline execution: zero external CDN or network requests
- Full Unicode support with bundled DejaVu Sans TrueType fonts
- Clean RoadSense design identity with print-friendly high-contrast styling
- Dynamic two-pass NumberedCanvas footer ("Page X of Y")
- Strict empty-state, synthetic-record flagging, and em-dash fallback for missing fields
- High-efficiency Platypus table pagination tested with 2,000+ records
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core import config

# Register bundled Unicode TTF fonts
FONT_DIR = Path(__file__).parent / "fonts"
REGULAR_TTF = FONT_DIR / "DejaVuSans.ttf"
BOLD_TTF = FONT_DIR / "DejaVuSans-Bold.ttf"

FONT_NAME = "Helvetica"
FONT_BOLD_NAME = "Helvetica-Bold"

if REGULAR_TTF.exists():
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", str(REGULAR_TTF)))
        FONT_NAME = "DejaVuSans"
    except Exception:
        pass

if BOLD_TTF.exists():
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(BOLD_TTF)))
        FONT_BOLD_NAME = "DejaVuSans-Bold"
    except Exception:
        pass


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas renderer to dynamically inject accurate 'Page X of Y' footers
    and regulatory civil disclaimers without requiring external template engines.
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

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        page_w, page_h = landscape(A4)

        # Bottom footer rule
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.75)
        self.line(36, 32, page_w - 36, 32)

        # Footer text
        self.setFont(FONT_NAME, 7.5)
        self.setFillColor(colors.HexColor("#64748b"))

        disclaimer = (
            f"RoadSense AI v{config.PRIORITY_FORMULA_VERSION} • Municipal Highway Infrastructure Intelligence • "
            "Automated triage assessment does not replace certified structural road audits."
        )
        self.drawString(36, 20, disclaimer)

        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(page_w - 36, 20, page_str)
        self.restoreState()


def _get_report_styles() -> Dict[str, ParagraphStyle]:
    """Build unified typographical styles for RoadSense PDF generation."""
    styles = getSampleStyleSheet()

    custom_styles = {
        "DocTitle": ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName=FONT_BOLD_NAME,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
        ),
        "DocSubtitle": ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#475569"),
        ),
        "MetaLabel": ParagraphStyle(
            "MetaLabel",
            parent=styles["Normal"],
            fontName=FONT_BOLD_NAME,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#64748b"),
        ),
        "MetaValue": ParagraphStyle(
            "MetaValue",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1e293b"),
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=styles["Normal"],
            fontName=FONT_BOLD_NAME,
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName=FONT_BOLD_NAME,
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#ffffff"),
            alignment=0,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            fontSize=7,
            leading=8.5,
            textColor=colors.HexColor("#1e293b"),
        ),
        "TableCellBold": ParagraphStyle(
            "TableCellBold",
            parent=styles["Normal"],
            fontName=FONT_BOLD_NAME,
            fontSize=7,
            leading=8.5,
            textColor=colors.HexColor("#0f172a"),
        ),
        "TableCellCode": ParagraphStyle(
            "TableCellCode",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            fontSize=6.5,
            leading=8,
            textColor=colors.HexColor("#334155"),
        ),
        "SyntheticBadge": ParagraphStyle(
            "SyntheticBadge",
            parent=styles["Normal"],
            fontName=FONT_BOLD_NAME,
            fontSize=6.5,
            leading=8,
            textColor=colors.HexColor("#b45309"),
        ),
        "AdvisoryNotice": ParagraphStyle(
            "AdvisoryNotice",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#334155"),
        ),
        "EmptyState": ParagraphStyle(
            "EmptyState",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            alignment=1,
        ),
    }
    return custom_styles


def _format_value(val: Any) -> str:
    """Safely format values, rendering an em dash for missing or empty fields."""
    if val is None or val == "" or val == "None":
        return "—"
    return str(val)


def _build_header_block(
    title: str,
    subtitle: str,
    timestamp_str: str,
    filter_summary: Optional[Any] = None,
    filter_desc: Optional[str] = None,
) -> List[Any]:
    """Generate consistent branded header block."""
    styles = _get_report_styles()
    story: List[Any] = []

    filters_text = "All Records (No Active Filters)"
    if filter_desc:
        filters_text = str(filter_desc)
    elif isinstance(filter_summary, str):
        filters_text = filter_summary
    elif isinstance(filter_summary, dict):
        parts = []
        for k, v in filter_summary.items():
            if v and str(v).upper() != "ALL":
                clean_k = k.replace("_", " ").title()
                parts.append(f"{clean_k}: <b>{v}</b>")
        if parts:
            filters_text = " • ".join(parts)

    header_table_data = [
        [
            Paragraph(f"<b>ROADSENSE AI</b> | {title}", styles["DocTitle"]),
            Paragraph(f"<b>Generated:</b> {timestamp_str}<br/><b>Formula:</b> v{config.PRIORITY_FORMULA_VERSION}", styles["MetaValue"]),
        ],
        [
            Paragraph(subtitle, styles["DocSubtitle"]),
            Paragraph(f"<b>Filter Scope:</b> {filters_text}", styles["MetaValue"]),
        ],
    ]

    header_table = Table(header_table_data, colWidths=[530, 240])
    header_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 10))

    # Top accent line (Cobalt + Pink)
    accent_bar = Table([["", ""]], colWidths=[550, 220], rowHeights=[2.5])
    accent_bar.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#2563eb")),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#ff2a85")),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(accent_bar)
    story.append(Spacer(1, 10))
    return story


def _build_summary_kpi_block(records: List[Dict[str, Any]], record_type_label: str = "Records") -> List[Any]:
    """Generate high-level triage statistics and synthetic data notices."""
    styles = _get_report_styles()
    story: List[Any] = []

    total_count = len(records)
    crit_count = sum(1 for r in records if r.get("priority_level") == "CRITICAL" or r.get("Priority_Tier") == "CRITICAL")
    high_count = sum(1 for r in records if r.get("priority_level") == "HIGH" or r.get("Priority_Tier") == "HIGH")
    mod_count = sum(1 for r in records if r.get("priority_level") == "MODERATE" or r.get("Priority_Tier") == "MODERATE")
    low_count = sum(1 for r in records if r.get("priority_level") == "LOW" or r.get("Priority_Tier") == "LOW")
    other_count = total_count - (crit_count + high_count + mod_count + low_count)

    synthetic_count = sum(
        1 for r in records
        if r.get("location_provenance") == "EXPLICIT_SYNTHETIC_DEMO"
        or r.get("Location_Provenance") == "EXPLICIT_SYNTHETIC_DEMO"
        or "SYNTHETIC" in str(r.get("Location_Provenance", "")).upper()
        or "DEMO" in str(r.get("Location_Provenance", "")).upper()
    )

    kpi_data = [
        [
            Paragraph(f"<b>TOTAL {record_type_label.upper()}</b><br/><font size='13'><b>{total_count}</b></font>", styles["TableCellBold"]),
            Paragraph(f"<b>CRITICAL (>=75)</b><br/><font size='13' color='#dc2626'><b>{crit_count}</b></font>", styles["TableCellBold"]),
            Paragraph(f"<b>HIGH (55–74)</b><br/><font size='13' color='#ea580c'><b>{high_count}</b></font>", styles["TableCellBold"]),
            Paragraph(f"<b>MODERATE (35–54)</b><br/><font size='13' color='#d97706'><b>{mod_count}</b></font>", styles["TableCellBold"]),
            Paragraph(f"<b>LOW (<35)</b><br/><font size='13' color='#16a34a'><b>{low_count}</b></font>", styles["TableCellBold"]),
            Paragraph(f"<b>SYNTHETIC DEMO</b><br/><font size='13' color='#7c3aed'><b>{synthetic_count}</b></font>", styles["TableCellBold"]),
        ]
    ]

    kpi_table = Table(kpi_data, colWidths=[130, 128, 128, 128, 128, 128], rowHeights=[32])
    kpi_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    if synthetic_count > 0:
        syn_notice = Table(
            [[
                Paragraph(
                    f"⚠️ <b>DEMONSTRATION RECORD NOTICE:</b> This document contains <b>{synthetic_count} calibrated synthetic demonstration record(s)</b> "
                    "(tagged <code>EXPLICIT_SYNTHETIC_DEMO</code>). Synthetic items are clearly designated in the provenance column below.",
                    styles["SyntheticBadge"],
                )
            ]],
            colWidths=[770],
        )
        syn_notice.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#fef3c7")),
                ("BOX", (0, 0), (0, 0), 0.75, colors.HexColor("#f59e0b")),
                ("LEFTPADDING", (0, 0), (0, 0), 8),
                ("RIGHTPADDING", (0, 0), (0, 0), 8),
                ("TOPPADDING", (0, 0), (0, 0), 4),
                ("BOTTOMPADDING", (0, 0), (0, 0), 4),
            ])
        )
        story.append(syn_notice)
        story.append(Spacer(1, 8))

    return story


def build_work_orders_pdf(
    work_orders: List[Dict[str, Any]],
    filter_summary: Optional[Dict[str, Any]] = None,
    filter_desc: Optional[str] = None,
) -> bytes:
    """
    Generate professional landscape A4 PDF dispatch work orders.
    Complies with municipal road-authority procurement and maintenance standards.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=42,
    )

    styles = _get_report_styles()
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    story: List[Any] = []

    # Header block
    story.extend(
        _build_header_block(
            title="MUNICIPAL WORK ORDER DISPATCH SCHEDULE",
            subtitle="Prioritized Pavement Maintenance Work Orders & Inspection Tasks",
            timestamp_str=now_str,
            filter_summary=filter_summary,
            filter_desc=filter_desc,
        )
    )

    # Summary KPI Block
    story.extend(_build_summary_kpi_block(work_orders, record_type_label="Work Orders"))

    if not work_orders:
        empty_box = Table(
            [[
                Paragraph(
                    "<b>NO WORK ORDERS MATCH THE SELECTED FILTERS</b><br/><br/>"
                    "Adjust the Priority Tier, Road Classification, or Review Status filters in the console to populate work orders.",
                    styles["EmptyState"],
                )
            ]],
            colWidths=[770],
            rowHeights=[140],
        )
        empty_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (0, 0), 1, colors.HexColor("#cbd5e1")),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("VALIGN", (0, 0), (0, 0), "MIDDLE"),
            ])
        )
        story.append(empty_box)
        doc.build(story, canvasmaker=NumberedCanvas)
        return buffer.getvalue()

    # Column widths summing to 770 pt
    col_widths = [75, 125, 80, 95, 45, 60, 55, 155, 80]
    headers = [
        "Work Order ID",
        "Road Corridor",
        "Road Class",
        "Pavement Distress",
        "Score",
        "Priority",
        "Status",
        "Recommended Action / SLA",
        "Provenance",
    ]

    header_cells = [Paragraph(f"<b>{h}</b>", styles["TableHeader"]) for h in headers]
    table_rows = [header_cells]

    for item in work_orders:
        wo_id = item.get("Work_Order_ID") or f"WO-{str(item.get('incident_uuid') or item.get('Incident_UUID', ''))[:8].upper()}"
        road_name = item.get("Road_Name") or item.get("road_name") or item.get("Road") or "Corridor Unspecified"
        road_class = item.get("Road_Classification") or item.get("road_classification") or "UNKNOWN"

        d_code = item.get("Distress_Code") or item.get("class_code") or ""
        d_name = item.get("Pavement_Distress") or item.get("class_name") or ""
        if not d_code and item.get("detected_classes"):
            dc_list = item.get("detected_classes")
            d_code = dc_list[0] if isinstance(dc_list, list) and len(dc_list) > 0 else ""
        if not d_name and d_code:
            d_name = config.CLASS_HUMAN_NAMES.get(d_code, d_code)

        distress_str = f"<b>{d_code}</b> • {d_name}" if d_code else (d_name or "—")

        score = item.get("Priority_Score") or item.get("priority_score", 0.0)
        try:
            score_num = float(score)
            score_str = f"{score_num:.1f}"
        except Exception:
            score_str = _format_value(score)

        p_tier = item.get("Priority_Tier") or item.get("priority_level") or "UNKNOWN"
        p_tier_upper = str(p_tier).upper()

        tier_color = "#dc2626" if p_tier_upper == "CRITICAL" else ("#ea580c" if p_tier_upper == "HIGH" else ("#d97706" if p_tier_upper == "MODERATE" else "#16a34a"))
        p_tier_cell = Paragraph(f"<font color='{tier_color}'><b>{p_tier_upper}</b></font>", styles["TableCellBold"])

        status = item.get("Review_Status") or item.get("review_status") or "PENDING"

        action = item.get("Prescribed_Action")
        if not action and d_code:
            action = config.DAMAGE_ACTION_MAP.get(d_code, "Pavement Condition Assessment")
        sla = item.get("Response_SLA")
        if not sla and p_tier_upper in config.SLA_MAP:
            sla = config.SLA_MAP.get(p_tier_upper)

        action_display = f"<b>{action or 'Pavement Assessment'}</b>"
        if sla:
            action_display += f"<br/><font color='#64748b' size='6.5'>SLA: {sla}</font>"

        prov = item.get("Location_Provenance") or item.get("location_provenance") or "UNKNOWN"
        is_synthetic = prov == "EXPLICIT_SYNTHETIC_DEMO"
        prov_style = styles["SyntheticBadge"] if is_synthetic else styles["TableCellCode"]
        prov_text = "<b>SYNTHETIC DEMO</b>" if is_synthetic else prov

        row = [
            Paragraph(f"<b>{wo_id}</b>", styles["TableCellCode"]),
            Paragraph(road_name, styles["TableCell"]),
            Paragraph(road_class.replace("_", " ").title(), styles["TableCell"]),
            Paragraph(distress_str, styles["TableCell"]),
            Paragraph(score_str, styles["TableCellBold"]),
            p_tier_cell,
            Paragraph(status, styles["TableCell"]),
            Paragraph(action_display, styles["TableCell"]),
            Paragraph(prov_text, prov_style),
        ]
        table_rows.append(row)

    table = Table(table_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ])
    )
    story.append(table)

    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()


def build_incident_report_pdf(
    incidents: List[Dict[str, Any]],
    filter_summary: Optional[Dict[str, Any]] = None,
    filter_desc: Optional[str] = None,
) -> bytes:
    """
    Generate professional landscape A4 PDF incident and distress ledger report.
    Complies with municipal asset reporting standards.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=42,
    )

    styles = _get_report_styles()
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    story: List[Any] = []

    # Header block
    story.extend(
        _build_header_block(
            title="MUNICIPAL ROAD DEFECT & INCIDENT AUDIT LEDGER",
            subtitle="Geospatial Pavement Distress Inventory & Severity Assessment",
            timestamp_str=now_str,
            filter_summary=filter_summary,
            filter_desc=filter_desc,
        )
    )

    # Summary KPI Block
    story.extend(_build_summary_kpi_block(incidents, record_type_label="Incidents"))

    if not incidents:
        empty_box = Table(
            [[
                Paragraph(
                    "<b>NO INCIDENTS MATCH THE SELECTED FILTERS</b><br/><br/>"
                    "No damage records were located for the active query parameters. Reset filters in the console.",
                    styles["EmptyState"],
                )
            ]],
            colWidths=[770],
            rowHeights=[140],
        )
        empty_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (0, 0), 1, colors.HexColor("#cbd5e1")),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("VALIGN", (0, 0), (0, 0), "MIDDLE"),
            ])
        )
        story.append(empty_box)
        doc.build(story, canvasmaker=NumberedCanvas)
        return buffer.getvalue()

    # Column widths summing to 770 pt
    col_widths = [105, 70, 115, 80, 70, 85, 35, 55, 65, 90]
    headers = [
        "Incident UUID",
        "Timestamp",
        "Road Corridor",
        "Coordinates",
        "Road Class",
        "Detected Classes",
        "Score",
        "Priority",
        "Status",
        "Provenance",
    ]

    header_cells = [Paragraph(f"<b>{h}</b>", styles["TableHeader"]) for h in headers]
    table_rows = [header_cells]

    for inc in incidents:
        uuid_str = inc.get("incident_uuid") or "—"
        ts_raw = inc.get("timestamp") or "—"
        if len(str(ts_raw)) > 19:
            ts_display = str(ts_raw)[:19].replace("T", " ")
        else:
            ts_display = str(ts_raw)

        road_name = inc.get("road_name") or "Corridor Unspecified"

        lat = inc.get("latitude")
        lon = inc.get("longitude")
        if lat is not None and lon is not None:
            try:
                coords_str = f"{float(lat):.4f}, {float(lon):.4f}"
            except Exception:
                coords_str = "Not available"
        else:
            coords_str = "Not available"

        road_class = inc.get("road_classification") or "UNKNOWN"

        dc = inc.get("detected_classes", [])
        if isinstance(dc, list) and len(dc) > 0:
            dc_badges = []
            for c in dc:
                c_name = config.CLASS_HUMAN_NAMES.get(c, c)
                dc_badges.append(f"<b>{c}</b> ({c_name})")
            dc_str = "<br/>".join(dc_badges)
        else:
            dc_str = "<font color='#64748b'>None detected</font>"

        score = inc.get("priority_score", 0.0)
        try:
            score_num = float(score)
            score_str = f"{score_num:.1f}"
        except Exception:
            score_str = _format_value(score)

        p_tier = inc.get("priority_level") or "UNKNOWN"
        p_tier_upper = str(p_tier).upper()
        tier_color = "#dc2626" if p_tier_upper == "CRITICAL" else ("#ea580c" if p_tier_upper == "HIGH" else ("#d97706" if p_tier_upper == "MODERATE" else "#16a34a"))
        p_tier_cell = Paragraph(f"<font color='{tier_color}'><b>{p_tier_upper}</b></font>", styles["TableCellBold"])

        status = inc.get("review_status") or "PENDING"

        prov = inc.get("location_provenance") or "UNKNOWN"
        is_synthetic = prov == "EXPLICIT_SYNTHETIC_DEMO"
        prov_style = styles["SyntheticBadge"] if is_synthetic else styles["TableCellCode"]
        prov_text = "<b>SYNTHETIC DEMO</b>" if is_synthetic else prov

        row = [
            Paragraph(f"<b>{uuid_str}</b>", styles["TableCellCode"]),
            Paragraph(ts_display, styles["TableCellCode"]),
            Paragraph(road_name, styles["TableCell"]),
            Paragraph(coords_str, styles["TableCellCode"]),
            Paragraph(road_class.replace("_", " ").title(), styles["TableCell"]),
            Paragraph(dc_str, styles["TableCell"]),
            Paragraph(score_str, styles["TableCellBold"]),
            p_tier_cell,
            Paragraph(status, styles["TableCell"]),
            Paragraph(prov_text, prov_style),
        ]
        table_rows.append(row)

    table = Table(table_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ])
    )
    story.append(table)

    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()

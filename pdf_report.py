
import os
import pandas as pd
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    LongTable
)


REPORT_DIR = "reports"


def load_report(filename):
    path = os.path.join(REPORT_DIR, filename)

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Could not read {filename}: {e}")

    return pd.DataFrame()


def add_section(elements, title, df, styles, max_rows=100):
    elements.append(Paragraph(title, styles["Heading2"]))
    elements.append(Spacer(1, 6))

    if df.empty:
        elements.append(
            Paragraph("No data available for this module.", styles["Normal"])
        )
        elements.append(Spacer(1, 12))
        return

    display_df = df.head(max_rows).fillna("-").astype(str)

    headers = [
        Paragraph(str(col).replace("_", " "), styles["TableHeader"])
        for col in display_df.columns
    ]

    table_data = [headers]

    for row in display_df.values.tolist():
        table_data.append([
            Paragraph(str(value)[:120], styles["TableCell"])
            for value in row
        ])

    available_width = landscape(A4)[0] - 30 * mm
    column_width = available_width / len(display_df.columns)

    table = LongTable(
        table_data,
        colWidths=[column_width] * len(display_df.columns),
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B66")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#EFF4F8")
        ])
    ]))

    elements.append(table)

    if len(df) > max_rows:
        elements.append(Spacer(1, 5))
        elements.append(
            Paragraph(
                f"Showing first {max_rows} of {len(df)} records.",
                styles["Italic"]
            )
        )

    elements.append(Spacer(1, 16))


def add_model_performance(elements, styles):

    elements.append(PageBreak())

    elements.append(Paragraph(
        "10. Model Performance Evaluation",
        styles["Heading1"]
    ))

    elements.append(Spacer(1, 10))

    model_files = {
        "Traffic Light Detection": (
            "runs/detect/runs/detect/traffic_light_detector/results.csv"
        ),
        "Number Plate Detection": (
            "runs/detect/runs/indian_plate_detector/results.csv"
        )
    }

    performance_data = [
        [
            "Model",
            "Epochs",
            "Precision",
            "Recall",
            "mAP50",
            "mAP50-95"
        ]
    ]

    for model_name, file_path in model_files.items():

        if os.path.exists(file_path):

            try:
                results = pd.read_csv(file_path)

                if not results.empty:

                    last = results.iloc[-1]

                    performance_data.append([
                        model_name,
                        str(int(last["epoch"])),
                        f'{last["metrics/precision(B)"] * 100:.2f}%',
                        f'{last["metrics/recall(B)"] * 100:.2f}%',
                        f'{last["metrics/mAP50(B)"] * 100:.2f}%',
                        f'{last["metrics/mAP50-95(B)"] * 100:.2f}%'
                    ])

            except Exception as e:
                performance_data.append([
                    model_name, "-", "-", "-", "-", "-"
                ])
                print(f"Could not load {model_name} metrics: {e}")

        else:
            performance_data.append([
                model_name, "-", "-", "-", "-", "-"
            ])

    performance_table = Table(
        performance_data,
        colWidths=[65*mm, 20*mm, 25*mm, 25*mm, 25*mm, 30*mm]
    )

    performance_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B66")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#EFF4F8")
        ])
    ]))

    elements.append(performance_table)
    elements.append(Spacer(1, 15))

    elements.append(Paragraph(
        "Performance metrics are taken from the final recorded training "
        "epoch and evaluated on the respective validation datasets. "
        "These metrics do not guarantee real-world performance.",
        styles["Normal"]
    ))


def generate_traffic_pdf(report_data, output_path):

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitleCustom",
        parent=styles["Title"],
        textColor=colors.HexColor("#123B66"),
        fontSize=19,
        alignment=1,
        spaceAfter=12
    ))

    styles.add(ParagraphStyle(
        name="TableHeader",
        parent=styles["Normal"],
        textColor=colors.white,
        fontSize=7,
        leading=9
    ))

    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["Normal"],
        fontSize=7,
        leading=9,
        wordWrap="CJK"
    ))

    elements = []

    # TITLE
    elements.append(Paragraph(
        "INTELLIGENT TRAFFIC & ROAD SAFETY",
        styles["ReportTitleCustom"]
    ))

    elements.append(Paragraph(
        "Complete Traffic Analytics Report",
        styles["Heading2"]
    ))

    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 20))

    # PROJECT SUMMARY
    elements.append(Paragraph(
        "1. Project Summary",
        styles["Heading1"]
    ))

    summary_data = [
        ["Metric", "Value"],
        ["Total Vehicles", str(report_data.get("total_vehicles", 0))],
        ["Average Vehicles Per Frame", str(report_data.get("average_vehicles", 0))],
        ["Frames Analysed", str(report_data.get("frames_analysed", 0))],
        ["Potential Red-Light Violations", str(report_data.get("violations", 0))]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[100 * mm, 70 * mm]
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B66")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # LOAD CSV REPORTS
    detection_df = load_report("traffic_detection_report.csv")
    density_df = load_report("traffic_density_report.csv")
    tracking_df = load_report("vehicle_tracking_report.csv")
    helmet_df = load_report("helmet_detection_report.csv")
    plate_df = load_report("number_plate_report.csv")
    violation_df = load_report("traffic_violations.csv")
    light_df = load_report("traffic_light_report.csv")
    red_light_df = load_report("red_light_violations.csv")

    if red_light_df.empty:
        red_light_df = report_data.get(
            "violations_df", pd.DataFrame()
        )

    # 8 MODULE REPORTS
    add_section(
        elements,
        "2. Vehicle Detection Analysis",
        detection_df,
        styles
    )

    add_section(
        elements,
        "3. Traffic Density Analysis",
        density_df,
        styles
    )

    add_section(
        elements,
        "4. Vehicle Tracking Analysis",
        tracking_df,
        styles
    )

    add_section(
        elements,
        "5. Helmet Detection Results",
        helmet_df,
        styles
    )

    add_section(
        elements,
        "6. Number Plate Recognition Results",
        plate_df,
        styles
    )

    add_section(
        elements,
        "7. Stop-Line Violation Results",
        violation_df,
        styles
    )

    add_section(
        elements,
        "8. Traffic Light Detection Results",
        light_df,
        styles
    )

    add_section(
        elements,
        "9. Red-Light Violation Records",
        red_light_df,
        styles
    )

    # MODEL PERFORMANCE
    add_model_performance(elements, styles)

    # REPORT NOTES
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(
        "11. Report Notes",
        styles["Heading2"]
    ))

    elements.append(Paragraph(
        "This report contains automated traffic detection and analytics "
        "results generated by the project modules. Detection outputs may "
        "contain false positives or false negatives. Potential violations "
        "and number plate readings must be manually verified before any "
        "enforcement or legal action.",
        styles["Normal"]
    ))

    doc.build(elements)

    return output_path

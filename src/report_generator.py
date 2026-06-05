"""
Automated Business Report Generator
====================================
Generates a professional PDF report with KPIs, charts, findings,
and actionable recommendations using ReportLab.
"""

import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.piecharts import Pie

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Styles ─────────────────────────────────────────────────────────────────────

def get_custom_styles():
    """Create custom ReportLab styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontSize=28,
        leading=34,
        textColor=colors.HexColor("#1A1A2E"),
        alignment=TA_CENTER,
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#6C63FF"),
        spaceBefore=20,
        spaceAfter=10,
        borderWidth=0,
        borderColor=colors.HexColor("#6C63FF"),
        borderPadding=5,
    ))

    styles.add(ParagraphStyle(
        name="SubSection",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#333333"),
        spaceBefore=14,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="BodyText2",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#333333"),
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="KPIValue",
        parent=styles["Normal"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#6C63FF"),
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="KPILabel",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#777777"),
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="RecommendationItem",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#333333"),
        leftIndent=20,
        spaceBefore=4,
        spaceAfter=4,
    ))

    return styles


# ── Report Builder ─────────────────────────────────────────────────────────────

def build_cover_page(styles) -> list:
    """Build the cover page."""
    elements = []
    elements.append(Spacer(1, 2 * inch))
    elements.append(Paragraph("Customer Churn Intelligence", styles["CoverTitle"]))
    elements.append(Paragraph("Business Analytics Report", styles["CoverSubtitle"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(HRFlowable(
        width="60%", thickness=2, color=colors.HexColor("#6C63FF"),
        spaceBefore=10, spaceAfter=10,
    ))
    elements.append(Spacer(1, 0.3 * inch))
    date_str = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Generated on {date_str}", styles["CoverSubtitle"]))
    elements.append(Paragraph(
        "Confidential — For Internal Use Only", styles["CoverSubtitle"]
    ))
    elements.append(PageBreak())
    return elements


def build_executive_summary(styles, df: pd.DataFrame) -> list:
    """Build the executive summary section with KPIs."""
    elements = []
    elements.append(Paragraph("1. Executive Summary", styles["SectionTitle"]))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#E0E0E0"),
        spaceAfter=10,
    ))

    total_customers = len(df)
    churn_rate = (df["Churn"] == "Yes").mean()
    revenue_at_risk = df[df["Churn"] == "Yes"]["TotalCharges"].sum()
    avg_tenure = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()

    elements.append(Paragraph(
        f"This report analyzes <b>{total_customers:,}</b> customers to identify churn patterns, "
        f"risk factors, and actionable retention strategies. The current churn rate stands at "
        f"<b>{churn_rate:.1%}</b>, with an estimated <b>${revenue_at_risk:,.0f}</b> in revenue at risk.",
        styles["BodyText2"],
    ))

    elements.append(Spacer(1, 0.2 * inch))

    # KPI table
    kpi_data = [
        ["Total Customers", "Churn Rate", "Revenue at Risk", "Avg Tenure"],
        [f"{total_customers:,}", f"{churn_rate:.1%}", f"${revenue_at_risk:,.0f}",
         f"{avg_tenure:.1f} months"],
    ]
    kpi_table = Table(kpi_data, colWidths=[1.5 * inch] * 4)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C63FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("TOPPADDING", (0, 1), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F5FF")),
    ]))
    elements.append(kpi_table)

    elements.append(Spacer(1, 0.3 * inch))
    return elements


def build_eda_section(styles) -> list:
    """Build the EDA findings section with embedded charts."""
    elements = []
    elements.append(Paragraph("2. Exploratory Data Analysis", styles["SectionTitle"]))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#E0E0E0"),
        spaceAfter=10,
    ))

    charts = [
        ("01_churn_distribution.png", "Churn Distribution",
         "The overall churn rate shows a significant portion of customers leaving the service."),
        ("02_revenue_analysis.png", "Revenue Analysis",
         "Revenue loss from churned customers represents a substantial business impact."),
        ("04_contract_churn.png", "Contract Type Analysis",
         "Month-to-month contracts show significantly higher churn rates compared to long-term contracts."),
        ("05_tenure_analysis.png", "Tenure Analysis",
         "Customers with shorter tenure are more likely to churn, indicating early engagement is critical."),
        ("06_payment_method_churn.png", "Payment Method Impact",
         "Electronic check users exhibit the highest churn rates, suggesting friction in the payment process."),
    ]

    reports_dir = os.path.abspath(REPORTS_DIR)
    for filename, title, description in charts:
        filepath = os.path.join(reports_dir, filename)
        if os.path.exists(filepath):
            elements.append(Paragraph(title, styles["SubSection"]))
            elements.append(Paragraph(description, styles["BodyText2"]))
            try:
                img = Image(filepath, width=5.5 * inch, height=2.2 * inch)
                img.hAlign = "CENTER"
                elements.append(img)
            except Exception:
                elements.append(Paragraph(f"[Chart: {filename}]", styles["BodyText2"]))
            elements.append(Spacer(1, 0.2 * inch))

    return elements


def build_segmentation_section(styles) -> list:
    """Build the customer segmentation section."""
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("3. Customer Segmentation", styles["SectionTitle"]))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#E0E0E0"),
        spaceAfter=10,
    ))

    elements.append(Paragraph(
        "Customers were segmented using K-Means clustering based on tenure, monthly charges, "
        "total charges, and number of services. The analysis identified four distinct customer segments.",
        styles["BodyText2"],
    ))

    # Load segment profiles
    profiles_path = os.path.join(os.path.abspath(REPORTS_DIR), "segment_profiles.json")
    if os.path.exists(profiles_path):
        with open(profiles_path) as f:
            profiles = json.load(f)

        header = ["Segment", "Count", "Avg Tenure", "Avg Monthly", "Churn Rate"]
        data = [header]
        for p in profiles:
            data.append([
                p.get("Segment", ""),
                str(p.get("Count", "")),
                f"{p.get('Avg Tenure', 0):.1f}",
                f"${p.get('Avg Monthly', 0):.2f}",
                f"{p.get('Churn Rate', 0):.1%}",
            ])

        table = Table(data, colWidths=[1.6 * inch, 0.8 * inch, 1 * inch, 1.1 * inch, 1 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C63FF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5FF")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)

    # Segment PCA plot
    seg_plot = os.path.join(os.path.abspath(REPORTS_DIR), "10_customer_segments.png")
    if os.path.exists(seg_plot):
        elements.append(Spacer(1, 0.2 * inch))
        try:
            img = Image(seg_plot, width=4.5 * inch, height=3.2 * inch)
            img.hAlign = "CENTER"
            elements.append(img)
        except Exception:
            pass

    return elements


def build_model_section(styles) -> list:
    """Build the model performance section."""
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("4. Prediction Model Performance", styles["SectionTitle"]))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#E0E0E0"),
        spaceAfter=10,
    ))

    elements.append(Paragraph(
        "Three machine learning models were trained and evaluated: Logistic Regression, "
        "Random Forest, and XGBoost. The best model was selected based on F1 Score.",
        styles["BodyText2"],
    ))

    # Model comparison table
    metrics_path = os.path.join(os.path.abspath(MODELS_DIR), "model_comparison.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        header = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        data = [header]
        best_f1 = max(m["f1_score"] for m in metrics)
        for m in metrics:
            row = [
                m["model"],
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['roc_auc']:.4f}",
            ]
            data.append(row)

        table = Table(data, colWidths=[1.3 * inch, 0.9 * inch, 0.9 * inch,
                                       0.8 * inch, 0.9 * inch, 0.9 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C63FF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5FF")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)

    # ROC curve
    roc_plot = os.path.join(os.path.abspath(REPORTS_DIR), "12_roc_curves.png")
    if os.path.exists(roc_plot):
        elements.append(Spacer(1, 0.2 * inch))
        try:
            img = Image(roc_plot, width=4 * inch, height=4 * inch)
            img.hAlign = "CENTER"
            elements.append(img)
        except Exception:
            pass

    return elements


def build_insights_section(styles) -> list:
    """Build the key insights section."""
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("5. Key Insights & Churn Drivers", styles["SectionTitle"]))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#E0E0E0"),
        spaceAfter=10,
    ))

    # SHAP feature importance
    fi_plot = os.path.join(os.path.abspath(REPORTS_DIR), "14_feature_importance.png")
    if os.path.exists(fi_plot):
        elements.append(Paragraph("Feature Importance (SHAP)", styles["SubSection"]))
        try:
            img = Image(fi_plot, width=5 * inch, height=3.5 * inch)
            img.hAlign = "CENTER"
            elements.append(img)
        except Exception:
            pass
        elements.append(Spacer(1, 0.2 * inch))

    # Insights
    insights_path = os.path.join(os.path.abspath(REPORTS_DIR), "insights.json")
    if os.path.exists(insights_path):
        with open(insights_path) as f:
            insights = json.load(f)

        elements.append(Paragraph("Business Insights", styles["SubSection"]))
        for i, ins in enumerate(insights, 1):
            priority_icon = {"high": "🔴", "medium": "🟡", "info": "🔵"}.get(
                ins.get("priority", ""), "")
            text = f"<b>{i}. [{ins['category']}]</b> {ins['insight']}"
            elements.append(Paragraph(text, styles["RecommendationItem"]))

    return elements


def build_recommendations_section(styles, df: pd.DataFrame) -> list:
    """Build the recommendations section."""
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("6. Recommendations", styles["SectionTitle"]))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#E0E0E0"),
        spaceAfter=10,
    ))

    recommendations = [
        {
            "title": "Introduce Loyalty Discounts",
            "detail": "Offer loyalty discounts for customers with tenure under 12 months "
                      "and monthly charges above $70. This targets the highest-risk segment.",
            "priority": "High",
            "impact": "Could reduce churn by 15-20% in the at-risk segment.",
        },
        {
            "title": "Promote Long-Term Contracts",
            "detail": "Incentivize month-to-month customers to switch to annual or biennial "
                      "contracts with a 10-15% discount. Month-to-month contracts are the "
                      "strongest predictor of churn.",
            "priority": "High",
            "impact": "Potential to stabilize 40%+ of at-risk customers.",
        },
        {
            "title": "Improve Fiber Optic Experience",
            "detail": "Fiber optic customers show higher churn rates despite paying more. "
                      "Investigate service quality and consider value-add bundles.",
            "priority": "Medium",
            "impact": "Address quality-price perception gap for premium customers.",
        },
        {
            "title": "Migrate from Electronic Check",
            "detail": "Customers paying by electronic check churn significantly more. "
                      "Offer auto-payment setup incentives (e.g., $5/month credit).",
            "priority": "Medium",
            "impact": "Reduce friction-based churn in the payment process.",
        },
        {
            "title": "Bundle Security & Support Services",
            "detail": "Customers without Online Security and Tech Support churn at higher rates. "
                      "Create discounted bundles to improve engagement and reduce churn.",
            "priority": "Medium",
            "impact": "Increase service stickiness and customer lifetime value.",
        },
        {
            "title": "Senior Citizen Retention Program",
            "detail": "Launch a dedicated senior-friendly support program with simplified "
                      "billing and priority assistance.",
            "priority": "Low",
            "impact": "Improve retention in the senior citizen demographic.",
        },
    ]

    for rec in recommendations:
        priority_color = {
            "High": "#FF6B6B", "Medium": "#FFD93D", "Low": "#00D4AA"
        }.get(rec["priority"], "#888")

        elements.append(Paragraph(
            f'<b>{rec["title"]}</b> '
            f'<font color="{priority_color}">[{rec["priority"]} Priority]</font>',
            styles["SubSection"],
        ))
        elements.append(Paragraph(rec["detail"], styles["BodyText2"]))
        elements.append(Paragraph(
            f'<i>Expected Impact: {rec["impact"]}</i>', styles["BodyText2"]
        ))
        elements.append(Spacer(1, 0.1 * inch))

    return elements


def generate_report(output_path: str = None) -> str:
    """Generate the full PDF report."""
    print("=" * 60)
    print("  BUSINESS REPORT GENERATOR")
    print("=" * 60)

    if output_path is None:
        output_path = os.path.join(os.path.abspath(REPORTS_DIR), "churn_report.pdf")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    data_path = os.path.join(os.path.abspath(DATA_DIR), "cleaned_churn.csv")
    if not os.path.exists(data_path):
        print(f"❌ Cleaned data not found at {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path)

    styles = get_custom_styles()

    # Build document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    elements = []
    elements.extend(build_cover_page(styles))
    print("  ✅ Cover page")

    elements.extend(build_executive_summary(styles, df))
    print("  ✅ Executive summary")

    elements.extend(build_eda_section(styles))
    print("  ✅ EDA findings")

    elements.extend(build_segmentation_section(styles))
    print("  ✅ Segmentation results")

    elements.extend(build_model_section(styles))
    print("  ✅ Model performance")

    elements.extend(build_insights_section(styles))
    print("  ✅ Key insights")

    elements.extend(build_recommendations_section(styles, df))
    print("  ✅ Recommendations")

    # Build PDF
    doc.build(elements)
    print(f"\n📄 Report saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    generate_report()

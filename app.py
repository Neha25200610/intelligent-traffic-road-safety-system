
import streamlit as st
import pandas as pd
import os
from datetime import datetime

from database import get_red_light_violations
from pdf_report import generate_traffic_pdf
from integration.vehicle_detection import run_vehicle_detection


# =====================================================
#                 PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Intelligent Traffic & Road Safety Detection",
    page_icon="🚦",
    layout="wide"
)


# =====================================================
#                  PROJECT DIRECTORIES
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
VIDEO_DIR = os.path.join(BASE_DIR, "outputs", "videos")
RUNS_DIR = os.path.join(BASE_DIR, "runs")


# =====================================================
#                    REPORT PATHS
# =====================================================

detection_path = os.path.join(REPORT_DIR, "traffic_detection_report.csv")
tracking_path = os.path.join(REPORT_DIR, "vehicle_tracking_report.csv")
density_path = os.path.join(REPORT_DIR, "traffic_density_report.csv")
helmet_report_path = os.path.join(REPORT_DIR, "helmet_detection_report.csv")
number_plate_report_path = os.path.join(REPORT_DIR, "number_plate_report.csv")
violation_report_path = os.path.join(REPORT_DIR, "traffic_violations.csv")
traffic_light_report_path = os.path.join(REPORT_DIR, "traffic_light_report.csv")


# =====================================================
#                     VIDEO PATHS
# =====================================================

def get_video_path(*filenames):
    """Return the first existing video from the supplied filenames."""
    for filename in filenames:
        path = os.path.join(VIDEO_DIR, filename)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    return None


# Vehicle detection: prefer H.264 if an encoded copy exists.
detection_video = get_video_path(
    "traffic_detected_new_h264.mp4",
    "traffic_detected_new.mp4"
)

tracking_video = get_video_path(
    "vehicle_tracking_h264.mp4",
    "vehicle_tracking.mp4"
)

helmet_video = get_video_path(
    "helmet_detected_h264.mp4",
    "helmet_detected.mp4"
)

number_plate_video = get_video_path(
    "number_plate_detected_h264.mp4",
    "number_plate_detected.mp4"
)

violation_video = get_video_path(
    "traffic_violations_h264.mp4",
    "traffic_violations.mp4"
)

traffic_light_video = get_video_path(
    "traffic_light_detected_h264.mp4",
    "traffic_light_detected.mp4"
)

red_light_video = get_video_path(
    "red_light_violations_h264.mp4",
    "red_light_violations.mp4"
)


# =====================================================
#                    VIDEO DISPLAY HELPER
# =====================================================

def display_video(video_path, heading, key):
    if video_path and os.path.isfile(video_path):
        st.subheader(heading)

        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

        st.caption(
            f"Video: {os.path.basename(video_path)} | "
            f"Size: {file_size_mb:.2f} MB"
        )

        try:
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()

            if not video_bytes:
                st.error("The video file is empty.")
                return

            st.video(
                video_bytes,
                format="video/mp4"
            )

        except Exception as e:
            st.error(f"Unable to load video: {e}")

    else:
        st.warning(
            f"{heading} is unavailable. "
            "Check that the processed video exists in outputs/videos."
        )


# =====================================================
#                    LOAD REPORTS
# =====================================================

def load_report(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Error loading {os.path.basename(path)}: {e}")

    return None


detection_df = load_report(detection_path)
tracking_df = load_report(tracking_path)
density_df = load_report(density_path)
helmet_df = load_report(helmet_report_path)
number_plate_df = load_report(number_plate_report_path)
violation_df = load_report(violation_report_path)
traffic_light_df = load_report(traffic_light_report_path)


# =====================================================
#             LOAD RED-LIGHT DATA FROM POSTGRESQL
# =====================================================

def load_red_light_data():
    try:
        df = get_red_light_violations()

        df = df.rename(columns={
            "vehicle_id": "Vehicle ID",
            "vehicle_type": "Vehicle Type",
            "frame_number": "Frame Number",
            "timestamp_seconds": "Timestamp (seconds)",
            "signal_state": "Signal State",
            "event": "Event",
            "created_at": "Created At"
        })

        return df, True, None

    except Exception as e:
        return None, False, str(e)


red_light_df, database_connected, database_error = load_red_light_data()


# =====================================================
#                       TITLE
# =====================================================

st.title("🚦 Intelligent Traffic & Road Safety Detection System")

st.markdown(
    "AI-powered vehicle detection, tracking, traffic analysis, "
    "helmet detection, number plate recognition, stop-line monitoring, "
    "traffic signal detection and red-light violation detection."
)

st.divider()


# =====================================================
#                   SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🚦 Traffic System")

menu = st.sidebar.selectbox(
    "Choose an option",
    [
        "Dashboard",
        "Vehicle Detection",
        "Traffic Analysis",
        "Traffic Violation Detection",
        "Helmet Detection",
        "Number Plate Recognition",
        "Traffic Light Detection",
        "Red-Light Violation Detection",
        "Model Performance",
        "Database Management"
    ]
)


# =====================================================
#                       DASHBOARD
# =====================================================

if menu == "Dashboard":

    st.header("📊 Traffic Monitoring Dashboard")

    total_tracked = 0

    if tracking_df is not None and "Unique Tracked Vehicles" in tracking_df.columns:
        total_tracked = int(tracking_df["Unique Tracked Vehicles"].fillna(0).sum())

    average_vehicles = 0

    if density_df is not None and "Vehicle Count" in density_df.columns and not density_df.empty:
        average_vehicles = round(density_df["Vehicle Count"].mean(), 2)

    helmet_total = len(helmet_df) if helmet_df is not None else 0
    number_plate_total = len(number_plate_df) if number_plate_df is not None else 0
    violation_total = len(violation_df) if violation_df is not None else 0
    red_light_total = len(red_light_df) if red_light_df is not None else 0

    traffic_light_total = 0

    if (
        traffic_light_df is not None
        and "Frame-level Detections" in traffic_light_df.columns
        and "Traffic Light State" in traffic_light_df.columns
        and not traffic_light_df.empty
    ):
        total_row = traffic_light_df[
            traffic_light_df["Traffic Light State"] == "Total"
        ]

        if not total_row.empty:
            traffic_light_total = int(total_row["Frame-level Detections"].iloc[0])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Unique Tracked Vehicles", total_tracked)
    col2.metric("Average Vehicles per Frame", average_vehicles)
    col3.metric("Helmet Detections", helmet_total)
    col4.metric("Number Plate Detections", number_plate_total)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Stop-Line Crossing Events", violation_total)
    col2.metric("Traffic Light Detections", traffic_light_total)
    col3.metric("Potential Red-Light Violations", red_light_total)

    st.divider()

    st.subheader("🚘 Vehicle Tracking Summary")

    if (
        tracking_df is not None
        and "Vehicle Type" in tracking_df.columns
        and "Unique Tracked Vehicles" in tracking_df.columns
    ):
        st.dataframe(tracking_df, use_container_width=True, hide_index=True)

        st.subheader("Vehicle Classification")

        chart_data = tracking_df.set_index("Vehicle Type")[["Unique Tracked Vehicles"]]
        st.bar_chart(chart_data)
    else:
        st.warning("Vehicle tracking report is missing or has incorrect columns.")

    st.divider()

    st.subheader("🚦 Traffic Density Summary")

    if density_df is not None and "Traffic Density" in density_df.columns:
        density_counts = (
            density_df["Traffic Density"]
            .value_counts()
            .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        )

        st.bar_chart(pd.DataFrame({"Number of Frames": density_counts}))

        total_frames = len(density_df)
        st.write(f"**Total Frames Analysed:** {total_frames}")

        if total_frames > 0:
            percentages = (density_counts / total_frames * 100).round(2)

            summary_df = pd.DataFrame({
                "Traffic Density": density_counts.index,
                "Number of Frames": density_counts.values,
                "Percentage": percentages.values
            })

            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Traffic density report not found.")

    st.divider()

    st.subheader("🔎 Number Plate Recognition Summary")

    if number_plate_df is not None:
        st.write(f"**Total plate detections:** {number_plate_total}")

        if "Detected Plate Text" in number_plate_df.columns:
            readable_df = number_plate_df[
                number_plate_df["Detected Plate Text"].notna()
                & (number_plate_df["Detected Plate Text"] != "Text not readable")
            ]

            st.write(f"**Readable plate results:** {len(readable_df)}")

            if not readable_df.empty:
                st.dataframe(readable_df, use_container_width=True, hide_index=True)
            else:
                st.info("Plate regions detected, but OCR text was not readable.")
    else:
        st.info("Number plate report has not been generated yet.")

    st.divider()

    st.subheader("🚦 Stop-Line Monitoring Summary")
    st.metric("Recorded Crossing Events", violation_total)

    if violation_total == 0:
        st.info(
            "No stop-line crossing events were recorded in the current video. "
            "This does not establish whether a traffic violation occurred."
        )

    st.divider()

    st.subheader("🚦 Traffic Signal Summary")

    if traffic_light_df is not None:
        if "Traffic Light State" in traffic_light_df.columns:
            signal_df = traffic_light_df[
                traffic_light_df["Traffic Light State"] != "Total"
            ]

            if not signal_df.empty:
                st.dataframe(signal_df, use_container_width=True, hide_index=True)

                if "Frame-level Detections" in signal_df.columns:
                    st.bar_chart(
                        signal_df.set_index("Traffic Light State")[["Frame-level Detections"]]
                    )

            st.caption(
                "Counts are frame-level detections, not unique signals "
                "or confirmed signal changes."
            )
    else:
        st.info("Traffic-light report has not been generated yet.")

    st.divider()

    st.subheader("🚨 Red-Light Violation Summary")
    st.metric("Potential Red-Light Violations", red_light_total)

    if red_light_total > 0:
        st.dataframe(red_light_df, use_container_width=True, hide_index=True)
    else:
        st.info("No potential red-light violations recorded.")

    st.caption(
        "These are automated candidate events and require manual verification."
    )


# =====================================================
#                   VEHICLE DETECTION
# =====================================================

elif menu == "Vehicle Detection":

    st.header("🚗 Vehicle Detection")

    st.write(
        "YOLO-based vehicle detection identifies vehicles in the input video "
        "and displays the annotated output."
    )

    st.divider()
    st.subheader("Run New Vehicle Detection")

    input_dir = os.path.join(BASE_DIR, "data", "videos")

    available_videos = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    ) if os.path.isdir(input_dir) else []

    if available_videos:
        selected_video = st.selectbox(
            "Select an input traffic video",
            available_videos,
            key="phase1_vehicle_input"
        )

        if st.button("Run Vehicle Detection", type="primary"):
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def update_progress(current, total):
                percentage = current / total if total else 0
                progress_bar.progress(min(percentage, 1.0))
                progress_text.text(
                    f"Processing frame {current} of {total}"
                )

            try:
                result = run_vehicle_detection(
                    os.path.join(input_dir, selected_video),
                    os.path.join(VIDEO_DIR, "phase1_vehicle_detection.mp4"),
                    os.path.join(REPORT_DIR, "phase1_vehicle_detection.csv"),
                    progress_callback=update_progress
                )

                st.session_state["phase1_vehicle_detection_result"] = result
                progress_bar.progress(1.0)
                progress_text.success("Vehicle detection completed!")

            except Exception as e:
                st.error(f"Vehicle detection failed: {e}")

    else:
        st.warning("No input videos found in data/videos.")

    latest_result = st.session_state.get(
        "phase1_vehicle_detection_result"
    )

    if latest_result:
        st.subheader("Latest Detection Results")

        counts = latest_result["vehicle_counts"]
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Cars", counts["car"])
        col2.metric("Motorcycles", counts["motorcycle"])
        col3.metric("Buses", counts["bus"])
        col4.metric("Trucks", counts["truck"])
        col5.metric("Total Detections", sum(counts.values()))

        result_df = pd.read_csv(latest_result["report_file"])

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        with open(latest_result["report_file"], "rb") as file:
            st.download_button(
                "Download New Detection Report",
                data=file.read(),
                file_name="phase1_vehicle_detection.csv",
                mime="text/csv"
            )

        st.subheader("Newly Processed Video")

        processed_video = latest_result["output_video"]

        if os.path.isfile(processed_video) and os.path.getsize(processed_video) > 0:
            st.video(processed_video)
            st.caption("Browser-compatible H.264 video")
        else:
            st.error("Processed H.264 video was not found.")


    st.divider()

    if detection_df is not None:
        st.subheader("Vehicle Detection Report")
        st.dataframe(
            detection_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Vehicle Detection CSV",
            data=detection_df.to_csv(index=False),
            file_name="traffic_detection_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("Vehicle detection report not found.")

    st.divider()

    display_video(
        detection_video,
        "🚗 Processed Vehicle Detection Video",
        "vehicle_detection_video"
    )


# =====================================================
#                 TRAFFIC ANALYTICS
# =====================================================

elif menu == "Traffic Analysis":

    st.header("📈 Traffic Analytics Dashboard")

    st.write(
        "Comprehensive analysis of vehicle movement, "
        "traffic density and potential traffic violations."
    )

    total_vehicles = 0

    if tracking_df is not None and "Unique Tracked Vehicles" in tracking_df.columns:
        total_vehicles = int(tracking_df["Unique Tracked Vehicles"].fillna(0).sum())

    average_density = 0

    if density_df is not None and "Vehicle Count" in density_df.columns and not density_df.empty:
        average_density = round(density_df["Vehicle Count"].mean(), 2)

    total_frames = len(density_df) if density_df is not None else 0
    total_red_violations = len(red_light_df) if red_light_df is not None else 0
    total_stopline_events = len(violation_df) if violation_df is not None else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Unique Tracked Vehicles", total_vehicles)
    col2.metric("Average Vehicles per Frame", average_density)
    col3.metric("Frames Analysed", total_frames)

    col4, col5 = st.columns(2)

    col4.metric("Potential Red-Light Violations", total_red_violations)
    col5.metric("Stop-Line Crossing Events", total_stopline_events)

    st.divider()

    st.subheader("🚘 Vehicle Classification Analysis")

    if (
        tracking_df is not None
        and "Vehicle Type" in tracking_df.columns
        and "Unique Tracked Vehicles" in tracking_df.columns
    ):
        vehicle_chart = tracking_df[
            ["Vehicle Type", "Unique Tracked Vehicles"]
        ].copy()

        vehicle_chart["Unique Tracked Vehicles"] = pd.to_numeric(
            vehicle_chart["Unique Tracked Vehicles"],
            errors="coerce"
        ).fillna(0)

        st.bar_chart(vehicle_chart.set_index("Vehicle Type"))
        st.dataframe(vehicle_chart, use_container_width=True, hide_index=True)
    else:
        st.warning("Vehicle tracking data is unavailable.")

    st.divider()

    st.subheader("🚦 Traffic Density Analysis")

    if density_df is not None and "Traffic Density" in density_df.columns and not density_df.empty:
        density_counts = (
            density_df["Traffic Density"]
            .astype(str)
            .str.upper()
            .value_counts()
            .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        )

        st.bar_chart(pd.DataFrame({"Number of Frames": density_counts}))

        density_summary = pd.DataFrame({
            "Density Level": density_counts.index,
            "Number of Frames": density_counts.values,
            "Percentage": (density_counts.values / len(density_df) * 100).round(2)
        })

        st.dataframe(density_summary, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Traffic Density Report",
            data=density_df.to_csv(index=False),
            file_name="traffic_density_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("Traffic density report is unavailable.")

    st.divider()

    st.subheader("🚨 Red-Light Violation Analytics")

    if red_light_df is not None and not red_light_df.empty:
        col1, col2 = st.columns(2)

        col1.metric("Total Potential Events", len(red_light_df))

        col2.metric(
            "Unique Vehicles",
            red_light_df["Vehicle ID"].nunique()
            if "Vehicle ID" in red_light_df.columns else 0
        )

        if "Vehicle Type" in red_light_df.columns:
            st.write("**Potential Events by Vehicle Type**")

            violation_chart = (
                red_light_df["Vehicle Type"]
                .value_counts()
                .rename_axis("Vehicle Type")
                .to_frame("Potential Events")
            )

            st.bar_chart(violation_chart)

        if "Created At" in red_light_df.columns:
            time_df = red_light_df.copy()

            time_df["Created At"] = pd.to_datetime(
                time_df["Created At"],
                errors="coerce"
            )

            time_df = time_df.dropna(subset=["Created At"])

            if not time_df.empty:
                daily_counts = time_df.set_index("Created At").resample("D").size()

                st.write("**Potential Events by Database Date**")
                st.line_chart(daily_counts.rename("Potential Events"))

        st.dataframe(red_light_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Red-Light Analytics CSV",
            data=red_light_df.to_csv(index=False),
            file_name="red_light_analytics.csv",
            mime="text/csv"
        )
    else:
        st.info("No red-light violation records are available.")

    st.caption(
        "Red-light events are automated candidate detections, "
        "not legally verified violations. Database dates represent "
        "record creation times, not necessarily the original video dates."
    )

    st.divider()

    st.subheader("🚥 Traffic Signal Analysis")

    if (
        traffic_light_df is not None
        and "Traffic Light State" in traffic_light_df.columns
        and "Frame-level Detections" in traffic_light_df.columns
    ):
        signal_df = traffic_light_df[
            traffic_light_df["Traffic Light State"] != "Total"
        ].copy()

        signal_df["Frame-level Detections"] = pd.to_numeric(
            signal_df["Frame-level Detections"],
            errors="coerce"
        ).fillna(0)

        signal_chart = signal_df.groupby("Traffic Light State")[
            "Frame-level Detections"
        ].sum()

        if not signal_chart.empty:
            st.bar_chart(signal_chart.to_frame("Detections"))
            st.dataframe(signal_df, use_container_width=True, hide_index=True)
    else:
        st.info("Traffic signal report is unavailable.")

    st.divider()

    st.subheader("📋 Vehicle Tracking Report")

    if tracking_df is not None:
        st.dataframe(tracking_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Vehicle Tracking Report",
            data=tracking_df.to_csv(index=False),
            file_name="vehicle_tracking_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("Vehicle tracking report is unavailable.")

    st.divider()

    st.subheader("📄 Download Complete Traffic Analytics Report")

    st.write(
        "Generate a complete PDF with all eight traffic modules "
        "and YOLO model performance evaluation."
    )

    pdf_path = os.path.join(REPORT_DIR, "traffic_analytics_report.pdf")

    if st.button("📄 Generate PDF Report", key="generate_traffic_pdf"):
        try:
            pdf_violations_df = pd.DataFrame()

            if red_light_df is not None:
                pdf_violations_df = red_light_df.rename(columns={
                    "Vehicle ID": "vehicle_id",
                    "Vehicle Type": "vehicle_type",
                    "Frame Number": "frame_number",
                    "Timestamp (seconds)": "timestamp_seconds",
                    "Signal State": "signal_state",
                    "Event": "event"
                })

            report_data = {
                "total_vehicles": total_vehicles,
                "average_vehicles": average_density,
                "frames_analysed": total_frames,
                "violations": total_red_violations,
                "violations_df": pdf_violations_df
            }

            os.makedirs(REPORT_DIR, exist_ok=True)
            generate_traffic_pdf(report_data, pdf_path)

            st.success("Complete PDF report generated successfully!")

        except Exception as e:
            st.error(f"PDF generation failed: {e}")

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="⬇️ Download Traffic Analytics PDF",
                data=pdf_file,
                file_name="traffic_analytics_report.pdf",
                mime="application/pdf",
                key="download_traffic_pdf"
            )


# =====================================================
#              TRAFFIC VIOLATION DETECTION
# =====================================================

elif menu == "Traffic Violation Detection":

    st.header("🚦 Traffic Violation Detection")

    st.write("Monitors tracked vehicles crossing the configured stop line.")

    if violation_df is not None:
        st.subheader("Stop-Line Crossing Summary")

        crossing_count = len(violation_df)
        st.metric("Recorded Crossing Events", crossing_count)

        if crossing_count > 0:
            st.dataframe(violation_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No crossing events were recorded in this video. "
                "A crossing event alone also does not prove a red-light violation."
            )

        st.download_button(
            "Download Traffic Violation CSV",
            data=violation_df.to_csv(index=False),
            file_name="traffic_violations.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "Traffic violation report not found. "
            "Run traffic_violation.py first."
        )

    st.divider()

    display_video(
        violation_video,
        "Traffic Stop-Line Detection Video",
        "traffic_violation_video"
    )


# =====================================================
#                    HELMET DETECTION
# =====================================================

elif menu == "Helmet Detection":

    st.header("🏍️ Motorcycle Helmet Detection")

    st.write("AI-powered detection of riders with and without helmets.")

    if helmet_df is not None and "Detection" in helmet_df.columns:
        helmet_counts = helmet_df["Detection"].value_counts()

        with_helmet = int(helmet_counts.get("With Helmet", 0))
        without_helmet = int(helmet_counts.get("Without Helmet", 0))
        total_detections = len(helmet_df)

        col1, col2, col3 = st.columns(3)

        col1.metric("With Helmet", with_helmet)
        col2.metric("Without Helmet", without_helmet)
        col3.metric("Total Detections", total_detections)

        st.divider()

        st.subheader("Helmet Detection Distribution")

        helmet_chart = pd.DataFrame({
            "Detection Category": ["With Helmet", "Without Helmet"],
            "Number of Detections": [with_helmet, without_helmet]
        })

        st.bar_chart(helmet_chart.set_index("Detection Category"))

        st.subheader("Helmet Detection Report")

        st.dataframe(helmet_df, use_container_width=True)

        st.download_button(
            "Download Helmet Detection CSV",
            data=helmet_df.to_csv(index=False),
            file_name="helmet_detection_report.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "Helmet detection report not found or Detection column is missing."
        )

    st.divider()

    display_video(
        helmet_video,
        "Processed Helmet Detection Video",
        "helmet_detection_video"
    )


# =====================================================
#                NUMBER PLATE RECOGNITION
# =====================================================

elif menu == "Number Plate Recognition":

    st.header("🔎 Automatic Number Plate Recognition (ANPR)")

    st.write(
        "YOLO detects the number plate region and EasyOCR attempts "
        "to read the registration text."
    )

    if number_plate_df is not None:
        st.subheader("Number Plate Detection Metrics")

        total_plates = len(number_plate_df)

        if "Detected Plate Text" in number_plate_df.columns:
            readable_df = number_plate_df[
                number_plate_df["Detected Plate Text"].notna()
                & (number_plate_df["Detected Plate Text"] != "Text not readable")
            ]

            readable_count = len(readable_df)
        else:
            readable_df = pd.DataFrame()
            readable_count = 0

        col1, col2 = st.columns(2)

        col1.metric("Total Plate Detections", total_plates)
        col2.metric("Readable Plate Text Results", readable_count)

        st.divider()

        st.subheader("Number Plate OCR Report")
        st.dataframe(number_plate_df, use_container_width=True)

        if readable_count > 0:
            st.subheader("Recognized Registration Text")
            st.dataframe(readable_df, use_container_width=True)
        else:
            st.info(
                "A number plate region was detected, but OCR could not "
                "read its text. Try a clearer or closer video for better results."
            )

        st.download_button(
            "Download Number Plate OCR CSV",
            data=number_plate_df.to_csv(index=False),
            file_name="number_plate_report.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "Number plate report not found. "
            "Run number_plate_detection.py first."
        )

    st.divider()

    display_video(
        number_plate_video,
        "Processed Number Plate Detection Video",
        "number_plate_video"
    )


# =====================================================
#                 TRAFFIC LIGHT DETECTION
# =====================================================

elif menu == "Traffic Light Detection":

    st.header("🚦 Traffic Light Detection")

    st.write(
        "YOLO-based traffic signal detection and classification "
        "using Red, Yellow and Green signal classes."
    )

    if traffic_light_df is not None:
        if (
            "Traffic Light State" in traffic_light_df.columns
            and "Frame-level Detections" in traffic_light_df.columns
        ):
            signal_df = traffic_light_df[
                traffic_light_df["Traffic Light State"] != "Total"
            ].copy()

            signal_df["Frame-level Detections"] = pd.to_numeric(
                signal_df["Frame-level Detections"],
                errors="coerce"
            ).fillna(0).astype(int)

            red_count = int(
                signal_df.loc[
                    signal_df["Traffic Light State"] == "Red",
                    "Frame-level Detections"
                ].sum()
            )

            yellow_count = int(
                signal_df.loc[
                    signal_df["Traffic Light State"] == "Yellow",
                    "Frame-level Detections"
                ].sum()
            )

            green_count = int(
                signal_df.loc[
                    signal_df["Traffic Light State"] == "Green",
                    "Frame-level Detections"
                ].sum()
            )

            total_count = red_count + yellow_count + green_count

            st.subheader("Traffic Signal Detection Metrics")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Red Detections", red_count)
            col2.metric("Yellow Detections", yellow_count)
            col3.metric("Green Detections", green_count)
            col4.metric("Total Detections", total_count)

            st.divider()

            st.subheader("Traffic Signal Classification")

            chart_df = pd.DataFrame({
                "Signal State": ["Red", "Yellow", "Green"],
                "Detections": [red_count, yellow_count, green_count]
            })

            st.bar_chart(chart_df.set_index("Signal State"))

            st.divider()

            st.subheader("Traffic Light Detection Report")
            st.dataframe(signal_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Traffic Light CSV",
                data=traffic_light_df.to_csv(index=False),
                file_name="traffic_light_report.csv",
                mime="text/csv"
            )

            st.info(
                "These values are frame-level detections, not unique traffic "
                "signals or confirmed signal changes. This module does not "
                "independently establish red-light violations."
            )
        else:
            st.warning("Traffic-light report has an unexpected format.")
    else:
        st.warning(
            "Traffic-light report not found. "
            "Run traffic_light_detection.py first."
        )

    st.divider()

    display_video(
        traffic_light_video,
        "Processed Traffic Signal Detection Video",
        "traffic_light_video"
    )


# =====================================================
#             RED-LIGHT VIOLATION DETECTION
# =====================================================

elif menu == "Red-Light Violation Detection":

    st.header("🚨 Red-Light Violation Detection")

    st.write(
        "Uses YOLO-based traffic signal classification and vehicle tracking "
        "to identify candidate stop-line crossings while the detected signal "
        "state is red."
    )

    if red_light_df is not None:
        st.subheader("Red-Light Violation Metrics")

        total_violations = len(red_light_df)

        col1, col2 = st.columns(2)

        col1.metric("Potential Red-Light Violations", total_violations)

        unique_vehicles = (
            red_light_df["Vehicle ID"].nunique()
            if "Vehicle ID" in red_light_df.columns else 0
        )

        col2.metric("Unique Vehicle IDs", unique_vehicles)

        st.divider()

        st.subheader("Violation Events by Vehicle Type")

        if "Vehicle Type" in red_light_df.columns and not red_light_df.empty:
            type_counts = (
                red_light_df["Vehicle Type"]
                .value_counts()
                .rename_axis("Vehicle Type")
                .to_frame("Potential Events")
            )

            st.bar_chart(type_counts)
        else:
            st.info("No vehicle-type data available.")

        st.divider()

        st.subheader("Potential Violation Records")
        st.dataframe(red_light_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Red-Light Violation CSV",
            data=red_light_df.to_csv(index=False),
            file_name="red_light_violations.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "Could not load red-light records from PostgreSQL. "
            "Check that PostgreSQL is running and database.py is configured correctly."
        )

    st.divider()

    display_video(
        red_light_video,
        "Processed Red-Light Violation Video",
        "red_light_video"
    )

    st.warning(
        "Important: These are potential automated violation events, not "
        "legally verified violations. Signal-to-lane association, vehicle "
        "direction, stop-line geometry and signal classification require "
        "further validation before real-world enforcement."
    )


# =====================================================
#                 MODEL PERFORMANCE
# =====================================================

elif menu == "Model Performance":

    st.header("📊 YOLO Model Performance Evaluation")

    st.write(
        "Evaluation of trained YOLO models using their recorded "
        "validation metrics."
    )

    model_files = {
        "Traffic Light Detection": os.path.join(
            RUNS_DIR, "detect", "runs", "detect",
            "traffic_light_detector", "results.csv"
        ),
        "Number Plate Detection": os.path.join(
            RUNS_DIR, "detect", "runs",
            "indian_plate_detector", "results.csv"
        )
    }

    performance_rows = []

    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            try:
                results = pd.read_csv(file_path)

                if not results.empty:
                    last = results.iloc[-1]

                    performance_rows.append({
                        "Model": model_name,
                        "Epochs": int(last["epoch"]),
                        "Precision (%)": round(last["metrics/precision(B)"] * 100, 2),
                        "Recall (%)": round(last["metrics/recall(B)"] * 100, 2),
                        "mAP50 (%)": round(last["metrics/mAP50(B)"] * 100, 2),
                        "mAP50-95 (%)": round(last["metrics/mAP50-95(B)"] * 100, 2)
                    })
            except Exception as e:
                st.error(f"Error loading {model_name}: {e}")
        else:
            st.warning(f"Training results not found: {file_path}")

    if performance_rows:
        performance_df = pd.DataFrame(performance_rows)

        st.subheader("Model Evaluation Summary")
        st.dataframe(performance_df, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Precision Comparison")
        st.bar_chart(performance_df.set_index("Model")[["Precision (%)"]])

        st.subheader("Recall Comparison")
        st.bar_chart(performance_df.set_index("Model")[["Recall (%)"]])

        st.subheader("mAP@50 Comparison")
        st.bar_chart(performance_df.set_index("Model")[["mAP50 (%)"]])

        st.subheader("mAP@50-95 Comparison")
        st.bar_chart(performance_df.set_index("Model")[["mAP50-95 (%)"]])

        st.subheader("Training Epochs")
        st.bar_chart(performance_df.set_index("Model")[["Epochs"]])

        st.divider()

        st.download_button(
            "⬇️ Download Model Performance CSV",
            data=performance_df.to_csv(index=False),
            file_name="model_performance_report.csv",
            mime="text/csv"
        )

        st.success("Model performance evaluation loaded successfully.")
    else:
        st.warning("No YOLO training results were found.")

    st.caption(
        "Metrics are taken from the final recorded training epoch "
        "and evaluated on the respective validation datasets. "
        "They do not guarantee real-world performance."
    )


# =====================================================
#                 DATABASE MANAGEMENT
# =====================================================

elif menu == "Database Management":

    st.header("🗄️ PostgreSQL Database Management")

    st.write(
        "Live monitoring of red-light violation records stored in PostgreSQL."
    )

    if st.button("🔄 Refresh Now", key="manual_db_refresh"):
        st.rerun()

    st.divider()

    if "db_refresh_count" not in st.session_state:
        st.session_state.db_refresh_count = 0

    @st.fragment(run_every="10s")
    def display_database_management():

        st.session_state.db_refresh_count += 1

        try:
            live_df = get_red_light_violations()

            live_df = live_df.rename(columns={
                "vehicle_id": "Vehicle ID",
                "vehicle_type": "Vehicle Type",
                "frame_number": "Frame Number",
                "timestamp_seconds": "Timestamp (seconds)",
                "signal_state": "Signal State",
                "event": "Event",
                "created_at": "Created At"
            })

            refresh_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            st.success(
                f"🟢 PostgreSQL Connected | Last refreshed: {refresh_time}"
            )

            st.info(
                f"🔄 Automatic refresh active | "
                f"Refresh count: {st.session_state.db_refresh_count}"
            )

            st.caption("The database is queried again every 10 seconds.")

            total_records = len(live_df)

            unique_vehicle_ids = (
                live_df["Vehicle ID"].nunique()
                if "Vehicle ID" in live_df.columns else 0
            )

            vehicle_types_count = (
                live_df["Vehicle Type"].nunique()
                if "Vehicle Type" in live_df.columns else 0
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Records Stored", total_records)
            col2.metric("Unique Vehicle IDs", unique_vehicle_ids)
            col3.metric("Vehicle Types", vehicle_types_count)

            st.divider()

            st.subheader("🔎 Search and Filter Records")

            search_col, type_col, signal_col = st.columns(3)

            with search_col:
                vehicle_search = st.text_input(
                    "Search Vehicle ID",
                    placeholder="Enter Vehicle ID",
                    key="vehicle_id_search"
                )

            with type_col:
                vehicle_options = ["All"]

                if "Vehicle Type" in live_df.columns:
                    vehicle_options += sorted(
                        live_df["Vehicle Type"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                selected_type = st.selectbox(
                    "Vehicle Type",
                    vehicle_options,
                    key="database_vehicle_filter"
                )

            with signal_col:
                signal_options = ["All"]

                if "Signal State" in live_df.columns:
                    signal_options += sorted(
                        live_df["Signal State"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                selected_signal = st.selectbox(
                    "Signal State",
                    signal_options,
                    key="database_signal_filter"
                )

            filtered_df = live_df.copy()

            if vehicle_search.strip() and "Vehicle ID" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["Vehicle ID"]
                    .astype(str)
                    .str.contains(
                        vehicle_search.strip(),
                        case=False,
                        na=False
                    )
                ]

            if selected_type != "All" and "Vehicle Type" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["Vehicle Type"].astype(str) == selected_type
                ]

            if selected_signal != "All" and "Signal State" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["Signal State"].astype(str) == selected_signal
                ]

            st.write(f"**Showing {len(filtered_df)} of {total_records} records**")

            st.divider()

            st.subheader("📋 PostgreSQL Violation Records")

            if filtered_df.empty:
                st.warning("No records match the selected filters.")
            else:
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    label="⬇️ Download Filtered Records",
                    data=filtered_df.to_csv(index=False),
                    file_name="filtered_red_light_violations.csv",
                    mime="text/csv",
                    key="filtered_database_download"
                )

            st.divider()

            st.subheader("📊 Database Summary")

            if "Vehicle Type" in filtered_df.columns and not filtered_df.empty:
                database_chart = (
                    filtered_df["Vehicle Type"]
                    .value_counts()
                    .rename_axis("Vehicle Type")
                    .to_frame("Number of Records")
                )

                st.bar_chart(database_chart)
            else:
                st.info("No vehicle type data available for the chart.")

            st.divider()

            st.subheader("🕒 Latest Database Entries")

            if not filtered_df.empty:
                st.dataframe(
                    filtered_df.head(5),
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Data retrieved directly from PostgreSQL: "
                "traffic_db → red_light_violations"
            )

        except Exception as e:
            st.error(f"PostgreSQL connection error: {e}")

    display_database_management()


# =====================================================
#                         FOOTER
# =====================================================

st.divider()

st.caption(
    "Intelligent Traffic & Road Safety Detection System | "
    "Powered by Python, YOLO, OpenCV, EasyOCR, PostgreSQL and Streamlit"
)

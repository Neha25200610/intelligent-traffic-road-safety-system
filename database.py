
import psycopg2
import pandas as pd


# =====================================================
# DATABASE CONNECTION
# =====================================================

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="traffic_db",
        user="neha",
        password=""
    )


# =====================================================
# FETCH RED-LIGHT VIOLATIONS
# =====================================================

def get_red_light_violations():

    conn = get_connection()

    try:
        query = """
            SELECT
                id,
                vehicle_id,
                vehicle_type,
                frame_number,
                timestamp_seconds,
                signal_state,
                event,
                created_at
            FROM red_light_violations
            ORDER BY id DESC;
        """

        df = pd.read_sql_query(query, conn)

        return df

    finally:
        conn.close()


# =====================================================
# SAVE RED-LIGHT VIOLATIONS
# =====================================================

def save_red_light_violations(rows):

    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            cursor.execute(
                "DELETE FROM red_light_violations"
            )

            if rows:

                query = """
                    INSERT INTO red_light_violations (
                        vehicle_id,
                        vehicle_type,
                        frame_number,
                        timestamp_seconds,
                        signal_state,
                        event
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                """

                cursor.executemany(query, rows)

        conn.commit()

        print("PostgreSQL records saved successfully.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

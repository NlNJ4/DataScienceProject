import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p, expm1
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

try:
    from pipelines.config import TRAFFY_MERGED_PATH
except ImportError:
    from config import TRAFFY_MERGED_PATH


class Modeler:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("TraffyFondueModeler") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .getOrCreate()

        # base features prepared by FeatureEngineer
        self.base_feature_cols = [
            "coords",
            "timestamp",
            "count_reopen",
            "comment_len",
            "lon",
            "lat",
            "pm25_avg",
            "pm10_avg",
            "dust_avg",
            "rainfall_mm",
            "has_rain",
            "dept_count",
            "district_code",
        ]

        self.feature_cols = None  # จะ set ใน preprocess()

    # ------------------------------------------------------------------
    # 1) LOAD
    # ------------------------------------------------------------------
    def load_data(self):
        print(f"Loading data from {TRAFFY_MERGED_PATH}...")
        if not os.path.exists(TRAFFY_MERGED_PATH):
            print(f"File not found: {TRAFFY_MERGED_PATH}")
            return None

        df = self.spark.read \
            .option("multiLine", "true") \
            .option("escape", "\"") \
            .csv(TRAFFY_MERGED_PATH, header=True, inferSchema=True)

        print(f"Loaded df shape: {df.count()} rows")
        return df

    # ------------------------------------------------------------------
    # 2) PREPROCESS
    # ------------------------------------------------------------------
    def preprocess(self, df):
        """
        - keep rows with non-null resolution_time
        - one-hot columns: all columns starting with 'type_' (จาก FeatureEngineer)
        - rename columns that contain '.' → replace with '_'
        - build feature_cols = base_feature_cols + safe type_* columns
        - cast features + resolution_time to double
        - fillna(0)
        - create resolution_time_log = log1p(resolution_time)
        """
        print("Preprocessing data for modeling...")

        df = df.filter(col("resolution_time").isNotNull())

        # 1) หา type_* columns (one-hot)
        type_cols = [
            c for c in df.columns
            if c.startswith("type_") and c != "type_clean"
        ]
        print(f"Found {len(type_cols)} one-hot type columns: {type_cols}")

        # 2) ทำให้ชื่อ column ปลอดภัย (ไม่มี '.')
        safe_type_cols = []
        for c in type_cols:
            safe_name = c.replace(".", "_")
            if safe_name != c:
                print(f"Renaming column '{c}' -> '{safe_name}'")
                df = df.withColumnRenamed(c, safe_name)
            safe_type_cols.append(safe_name)

        # base features ก็กัน '.' ไว้เหมือนกันเผื่ออนาคต
        safe_base_features = []
        for c in self.base_feature_cols:
            safe_name = c.replace(".", "_")
            if safe_name != c and c in df.columns:
                print(f"Renaming column '{c}' -> '{safe_name}'")
                df = df.withColumnRenamed(c, safe_name)
            safe_base_features.append(safe_name)

        # 3) ตั้ง feature_cols ใหม่ที่ใช้แต่ชื่อปลอดภัย
        self.feature_cols = safe_base_features + safe_type_cols
        print(f"Using {len(self.feature_cols)} feature columns.")
        # print(self.feature_cols)  # uncomment ถ้าอยาก debug

        # 4) เลือก feature + target (ชั่วโมง) พร้อม cast เป็น double
        selected_cols = [
            col(c).cast("double").alias(c) for c in self.feature_cols
        ] + [
            col("resolution_time").cast("double").alias("resolution_time")
        ]

        df_model = df.select(selected_cols)

        # missing numeric -> 0
        df_model = df_model.fillna(0)

        # log1p(target) เก็บเป็นคอลัมน์ใหม่
        df_model = df_model.withColumn(
            "resolution_time_log", log1p(col("resolution_time"))
        )

        return df_model

    # ------------------------------------------------------------------
    # 3) TRAIN
    # ------------------------------------------------------------------
    def train(self, df):
        if not self.feature_cols:
            raise ValueError("feature_cols is not set. Make sure to call preprocess() before train().")

        print("Training model with GBTRegressor...")

        train_data, test_data = df.randomSplit([0.8, 0.2], seed=67)

        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features",
            handleInvalid="skip",
        )

        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="resolution_time_log",
            predictionCol="prediction",
            maxDepth=8,
            maxBins=256,
            maxIter=80,
            seed=67,
        )

        pipeline = Pipeline(stages=[assembler, gbt])

        print("Fitting pipeline...")
        model = pipeline.fit(train_data)

        print("Evaluating model...")
        predictions = model.transform(test_data)

        # แปลงกลับไปเป็นชั่วโมง: expm1(prediction_log)
        predictions = predictions.withColumn(
            "prediction_hours",
            expm1(col("prediction"))
        )

        # ---------- metrics บน log scale ----------
        eval_rmse_log = RegressionEvaluator(
            labelCol="resolution_time_log",
            predictionCol="prediction",
            metricName="rmse",
        )
        eval_mae_log = RegressionEvaluator(
            labelCol="resolution_time_log",
            predictionCol="prediction",
            metricName="mae",
        )
        eval_r2_log = RegressionEvaluator(
            labelCol="resolution_time_log",
            predictionCol="prediction",
            metricName="r2",
        )

        rmse_log = eval_rmse_log.evaluate(predictions)
        mae_log = eval_mae_log.evaluate(predictions)
        r2_log = eval_r2_log.evaluate(predictions)

        # ---------- metrics บน hour scale ----------
        eval_rmse_hr = RegressionEvaluator(
            labelCol="resolution_time",
            predictionCol="prediction_hours",
            metricName="rmse",
        )
        eval_mae_hr = RegressionEvaluator(
            labelCol="resolution_time",
            predictionCol="prediction_hours",
            metricName="mae",
        )
        eval_r2_hr = RegressionEvaluator(
            labelCol="resolution_time",
            predictionCol="prediction_hours",
            metricName="r2",
        )

        rmse_hr = eval_rmse_hr.evaluate(predictions)
        mae_hr = eval_mae_hr.evaluate(predictions)
        r2_hr = eval_r2_hr.evaluate(predictions)

        # ---------- print results ----------
        print("\n=== Evaluation Metrics ===")
        print("Log scale:")
        print(f"  RMSE (log): {rmse_log:.4f}")
        print(f"  MAE  (log): {mae_log:.4f}")
        print(f"  R^2  (log): {r2_log:.4f}")
        print("")
        print("Hour scale (resolution_time in hours):")
        print(f"  RMSE (hours): {rmse_hr:.4f}")
        print(f"  MAE  (hours): {mae_hr:.4f}")
        print(f"  R^2  (hours): {r2_hr:.4f}")
        print("=====================================\n")

        return model


    # ------------------------------------------------------------------
    # 4) SAVE
    # ------------------------------------------------------------------
    def save_model(self, model, path="models/gbt_spark_model2"):
        if not os.path.exists("models"):
            os.makedirs("models", exist_ok=True)

        print(f"Saving model to {path}...")
        model.write().overwrite().save(path)
        print("Model saved successfully!")

    # ------------------------------------------------------------------
    # 5) ORCHESTRATE
    # ------------------------------------------------------------------
    def run(self):
        df = self.load_data()
        if df is not None:
            df_proc = self.preprocess(df)
            model = self.train(df_proc)
            self.save_model(model)

        self.spark.stop()


if __name__ == "__main__":
    modeler = Modeler()
    modeler.run()

import pandas as pd
import numpy as np
import joblib
import warnings
import category_encoders as ce

warnings.filterwarnings("ignore")


class TabularPostprocessor:
    """
    Postprocessor that works with the TabularPreprocessor.
    Handles inverse transformations, predictions processing, and data reconstruction.
    Supports all encoding strategies: 'label', 'onehot', 'binary', 'embedding', 'target', 'frequency', 'hash', or None.
    """

    def __init__(self, preprocessor_path=None, preprocessor_object=None, verbose=True):
        self.verbose = verbose

        # Load serialized preprocessor state
        if preprocessor_path:
            state_bundle = joblib.load(preprocessor_path)
            self.config = state_bundle.get("config", {})
            state = state_bundle.get("state", state_bundle)
            self._print_step(f"Loaded preprocessor from {preprocessor_path}")
        elif preprocessor_object and preprocessor_object.is_fitted_:
            bundle = self._extract_preprocessor_state(preprocessor_object)
            self.config = bundle.get("config", {})
            state = bundle.get("state", bundle)
            self._print_step("Loaded preprocessor from object")
        else:
            raise ValueError(
                "Either preprocessor_path or fitted preprocessor_object must be provided"
            )

        # Extract fields from state
        self.continuous_cols = state.get("continuous_cols_", [])
        self.categorical_cols = state.get("categorical_cols_", [])
        self.binary_cols = state.get("binary_cols_", [])
        self.long_text_cols = state.get("long_text_cols_", [])
        self.outlier_bounds = state.get("outlier_bounds_", {})
        self.pca = state.get("pca_", None)
        self.scaler = state.get("scaler_", None)
        self.imputers = state.get("imputers_", {})
        self.encoder = state.get("encoder_", None)
        self.column_mappings = state.get("column_mappings_", {})
        self.feature_names = state.get("feature_names_", [])
        self.encoding_strategy = self.config.get("encoding_strategy")

        self._print_step("Postprocessor initialized successfully")

    def _print_step(self, message):
        if self.verbose:
            print(f"[TabularPostprocessor] {message}")

    def _extract_preprocessor_state(self, pre):
        config = {
            "cardinality_ratio_threshold": pre.cardinality_ratio_threshold,
            "outlier_iqr_multiplier": pre.outlier_iqr_multiplier,
            "text_length_threshold": pre.text_length_threshold,
            "max_char_threshold": pre.max_char_threshold,
            "n_pca_components": pre.n_pca_components,
            "scaler_type": pre.scaler_type,
            "imputation_strategy": pre.imputation_strategy,
            "encoding_strategy": pre.encoding_strategy,
            "random_state": pre.random_state,
            "verbose": pre.verbose,
        }
        state = {
            "continuous_cols_": pre.continuous_cols_,
            "categorical_cols_": pre.categorical_cols_,
            "binary_cols_": pre.binary_cols_,
            "long_text_cols_": pre.long_text_cols_,
            "outlier_bounds_": pre.outlier_bounds_,
            "pca_": pre.pca_,
            "scaler_": pre.scaler_,
            "imputers_": pre.imputers_,
            "encoder_": getattr(pre, "encoder_", None),
            "column_mappings_": pre.column_mappings_,
            "feature_names_": pre.feature_names_,
        }
        return {"config": config, "state": state}

    def inverse_transform_modified_data(self, modified_df):
        self._print_step("Reconstructing to original feature space...")
        recon = pd.DataFrame(index=modified_df.index)

        # 1. Inverse scale continuous
        cont = [c for c in self.continuous_cols if c in modified_df.columns]
        if cont and self.scaler:
            inv = self.scaler.inverse_transform(modified_df[cont])
            recon[cont] = pd.DataFrame(inv, columns=cont, index=modified_df.index)
            self._print_step("Inverse scaled continuous columns")

        # 2. Inverse PCA to continuous
        if self.pca:
            pca_cols = [c for c in modified_df.columns if c.startswith("PCA_")]
            if pca_cols:
                inv = self.pca.inverse_transform(modified_df[pca_cols].values)
                for i, col in enumerate(self.continuous_cols):
                    if col not in recon:
                        recon[col] = inv[:, i]
                self._print_step("Inverse PCA reconstructed continuous")

        # 3. Inverse encode categoricals (not binaries)
        if self.encoder and self.encoding_strategy in [
            "onehot",
            "ordinal",
            "label",
            "embedding",
        ]:
            if self.encoding_strategy in ["ordinal", "label", "embedding"]:
                # For ordinal/label/embedding encoding
                categorical_cols = self.categorical_cols
                ordinal_cols = [
                    col for col in categorical_cols if col in modified_df.columns
                ]
                if ordinal_cols:
                    arr = modified_df[ordinal_cols].values
                    inv = self.encoder.inverse_transform(arr)
                    recon[ordinal_cols] = pd.DataFrame(
                        inv, columns=ordinal_cols, index=modified_df.index
                    )
                    self._print_step(
                        "Inverse ordinal/label/embedding encoded categoricals"
                    )
            elif self.encoding_strategy == "onehot":
                # For one-hot encoding - FIX: Call without arguments
                expected = list(self.encoder.get_feature_names_out())
                # Ensure all expected encoded cols exist (fill zeros for missing)
                for col in expected:
                    if col not in modified_df.columns:
                        modified_df[col] = 0
                # Get the encoded columns that exist in modified_df
                enc_input_cols = [col for col in expected if col in modified_df.columns]
                if enc_input_cols:
                    inv = self.encoder.inverse_transform(modified_df[enc_input_cols])
                    categorical_cols = self.categorical_cols
                    recon[categorical_cols] = pd.DataFrame(
                        inv, columns=categorical_cols, index=modified_df.index
                    )
                    self._print_step("Inverse one-hot encoded categoricals")
        elif self.encoding_strategy in ["binary", "hash", "frequency", "target"]:
            # Not invertible: just copy encoded columns and warn
            for col in modified_df.columns:
                if col not in recon.columns:
                    recon[col] = modified_df[col]
            self._print_step(
                f"Warning: {self.encoding_strategy} encoding is not invertible. Encoded columns are copied as-is."
            )
        else:
            # direct copy back for binary or no-encoding
            for col in self.categorical_cols + self.binary_cols:
                if col in modified_df:
                    recon[col] = modified_df[col]

        self._print_step(f"Reconstruction complete: {recon.shape}")
        return recon

    def transform_new_data(self, df):
        self._print_step("Transforming new raw data...")
        # 1. Drop long text
        df_clean = df.drop(columns=self.long_text_cols, errors="ignore")

        # 2. Impute missing using stored values (skip ML models)
        df_imp = df_clean.copy()
        for col, info in self.imputers.items():
            if col in df_imp and "value" in info:
                df_imp[col].fillna(info["value"], inplace=True)

        # 3. Outlier flags
        flags = pd.DataFrame(index=df.index)
        for col, (low, high) in self.outlier_bounds.items():
            if col in df_imp:
                vals = pd.to_numeric(df_imp[col], errors="coerce")
                flags[f"{col}_outlier"] = ((vals < low) | (vals > high)).astype(int)

        # 4. Scale continuous
        df_scaled = df_imp.copy()
        if self.scaler and self.continuous_cols:
            df_scaled[self.continuous_cols] = self.scaler.transform(
                df_imp[self.continuous_cols]
            )

        # 5. PCA
        pca_df = pd.DataFrame(index=df.index)
        if self.pca:
            comps = self.pca.transform(df_scaled[self.continuous_cols])
            cols = [f"PCA_{i+1}" for i in range(self.pca.n_components_)]
            pca_df = pd.DataFrame(comps, columns=cols, index=df.index)

        # 6. Encode categoricals (not binaries)
        enc_df = pd.DataFrame(index=df.index)
        if (
            self.encoder
            and self.encoding_strategy in ["onehot", "ordinal"]
            and self.categorical_cols
        ):
            cols_to_encode = [
                col for col in self.categorical_cols if col in df_imp.columns
            ]

            if cols_to_encode:
                arr = self.encoder.transform(df_imp[cols_to_encode])
                if self.encoding_strategy == "onehot":
                    cols = list(self.encoder.get_feature_names_out())
                else:
                    cols = cols_to_encode
                enc_df = pd.DataFrame(arr, columns=cols, index=df.index)

        # 7. Assemble
        pre = pd.concat(
            [
                (
                    df_scaled[self.continuous_cols]
                    if self.continuous_cols
                    else pd.DataFrame(index=df.index)
                ),
                enc_df,
                flags,
                pca_df,
            ],
            axis=1,
        )

        return {
            "preprocessed": pre,
            "continuous": (
                df_scaled[self.continuous_cols]
                if self.continuous_cols
                else pd.DataFrame(index=df.index)
            ),
            "categorical": enc_df,
            "outlier_flags": flags,
            "pca_features": pca_df,
        }

    def get_column_mapping(self):
        return {
            "continuous_columns": self.continuous_cols,
            "categorical_columns": self.categorical_cols,
            "binary_columns": self.binary_cols,
            "dropped_text_columns": self.long_text_cols,
            "outlier_bounds": self.outlier_bounds,
            "feature_names": self.feature_names,
        }

    def get_preprocessing_summary(self):
        summary = {
            "column_classification": {
                "continuous_columns": len(self.continuous_cols),
                "categorical_columns": len(self.categorical_cols),
                "binary_columns": len(self.binary_cols),
                "dropped_text_columns": len(self.long_text_cols),
            },
            "transformations_applied": {
                "encoding": bool(self.encoding_strategy),
                "scaling": bool(self.config.get("scaler_type")),
                "pca": self.pca is not None,
                "outlier_detection": bool(self.outlier_bounds),
                "imputation": bool(self.imputers),
            },
            "output_features": {
                "total_features": len(self.feature_names),
                "continuous_features": len(self.continuous_cols),
                "categorical_features": len(self.categorical_cols)
                + len(self.binary_cols),
                "outlier_flags": len(self.outlier_bounds),
                "pca_components": self.pca.n_components_ if self.pca else 0,
            },
        }
        if self.pca:
            summary["pca_info"] = {
                "n_components": self.pca.n_components_,
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                "total_explained_variance": float(
                    self.pca.explained_variance_ratio_.sum()
                ),
            }
        return summary

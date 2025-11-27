import os
import joblib
import pickle

# Import framework-specific save/load methods
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
except ImportError:
    torch = None


def save_model(model, filepath: str):
    ext = os.path.splitext(filepath)[1].lower()

    # Scikit-learn & pickle
    if ext in {".pkl", ".pickle", ".sav", ".joblib"}:
        try:
            joblib.dump(model, filepath)
        except Exception:
            with open(filepath, "wb") as f:
                pickle.dump(model, f)
        return

    # XGBoost
    if xgb is not None and isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
        if ext in {".json", ".bin", ".model"}:
            model.save_model(filepath)
            return
        raise ValueError(
            "Unsupported file extension for XGBoost model. Use .json, .bin or .model"
        )

    # LightGBM
    if lgb is not None and isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
        if ext in {".txt", ".model"}:
            model.booster_.save_model(filepath)
            return
        raise ValueError(
            "Unsupported file extension for LightGBM model. Use .txt or .model"
        )

    # CatBoost
    if (
        CatBoostClassifier is not None
        and isinstance(model, (CatBoostClassifier, CatBoostRegressor))
        and ext == ".cbm"
    ):
        model.save_model(filepath)
        return

    # TensorFlow / Keras
    if tf is not None:
        if hasattr(model, "save"):
            model.save(filepath)
            return

    # PyTorch
    if torch is not None:
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), filepath)
            return

    raise ValueError("Unsupported model type or file extension for saving.")


def load_model(filepath: str, model_type=None):
    ext = os.path.splitext(filepath)[1].lower()

    # Scikit-learn / pickle
    if ext in {".pkl", ".pickle", ".sav", ".joblib"}:
        try:
            return joblib.load(filepath)
        except Exception:
            with open(filepath, "rb") as f:
                return pickle.load(f)

    # XGBoost
    if xgb is not None and ext in {".json", ".bin", ".model"}:
        model = xgb.Booster()
        model.load_model(filepath)
        return model

    # LightGBM
    if lgb is not None and ext in {".txt", ".model"}:
        return lgb.Booster(model_file=filepath)

    # CatBoost
    if CatBoostClassifier is not None and ext == ".cbm":
        if model_type == "classifier":
            model = CatBoostClassifier()
        elif model_type == "regressor":
            model = CatBoostRegressor()
        else:
            raise ValueError("Specify 'model_type' as 'classifier' or 'regressor' for CatBoost")
        model.load_model(filepath)
        return model

    # TensorFlow / Keras
    if tf is not None:
        try:
            return tf.keras.models.load_model(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load TensorFlow/Keras model: {e}")

    # PyTorch
    if torch is not None:
        if model_type is None:
            raise ValueError("You must specify 'model_type' (torch.nn.Module subclass instance) to load PyTorch model")
        if not isinstance(model_type, torch.nn.Module):
            raise ValueError("'model_type' must be an instance of torch.nn.Module")
        model = model_type
        model.load_state_dict(torch.load(filepath))
        model.eval()
        return model

    raise ValueError("Unsupported or unknown model file format for loading.")

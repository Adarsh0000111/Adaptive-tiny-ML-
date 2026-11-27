import os
import logging
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from scipy.special import softmax

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class KnowledgeDistiller:
    """
    Knowledge Distillation pipeline:
    Transfers dark knowledge & decision boundaries from the high-tier 1D CNN teacher
    into a lightweight TinyML Decision Tree student.
    """
    def __init__(self, temperature=2.0, student_max_depth=5):
        self.temperature = temperature
        self.student_max_depth = student_max_depth
        self.student_model = None

    def distill(self, teacher_model, X_train, y_train, X_val, y_val):
        """
        Train Student Decision Tree on Teacher Soft Probabilities.
        """
        logger.info(f"Starting Knowledge Distillation (Temperature T={self.temperature}, Max Depth={self.student_max_depth})...")
        
        # 1. Generate soft teacher predictions
        if hasattr(teacher_model, "predict_proba"):
            teacher_logits = teacher_model.predict_proba(X_train)
        else:
            teacher_logits = teacher_model.predict(X_train)

        # Apply Temperature scaling to probabilities
        soft_labels = softmax(teacher_logits / self.temperature, axis=1)
        pseudo_labels = np.argmax(soft_labels, axis=1)

        # 2. Train student Decision Tree on soft pseudo-labels
        self.student_model = DecisionTreeClassifier(
            max_depth=self.student_max_depth,
            criterion="gini",
            random_state=42
        )
        self.student_model.fit(X_train, pseudo_labels)

        # 3. Evaluate Student Accuracy vs Hard y_val
        y_val_pred = self.student_model.predict(X_val)
        student_acc = np.mean(y_val_pred == y_val)
        
        # Evaluate Teacher Accuracy
        teacher_val_pred = teacher_model.predict(X_val)
        if len(teacher_val_pred.shape) > 1:
            teacher_val_pred = np.argmax(teacher_val_pred, axis=1)
        teacher_acc = np.mean(teacher_val_pred == y_val)

        logger.info(f"Knowledge Distillation Complete!")
        logger.info(f"Teacher 1D CNN Accuracy: {teacher_acc * 100:.2f}% | Distilled Student Tree Accuracy: {student_acc * 100:.2f}%")

        return {
            "teacher_acc": round(float(teacher_acc), 4),
            "distilled_student_acc": round(float(student_acc), 4),
            "temperature": self.temperature,
            "max_depth": self.student_max_depth
        }

    def save_student(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.student_model, output_path)
        logger.info(f"Distilled Student Model saved to {output_path}")

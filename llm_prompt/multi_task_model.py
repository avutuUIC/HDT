# HDT/WESAD/cognition_model/multi_task_model.py

import pandas as pd

class MultiTaskModel:
    def __init__(self, clf_model, reg1_model, reg2_model, label_encoder):
        self.clf = clf_model
        self.reg1 = reg1_model
        self.reg2 = reg2_model
        self.encoder = label_encoder

    def fit(self, X, y_focus, y_acc, y_dec):
        self.clf.fit(X, y_focus)
        self.reg1.fit(X, y_acc)
        self.reg2.fit(X, y_dec)

    def predict(self, X):
        focus_pred = self.clf.predict(X)
        focus_label = self.encoder.inverse_transform(focus_pred)

        acc_pred = self.reg1.predict(X)
        dec_pred = self.reg2.predict(X)

        focus_map = {
            'low': 'low',
            'medium': 'normal',
            'high': 'hyper'
        }
        mapped_focus = [focus_map.get(label, label) for label in focus_label]

        def interpret_decision_score(score):
            if score >= 0.00003:
                return "fast"
            elif score >= 0.000015:
                return "average"
            else:
                return "hesitant"

        interpreted_decisions = [interpret_decision_score(d) for d in dec_pred]
        reaction_times = X['time_on_task'].tolist() if 'time_on_task' in X.columns else [None] * len(X)

        return pd.DataFrame({
            "focus_level": mapped_focus,
            "decision_making": interpreted_decisions,
            "reaction_time": reaction_times,
            "accuracy": [round(a, 3) for a in acc_pred]
        })

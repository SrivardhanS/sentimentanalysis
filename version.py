import joblib
model = joblib.load("ticket_classifier.pkl")
joblib.dump(model, "ticket_classifier_v2.pkl")

import predictor as object_detector_app

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "predictor" above to the
# new file.

app = object_detector_app.app

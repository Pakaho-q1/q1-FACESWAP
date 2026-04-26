def run_detect(frame, model_manager):
    """Find all faces in the frame with InsightFace"""
    faces = model_manager.detect_faces(frame)
    return faces
